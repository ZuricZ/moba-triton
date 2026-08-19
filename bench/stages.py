"""Times each stage of the MoBA forward separately.

The design's central risk is that measuring query-major utilization at 0.121
proves query-major *cannot* win without proving the inverted kernel *does* --
because index construction, the gathered reads and the combine pass are all
unpriced. This prices them.

Four stages, not three. The fourth is buffer allocation, which the traffic
accounting in the design notes overlooked: `part_acc` is allocated with
`torch.zeros`, and zeroing 24 MB writes 24 MB. That is the same order as the
attention kernel's entire K/V traffic, so it is measured rather than folded
into the attention number.

D11 keeps the gate in PyTorch on the explicit argument that it should be
measured before being optimized. This is that measurement: if `index` exceeds
roughly a fifth of the total, the deferral trigger has fired.

The bfloat16 arm tests the other open question. `part_acc` is the single
largest term in the design's memory traffic -- at S=4096, head_dim 128,
top_k 12 it is 24 MB written plus 24 MB read per batch-head against 15 MB for
all of Q, K and V together. Halving it is worth measuring on both axes at
once, so the accuracy cost is reported beside the speed.
"""

import torch
import triton.testing

from moba.config import MoBAConfig
from moba.device import BYTES_PER_MIB, MICROSECONDS_PER_MILLISECOND
from moba.kernels.index import build_csc
from moba.kernels.moba_fwd import (
    attend_partials,
    combine_partials,
    empty_partials,
    moba_forward,
)

# Realistic gate, matching what the references benchmark: FlashMoBA sweeps
# block_size 128 with top_k 8, MoonshotAI uses chunk 512/4096 with top_k 3/12.
# Sparsity comes from `num_blocks = S / block_size` growing with context, not
# from shrinking the block.
DEFAULT_GATE = (128, 8)

# head_dim 128 is what both reference implementations use, and it is the case
# where `part_acc` is largest, so it is where the storage question bites.
# The seq_len range matters more than any single point: the gate's cost is
# expected to be flat in S (launch-bound) while the kernel's grows linearly,
# so their ratio is only meaningful as a trend.
DEFAULT_SHAPES = [(1, 4, s, 128) for s in (1024, 2048, 4096, 8192)]

# Share of total runtime above which D11's "optimize the gate" trigger fires.
INDEX_FRACTION_TRIGGER = 0.20

# `do_bench` runs per measurement, of which the minimum is kept. This machine
# runs WDDM, so the GPU is shared with the desktop compositor and submissions
# are batched by the driver rather than dispatched immediately. Repeated
# measurements of the same call were observed to spread 1.4x. The minimum is
# the run least polluted by outside interference; a mean would average over how
# busy the desktop happened to be. Ratios survive this noise, which is what
# this benchmark reports; absolute numbers from this box should not be quoted.
BENCH_REPEATS = 5


def _best(fn) -> float:
    """Times `fn` several times and keeps the least-disturbed run.

    Args:
        fn: Zero-argument callable to time.

    Returns:
        Minimum observed latency in milliseconds.
    """
    return min(triton.testing.do_bench(fn) for _ in range(BENCH_REPEATS))


def time_stages(
    shape: tuple[int, int, int, int],
    gate: tuple[int, int] = DEFAULT_GATE,
    dtype: torch.dtype = torch.bfloat16,
    partials_dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> dict:
    """Times index build, attention and combine for one configuration.

    Every kernel buffer is allocated before the timed regions begin. An earlier
    version allocated inside them and produced nonsense -- `index` timed at
    120% of `total` -- because on WDDM a `cudaMalloc` goes through the display
    driver and costs milliseconds. What that measured was the Windows memory
    manager, not the kernel.

    `index` is still an over-estimate for the same reason: its allocations
    cannot be hoisted, since `nonzero()` returns a data-dependent size that the
    caching allocator cannot reuse. It is an upper bound on the gate's cost.

    Args:
        shape: (batch, heads, seq_len, head_dim).
        gate: (block_size, top_k).
        dtype: Compute dtype for q, k, v.
        partials_dtype: Storage dtype for `part_acc`.
        seed: RNG seed.

    Returns:
        A dict of per-stage milliseconds, derived shares, and the accuracy of
        this configuration against a float32-partials run.
    """
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=dtype)
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (torch.randn(shape, generator=g, device="cuda", dtype=dtype)
               for _ in range(3))

    index = build_csc(q, k, cfg)
    slots = index.max_slots
    buffers = empty_partials(shape, slots, q.device, partials_dtype)
    part_m, part_l, part_acc = attend_partials(
        q, k, v, cfg, index, partials=buffers
    )

    t_index = _best(lambda: build_csc(q, k, cfg))
    t_attn = _best(
        lambda: attend_partials(q, k, v, cfg, index, partials=buffers)
    )
    t_combine = _best(lambda: combine_partials(part_m, part_l, part_acc, dtype))

    # Accuracy of this storage choice against float32 partials, which is what
    # the correctness suite validated.
    truth = moba_forward(q, k, v, cfg, index=index,
                         partials_dtype=torch.float32).float()
    got = moba_forward(q, k, v, cfg, index=index,
                       partials_dtype=partials_dtype).float()
    rel = ((got - truth).abs().max() / truth.abs().max()).item()

    acc_bytes = part_acc.numel() * part_acc.element_size()
    kernel_ms = t_attn + t_combine
    total_ms = t_index + kernel_ms
    del q, k, v, buffers, part_m, part_l, part_acc, index, truth, got
    torch.cuda.empty_cache()

    return {
        "seq_len": shape[2],
        "head_dim": shape[3],
        "block_size": block_size,
        "top_k": top_k,
        "slots": slots,
        "partials_dtype": str(partials_dtype).replace("torch.", ""),
        "part_acc_mb": acc_bytes / BYTES_PER_MIB,
        "index_ms": t_index,
        "attn_ms": t_attn,
        "combine_ms": t_combine,
        "kernel_ms": kernel_ms,
        "total_ms": total_ms,
        "index_fraction": t_index / total_ms,
        "combine_fraction": t_combine / kernel_ms,
        "rel_err_vs_fp32": rel,
    }


def run(
    shapes: list[tuple[int, int, int, int]] | None = None,
    gate: tuple[int, int] = DEFAULT_GATE,
    partials_dtypes: tuple[torch.dtype, ...] = (torch.float32, torch.bfloat16),
) -> list[dict]:
    """Sweeps shapes and partial-storage dtypes.

    Args:
        shapes: Shapes to measure; `DEFAULT_SHAPES` when omitted.
        gate: (block_size, top_k).
        partials_dtypes: Storage dtypes for `part_acc` to compare.

    Returns:
        A list of records from `time_stages`.
    """
    return [
        time_stages(shape, gate, partials_dtype=pd)
        for shape in (shapes or DEFAULT_SHAPES)
        for pd in partials_dtypes
    ]


def to_markdown(rows: list[dict]) -> str:
    """Renders the sweep as a markdown table.

    Args:
        rows: Records from `run`.

    Returns:
        A markdown table string.
    """
    head = ("| S | part_acc | size | gate (index) | attn | combine | kernel | "
            "gate share | combine share of kernel | rel err |")
    sep = "|" + "---:|" * 10
    us = MICROSECONDS_PER_MILLISECOND
    body = [
        f"| {r['seq_len']} | {r['partials_dtype']} | {r['part_acc_mb']:.0f} MB "
        f"| {r['index_ms'] * us:.0f} us | {r['attn_ms'] * us:.0f} us "
        f"| {r['combine_ms'] * us:.0f} us | {r['kernel_ms'] * us:.0f} us "
        f"| {r['index_fraction']:.0%} | {r['combine_fraction']:.0%} "
        f"| {r['rel_err_vs_fp32']:.1e} |"
        for r in rows
    ]
    return "\n".join([head, sep, *body])


if __name__ == "__main__":
    import json
    import sys

    rows = run()
    if "--json" in sys.argv:
        print(json.dumps(rows))
    else:
        print(to_markdown(rows))
        worst = max(r["index_fraction"] for r in rows)
        if worst > INDEX_FRACTION_TRIGGER:
            print(f"\nD11 trigger FIRED: index is {worst:.0%} of total, above "
                  f"the {INDEX_FRACTION_TRIGGER:.0%} threshold. The PyTorch "
                  "gate needs a Triton replacement.")
        else:
            print(f"\nD11 trigger not fired: index peaks at {worst:.0%}, "
                  f"below {INDEX_FRACTION_TRIGGER:.0%}.")
