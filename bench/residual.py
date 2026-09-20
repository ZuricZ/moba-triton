"""What the FlashMoBA gap is made of, beyond the tiles we already counted.

WHY THIS EXISTS. `bench/tile_census.py` counts the tensor-core tiles each
design issues for the same selection and finds FlashMoBA issuing 1.24x ours at
S=8192, rising to 2.26x at S=65536. The measured gap is 2.46x and 4.55x. So
roughly **2x is unaccounted for**, and it barely moves across a 64x range of
sequence length. That flatness is the only evidence we have, and it is not
enough to pick between the candidates.

    S        tiles counted   measured   unexplained
    1024        1.00x         2.79x       2.79x
    8192        1.24x         2.46x       1.98x
    65536       2.26x         4.55x       2.01x

WHAT EACH CANDIDATE PREDICTS. Three survive the flatness argument, and this
module is built to separate them rather than to confirm any one.

* **Their gate.** Three launches, a host sync per call
  (`triton_mean_pool.py:140`), 67-134 MB of `torch.zeros` per call. Panel A
  times gate and attention apart, so this is answered outright: if their
  attention alone is still ~2x ours, the gate is not the story.
* **Under-subscription.** Their grid is `(ceil(S / lg_block_m), batch, heads)`
  (`flash_fwd_launch_template.h:52`). At the shape every result here uses --
  B=1, H=4, S=8192 -- that is **64 programs on 108 SMs**, so 40% of the card
  is idle before the kernel starts. Ours launches one program per KV block:
  256. Panel B raises `batch * heads` until both oversubscribe. If the
  residual collapses, this was it.
* **Per-SM inefficiency** -- occupancy or the fp32 `o_tmp` round trip. These
  are what is *left* if Panel B changes nothing, and Panel C says whether
  either implementation is anywhere near the bandwidth roof, which decides
  between them.

WHY NOT SWEEP head_dim. It was the obvious discriminator and it does not work:
every term in `traffic_flash_moba` scales with `head_dim`, so `o_tmp`'s share
of their bytes is invariant in it. A sweep would separate nothing.

RUN IT
    uv run python -m bench.residual                 # everything, ~15 min
    uv run python -m bench.residual --panel a       # one panel
    uv run python -m bench.residual --seq-lens 8192
"""

import argparse
import json
import os

import torch

try:                                    # `do_bench` moved between versions
    from triton.testing import do_bench
except ImportError:                                  # pragma: no cover
    from triton.testing import do_bench_cudagraph as do_bench

from bench.roofline import traffic_flash_moba, traffic_ours
from bench.suite import cool_down_device, warm_up_device
from bench.tile_census import decide_lg_block_m
from implementations.layout import cu_seqlens_for, to_varlen
from moba.config import MoBAConfig
from moba.kernels.moba_fwd import moba_forward
from moba.kernels.routing import build_csc_fused

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BLOCK_SIZE, TOP_K = 128, 8
DTYPE = torch.bfloat16
RESULTS = "results"

# The shape every other result in this repository uses, so panel A's numbers
# sit beside them without a translation note.
SWEEP_SHAPE = (1, 4, 64)                             # batch, heads, head_dim

# Panel B: the same work per program, progressively more programs. The third
# row oversubscribes 108 SMs for both designs, which is the point.
PARALLELISM = (
    (1, 4, 64, "the sweep shape"),
    (1, 32, 128, "GPT-3 13B-ish"),
    (4, 32, 128, "same, batch 4"),
)

# A100-PCIE-40GB, measured rather than datasheet (`docs/a100-results.md`).
A100_BANDWIDTH_GIBS = 1239.0


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def time_ms(fn) -> float | None:
    """Returns the median milliseconds for `fn`, or None if it refuses.

    A backend that cannot run a shape is data, not an error: the caller prints
    the blank and carries on. Absence has been a finding twice in this project
    already -- FlashMoBA below block_size 64, and the compile ceiling.

    Args:
        fn: Zero-argument callable to time.

    Returns:
        Median milliseconds, or None if the call raised.
    """
    try:
        fn()
        torch.cuda.synchronize()
    except Exception as exc:                    # noqa: BLE001 - third party
        print(f"      (refused: {type(exc).__name__}: {str(exc)[:70]})")
        return None
    cool_down_device()
    warm_up_device()
    return min(do_bench(fn) for _ in range(3))


def make(batch, seq_len, heads, head_dim):
    """Returns q, k, v for one shape, from a fixed seed."""
    g = torch.Generator(device="cuda").manual_seed(0)
    return [torch.randn(batch, seq_len, heads, head_dim, generator=g,
                        device="cuda", dtype=DTYPE) for _ in range(3)]


# ---------------------------------------------------------------------------
# The two pipelines, split into gate and attention
# ---------------------------------------------------------------------------


def flash_moba_parts(q, k, v, cfg):
    """Returns (gate, attend, whole) callables for FlashMoBA.

    Their pipeline composes two public entry points, and
    `flash_moba_varlen_func` is literally the composition -- it calls
    `flash_topk_varlen_func` then `flash_moba_attn_varlen_func` with
    `decide_lg_block_m`'s answer (`flash_moba_interface.py:695-727`). So
    timing the halves separately measures their own stages rather than a
    rearrangement of them.

    Args:
        q: Queries [B, S, H, D].
        k: Keys, same shape.
        v: Values, same shape.
        cfg: Gate configuration.

    Returns:
        Tuple of (gate, attend, whole). `gate` and `whole` take no arguments;
        `attend` takes an optional query-window override, which is what panel
        D sweeps -- their own policy picks it, but the kernel takes it as a
        parameter, so a smaller one can be asked for.
    """
    from flash_moba import (                         # type: ignore
        flash_moba_attn_varlen_func,
        flash_topk_varlen_func,
        flash_moba_varlen_func,
    )

    b, s = q.shape[:2]
    cu = cu_seqlens_for(b, s, q.device)
    qv, kv_, vv = to_varlen(q), to_varlen(k), to_varlen(v)

    def gate():
        return flash_topk_varlen_func(qv, kv_, cu, cu, s, s,
                                      cfg.top_k, cfg.block_size, causal=True)

    offsets, nnz, indices = gate()
    lg_block_m = decide_lg_block_m(cfg.top_k, cfg.block_size, s, causal=True)

    def attend(window: int | None = None):
        return flash_moba_attn_varlen_func(
            qv, kv_, vv, cu, cu, s, s,
            offsets, nnz, indices,
            window or lg_block_m, cfg.block_size,
            dropout_p=0.0, causal=True,
        )

    def whole():
        return flash_moba_varlen_func(qv, kv_, vv, cu, cu, s, s,
                                      cfg.block_size, cfg.top_k, causal=True)

    return gate, attend, whole


def ours_parts(q, k, v, cfg):
    """Returns (gate, attend, whole) callables for this project's kernel.

    `moba_forward` builds the index itself when none is passed, so handing it
    one measures the attend and combine kernels with routing held fixed --
    the same split `flash_moba_parts` takes, on the same side of the same
    boundary.

    Args:
        q: Queries [B, S, H, D].
        k: Keys, same shape.
        v: Values, same shape.
        cfg: Gate configuration.

    Returns:
        Tuple of three zero-argument callables, as `flash_moba_parts`.
    """
    def gate():
        return build_csc_fused(q, k, cfg)

    index = gate()

    def attend():
        return moba_forward(q, k, v, cfg, index=index)

    def whole():
        return moba_forward(q, k, v, cfg)

    return gate, attend, whole


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def panel_a(seq_lens) -> list[dict]:
    """Gate against attention, both implementations, at the sweep shape.

    Answers the gate candidate outright. The number that matters is the
    attention-only ratio: if it is still near 2x after the tile count is
    divided out, their gate is not what the residual is made of.

    Args:
        seq_lens: Sequence lengths to cover.

    Returns:
        One row dict per sequence length.
    """
    b, h, d = SWEEP_SHAPE
    cfg = MoBAConfig(block_size=BLOCK_SIZE, top_k=TOP_K, dtype=DTYPE)
    rows = []
    print(f"\n{'=' * 78}\nPANEL A -- gate against attention  "
          f"(B={b} H={h} D={d} block={BLOCK_SIZE} top_k={TOP_K})\n{'=' * 78}")
    print(f"{'S':>7} {'':>3} {'gate':>9} {'attend':>9} {'whole':>9} "
          f"{'gate share':>11}")

    for seq_len in seq_lens:
        q, k, v = make(b, seq_len, h, d)
        row = {"seq_len": seq_len}
        for name, parts in (("ours", ours_parts),
                            ("theirs", flash_moba_parts)):
            try:
                gate, attend, whole = parts(q, k, v, cfg)
            except Exception as exc:                 # noqa: BLE001
                print(f"{seq_len:>7} {name:>6}   unavailable: "
                      f"{type(exc).__name__}: {str(exc)[:50]}")
                continue
            g, a, w = time_ms(gate), time_ms(attend), time_ms(whole)
            row[name] = {"gate": g, "attend": a, "whole": w}
            share = f"{g / w:>10.1%}" if g and w else f"{'-':>11}"
            print(f"{seq_len:>7} {name:>6} "
                  f"{_ms(g)} {_ms(a)} {_ms(w)} {share}")
        _report_ratio(row, seq_len)
        rows.append(row)
        del q, k, v
        torch.cuda.empty_cache()
    return rows


def _ms(value) -> str:
    return f"{value:>9.3f}" if value is not None else f"{'-':>9}"


def _report_ratio(row, seq_len) -> None:
    """Prints the whole-pipeline and attention-only ratios side by side."""
    ours, theirs = row.get("ours"), row.get("theirs")
    if not (ours and theirs):
        return
    for label, key in (("whole pipeline", "whole"), ("attention only",
                                                     "attend")):
        a, b_ = ours.get(key), theirs.get(key)
        if a and b_:
            print(f"{'':>7} {'':>6} {label:>28}: theirs / ours = "
                  f"{b_ / a:.2f}x")


def panel_b(seq_lens) -> list[dict]:
    """More programs, until both designs oversubscribe the GPU.

    Their grid is one program per (query window, batch, head); ours is one per
    (KV block, batch, kv head). At B=1 H=4 that is 64 against 256 on 108 SMs,
    so they start 40% idle and we do not. Raising `batch * heads` removes the
    difference. If the residual tracks it down, under-subscription was the
    cause and nothing per-SM needs invoking.

    Args:
        seq_lens: Sequence lengths to cover.

    Returns:
        One row dict per (shape, sequence length).
    """
    cfg = MoBAConfig(block_size=BLOCK_SIZE, top_k=TOP_K, dtype=DTYPE)
    rows = []
    print(f"\n{'=' * 78}\nPANEL B -- does the gap survive a busy GPU?"
          f"\n{'=' * 78}")
    print(f"{'B':>3} {'H':>4} {'D':>4} {'S':>7} | {'their progs':>11} "
          f"{'ours':>7} | {'ours ms':>9} {'theirs ms':>10} {'ratio':>7}")

    for batch, heads, head_dim, label in PARALLELISM:
        for seq_len in seq_lens:
            n_theirs = (-(-seq_len // decide_lg_block_m(
                TOP_K, BLOCK_SIZE, seq_len, causal=True)) * batch * heads)
            n_ours = (seq_len // BLOCK_SIZE) * batch * heads
            try:
                q, k, v = make(batch, seq_len, heads, head_dim)
            except torch.OutOfMemoryError:
                print(f"{batch:>3} {heads:>4} {head_dim:>4} {seq_len:>7} | "
                      f"{n_theirs:>11,} {n_ours:>7,} |   OOM building inputs")
                continue

            ours_ms = time_ms(ours_parts(q, k, v, cfg)[2])
            try:
                theirs_ms = time_ms(flash_moba_parts(q, k, v, cfg)[2])
            except Exception as exc:                 # noqa: BLE001
                print(f"      (theirs unavailable: {type(exc).__name__})")
                theirs_ms = None
            ratio = (f"{theirs_ms / ours_ms:>6.2f}x"
                     if ours_ms and theirs_ms else f"{'-':>7}")
            print(f"{batch:>3} {heads:>4} {head_dim:>4} {seq_len:>7} | "
                  f"{n_theirs:>11,} {n_ours:>7,} | {_ms(ours_ms)} "
                  f"{_ms(theirs_ms):>10} {ratio}   {label}")
            rows.append({"batch": batch, "heads": heads, "head_dim": head_dim,
                         "seq_len": seq_len, "programs_theirs": n_theirs,
                         "programs_ours": n_ours, "ours_ms": ours_ms,
                         "theirs_ms": theirs_ms})
            del q, k, v
            torch.cuda.empty_cache()
    return rows


def panel_c(rows_a) -> list[dict]:
    """Achieved bandwidth against the measured roof, from panel A's timings.

    Decides between the two per-SM candidates without a profiler. If either
    implementation sits near 1239 GiB/s then traffic -- the fp32 `o_tmp` round
    trip being 37% of their modelled bytes -- is binding and occupancy is not.
    If both sit far below it, traffic cannot be the constraint and latency is.

    The traffic is modelled, not measured (`docs/traffic-derivation.md`), so
    this bounds rather than settles. `ncu` would measure it and has been
    refused on every machine this project has touched.

    Args:
        rows_a: Rows as `panel_a` returns.

    Returns:
        One row dict per (implementation, sequence length).
    """
    b, h, d = SWEEP_SHAPE
    cfg = MoBAConfig(block_size=BLOCK_SIZE, top_k=TOP_K, dtype=DTYPE)
    out = []
    print(f"\n{'=' * 78}\nPANEL C -- modelled bytes over measured time"
          f"\n{'=' * 78}")
    print(f"{'S':>7} {'':>6} {'modelled MiB':>13} {'attend ms':>10} "
          f"{'GiB/s':>9} {'% of roof':>10}")

    for row in rows_a:
        seq_len = row["seq_len"]
        for name, model in (("ours", traffic_ours),
                            ("theirs", traffic_flash_moba)):
            cell = row.get(name)
            if not cell or not cell.get("attend"):
                continue
            total = model(b, seq_len, h, d, cfg).total
            ms = cell["attend"]
            gibs = total / (ms * 1e-3) / (1 << 30)
            print(f"{seq_len:>7} {name:>6} {total / (1 << 20):>13.1f} "
                  f"{ms:>10.3f} {gibs:>9.1f} "
                  f"{gibs / A100_BANDWIDTH_GIBS:>9.1%}")
            out.append({"seq_len": seq_len, "impl": name, "bytes": total,
                        "attend_ms": ms, "gibs": gibs})
    return out


def panel_d(seq_lens) -> list[dict]:
    """Their attention at query windows narrower than their policy picks.

    This is the H100 question, asked directly. `SmemLayoutMobaRowStats` costs
    32 bytes of shared memory per query in the window (`kernel_traits.h`), so
    the window sets the footprint, and the footprint sets how many of their
    programs fit on an SM:

        H100  228 KB per SM  /  ~100 KiB  ->  2 programs
        A100  164 KB per SM  /  ~100 KiB  ->  1 program

    They tuned to just fit an H100 and an A100 has ~30% less, which is exactly
    enough to halve their residency. They already know the failure mode --
    `decide_lg_block_m` caps the window at 512 on sm_86/sm_89 with the comment
    "[A6000 Fix] Use 512 cap ... to avoid Shared Memory OOM" -- but sm_80 is
    not in that branch, so the A100 keeps the full 1024.

    A narrower window costs them tiles: fewer queries to fill a 64-row tile
    from, which `bench/tile_census.py` counts. So this trades padding against
    residency, and the shape of the curve is the answer:

    * **Narrower is faster** -- the footprint is binding on this card, their
      own policy is mistuned for sm_80, and the residual is largely an artefact
      of measuring an H100 kernel on an A100. That has to go in the write-up.
    * **Narrower is slower, monotonically** -- the footprint is not binding,
      their policy is right, and the residual is something else.

    Args:
        seq_lens: Sequence lengths to cover.

    Returns:
        One row dict per (sequence length, window).
    """
    b, h, d = SWEEP_SHAPE
    cfg = MoBAConfig(block_size=BLOCK_SIZE, top_k=TOP_K, dtype=DTYPE)
    rows = []
    print()
    print("=" * 78)
    print("PANEL D -- their window against their own policy")
    print("=" * 78)
    print(f"{'S':>7} {'window':>7} {'':>9} {'attend ms':>10} {'vs policy':>10}")

    for seq_len in seq_lens:
        q, k, v = make(b, seq_len, h, d)
        try:
            _, attend, _ = flash_moba_parts(q, k, v, cfg)
        except Exception as exc:                     # noqa: BLE001
            print(f"{seq_len:>7}   unavailable: {type(exc).__name__}")
            del q, k, v
            continue

        chosen = decide_lg_block_m(TOP_K, BLOCK_SIZE, seq_len, causal=True)
        baseline = None
        for window in (128, 256, 512, 768, 1024):
            if window > seq_len:
                continue
            ms = time_ms(lambda w=window: attend(w))
            if window == chosen:
                baseline = ms
            mark = "  <- their policy" if window == chosen else ""
            rel = (f"{ms / baseline:>9.2f}x"
                   if ms and baseline else f"{'-':>10}")
            print(f"{seq_len:>7} {window:>7} {'':>9} {_ms(ms):>10} "
                  f"{rel}{mark}")
            rows.append({"seq_len": seq_len, "window": window, "ms": ms,
                         "is_policy": window == chosen})
        del q, k, v
        torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", default="8192,16384,65536",
                        help="comma separated")
    parser.add_argument("--panel", default="abcd",
                        help="any of a, b, c, d; default all four")
    parser.add_argument("--out", default=f"{RESULTS}/a100-residual.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device; this measures FlashMoBA and needs the A100")
        return

    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    print(f"{torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).multi_processor_count} SMs")
    print(f"sequence lengths: {seq_lens}")

    data = {"device": torch.cuda.get_device_name(0), "seq_lens": seq_lens}
    rows_a = []
    if "a" in args.panel:
        rows_a = panel_a(seq_lens)
        data["panel_a"] = rows_a
    if "b" in args.panel:
        data["panel_b"] = panel_b(seq_lens)
    if "c" in args.panel:
        if not rows_a:
            rows_a = panel_a(seq_lens)
        data["panel_c"] = panel_c(rows_a)
    if "d" in args.panel:
        data["panel_d"] = panel_d(seq_lens)

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
