"""What load imbalance across KV blocks actually costs, once scheduling counts.

Settles design question B. `docs/divergence-measurements.md` Finding 3 reports
6.2x max/mean consumers per KV block, which looks alarming. This measures what
that skew costs *after* the GPU's scheduler has had a chance to absorb it.

Work model. A program owns one KV block, loads it once, then processes its
consumers in tiles of GROUP queries. A block with 5 consumers still pays for a
whole tile, so work is counted in tiles:

    tiles(block) = ceil(consumers / GROUP)

Tiling is not an artifact of the model -- it is what the kernel does for
tensor-core reasons -- and it damps the skew for free, since `ceil` compresses
the distribution.

Scheduling model. GPUs dispatch programs to processors as they free up, which
is greedy list scheduling in dispatch order. Four orders are compared:

    ascending   shortest first -- adversarial: the longest job starts last
    random      no ordering    -- the default
    descending  longest first  -- LPT, the cheap fix under consideration
    chunked     every tile its own program -- perfect balance by construction,
                the expensive fix, included to show what it would buy

Limitations, all of which make this the *pessimistic* reading:
  - fixed per-program cost (loading the KV block) is ignored; including it adds
    a constant to every job and reduces relative imbalance
  - bandwidth contention between concurrent programs is ignored
  - greedy list scheduling only approximates the hardware dispatcher
"""

import math

import torch

from moba.config import MoBAConfig
from moba.device import multiprocessor_count
from moba.kernels.tuning import choose_group
from moba.ref.naive import moba_selection

# Resident programs per SM. Occupancy is configuration-dependent; 2 is what
# learning note 02 measured for a comparable tile at head_dim 64. Passed to
# `run` so it can be varied rather than assumed.
DEFAULT_PROGRAMS_PER_SM = 2


def makespan(jobs: list[int], n_proc: int) -> float:
    """Greedy list scheduling: each job goes to the earliest-free processor.

    Args:
        jobs: Job costs, in dispatch order.
        n_proc: Number of concurrent processors.

    Returns:
        Time at which the last processor finishes.
    """
    finish = [0.0] * n_proc
    for j in jobs:
        i = min(range(n_proc), key=finish.__getitem__)
        finish[i] += j
    return max(finish)


def analyse(
    seq_len: int,
    block_size: int,
    top_k: int,
    n_proc: int,
    heads: int = 8,
    head_dim: int = 128,
    seed: int = 0,
) -> dict:
    """Measures the scheduling penalty for one configuration.

    Args:
        seq_len: Sequence length.
        block_size: MoBA block size.
        top_k: Gate budget, inclusive convention.
        n_proc: Concurrent processors (SMs x resident programs per SM).
        heads: Number of attention heads.
        head_dim: Head dimension.
        seed: RNG seed.

    Returns:
        A flat dict of job statistics and makespan ratios.
    """
    cfg = MoBAConfig(block_size=block_size, top_k=top_k)
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k = (
        torch.randn(1, seq_len, heads, head_dim, generator=g,
                    device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    )
    sel = moba_selection(q, k, cfg)                 # [1, H, S, num_blocks]
    consumers = sel.sum(dim=2).flatten().tolist()   # one entry per (head, block)
    del q, k, sel

    # The tile height must come from the kernel, not a constant. An earlier
    # version hardcoded 32 here; once `choose_group` began deriving it from the
    # shared-memory budget, this simulation was silently predicting a kernel
    # that no longer existed.
    group = choose_group(cfg, head_dim, torch.finfo(torch.bfloat16).bits // 8)
    jobs = [math.ceil(c / group) for c in consumers if c > 0]
    total = sum(jobs)
    ideal = total / n_proc

    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(jobs), generator=rng).tolist()
    shuffled = [jobs[i] for i in order]

    return {
        "seq_len": seq_len,
        "block_size": block_size,
        "top_k": top_k,
        "n_proc": n_proc,
        "group": group,
        "programs": len(jobs),
        "mean_tiles": total / len(jobs),
        "max_tiles": max(jobs),
        "skew": max(jobs) / (total / len(jobs)),
        "jobs_per_proc": len(jobs) / n_proc,
        "ascending": makespan(sorted(jobs), n_proc) / ideal,
        "random": makespan(shuffled, n_proc) / ideal,
        "descending": makespan(sorted(jobs, reverse=True), n_proc) / ideal,
        "chunked": makespan([1] * total, n_proc) / ideal,
    }


def run(
    cases: list[tuple[int, int, int]] | None = None,
    n_proc: int | None = None,
    programs_per_sm: int = DEFAULT_PROGRAMS_PER_SM,
) -> list[dict]:
    """Sweeps configurations.

    Args:
        cases: List of (seq_len, block_size, top_k). Defaults to a sweep
            spanning the target regime and the low-program-count regime.
        n_proc: Concurrent processors. Probed from the device when omitted,
            as `multiprocessor_count() * programs_per_sm`.
        programs_per_sm: Resident programs per SM.

    Returns:
        One record per case.
    """
    if cases is None:
        # Matches the realistic gate range established against the vendored
        # references: block_size 32-128, top_k 4-12. See tests/conftest.py.
        cases = [
            (4096, 32, 4), (4096, 32, 8), (4096, 64, 8),
            (4096, 128, 8), (4096, 128, 12), (1024, 32, 4),
        ]
    if n_proc is None:
        n_proc = multiprocessor_count() * programs_per_sm
    return [analyse(s, bk, tk, n_proc) for s, bk, tk in cases]


def to_markdown(rows: list[dict]) -> str:
    """Renders the sweep as a markdown table.

    Args:
        rows: Records from `run`.

    Returns:
        A markdown table string.
    """
    head = ("| config | programs | jobs/proc | skew | ascending | **random** | "
            "**descending** | chunked |")
    sep = "|" + "---:|" * 8
    body = [
        f"| S={r['seq_len']} blk={r['block_size']} k={r['top_k']} | "
        f"{r['programs']} | {r['jobs_per_proc']:.1f} | {r['skew']:.1f}x | "
        f"{r['ascending']:.2f}x | **{r['random']:.2f}x** | "
        f"**{r['descending']:.2f}x** | {r['chunked']:.2f}x |"
        for r in rows
    ]
    return "\n".join([head, sep, *body])


if __name__ == "__main__":
    rows = run()
    print(f"Work unit = one tile of `choose_group` queries. "
          f"{rows[0]['n_proc']} concurrent slots.")
    print("Columns are makespan / ideal. 1.00x = perfect balance.\n")
    print(to_markdown(rows))
