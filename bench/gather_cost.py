"""Measures what gathering scattered rows actually costs versus reading in order.

This settles design question A: whether the MoBA kernel should gather query
rows directly from index arrays, or whether a permuted copy of Q should be
materialized in HBM first so the kernel can read contiguously.

The hypothesis under test. A cache line is 128 bytes. One query row is
`D * sizeof(dtype)` bytes -- exactly 128 at D=64 in bf16. Gathering whole,
cache-line-sized rows should therefore cost the same number of memory
transactions as reading them in order, making the usual "gathers are slow"
warning inapplicable. That warning is about gathering *elements*, where a
128-byte line is fetched to use 4 bytes of it.

Three modes read exactly the same rows, so data volume is identical and only
the access pattern varies:

    contiguous  rows 0..N-1, no index array at all
    identity    same order, addresses read from an index array
                -> isolates the cost of loading indices
    permuted    a random permutation of the same rows
                -> isolates the cost of scattering

Holding the rows fixed and varying only their order is what makes the ratio
meaningful. Vary both volume and pattern and the number means nothing.

`D` is swept because the hypothesis is about row size against line size.
"""

import torch
import triton
import triton.language as tl
import triton.testing


@triton.jit
def _read_rows(
    Q, Idx, Out, N_ROWS,
    D: tl.constexpr, BLOCK_M: tl.constexpr, USE_IDX: tl.constexpr,
):
    """Reads BLOCK_M rows and reduces them.

    The reduction exists so the loads have an observable effect and cannot be
    eliminated by the compiler. It is cheap relative to the memory traffic,
    which is what this benchmark is actually timing.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    valid = offs < N_ROWS

    if USE_IDX:
        rows = tl.load(Idx + offs, mask=valid, other=0)
    else:
        rows = offs

    q = tl.load(
        Q + rows[:, None] * D + offs_d[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    tl.store(Out + offs, tl.sum(q.to(tl.float32), axis=1), mask=valid)


def time_mode(q, idx, out, n_rows, d, block_m, use_idx) -> float:
    """Times one access pattern.

    Args:
        q: Source rows shaped [n_rows, d].
        idx: Row index array, int32.
        out: Output buffer shaped [n_rows].
        n_rows: Number of rows to read.
        d: Row width.
        block_m: Rows per program.
        use_idx: Whether to read addresses from `idx`.

    Returns:
        Median latency in milliseconds.
    """
    grid = (triton.cdiv(n_rows, block_m),)

    def run():
        _read_rows[grid](q, idx, out, n_rows, D=d, BLOCK_M=block_m,
                         USE_IDX=use_idx)

    return triton.testing.do_bench(run)


def run(
    n_rows: int = 8 * 4096,
    block_m: int = 32,
    head_dims: tuple[int, ...] = (32, 64, 128),
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> list[dict]:
    """Sweeps head dimension and returns one record per configuration.

    Args:
        n_rows: Total rows read; defaults to B*H*S at S=4096, H=8.
        block_m: Rows per program; defaults to the expected MoBA group size
            `top_k * block_size` at the project's hardest configuration.
        head_dims: Head dimensions to sweep.
        dtype: Element dtype.
        seed: RNG seed.

    Returns:
        A list of flat dicts suitable for tabulation.
    """
    rows = []
    for d in head_dims:
        g = torch.Generator(device="cuda").manual_seed(seed)
        q = torch.randn(n_rows, d, generator=g, device="cuda", dtype=dtype)
        out = torch.empty(n_rows, device="cuda", dtype=torch.float32)

        identity = torch.arange(n_rows, device="cuda", dtype=torch.int32)
        permuted = torch.randperm(n_rows, device="cuda").to(torch.int32)

        t_contig = time_mode(q, identity, out, n_rows, d, block_m, False)
        t_ident = time_mode(q, identity, out, n_rows, d, block_m, True)
        t_perm = time_mode(q, permuted, out, n_rows, d, block_m, True)

        rows.append({
            "head_dim": d,
            "row_bytes": d * q.element_size(),
            "working_set_mb": n_rows * d * q.element_size() / 1024**2,
            "contiguous_us": t_contig * 1000,
            "identity_us": t_ident * 1000,
            "permuted_us": t_perm * 1000,
            "scatter_cost": t_perm / t_contig,
            "index_cost": t_ident / t_contig,
        })
        del q, out, identity, permuted
    return rows


def to_markdown(rows: list[dict]) -> str:
    """Renders the sweep as a markdown table.

    Args:
        rows: Records from `run`.

    Returns:
        A markdown table string.
    """
    head = ("| head_dim | row bytes | working set | contiguous | identity | "
            "permuted | index cost | **scatter cost** |")
    sep = "|" + "---:|" * 8
    body = [
        f"| {r['head_dim']} | {r['row_bytes']} | {r['working_set_mb']:.0f} MB | "
        f"{r['contiguous_us']:.1f} us | {r['identity_us']:.1f} us | "
        f"{r['permuted_us']:.1f} us | {r['index_cost']:.2f}x | "
        f"**{r['scatter_cost']:.2f}x** |"
        for r in rows
    ]
    return "\n".join([head, sep, *body])


if __name__ == "__main__":
    print(to_markdown(run()))
