"""The flat roof, measured at the tile shape this kernel actually uses.

WHY THE ROOFLINE'S CURRENT ROOF IS WRONG. `bench/roofline.py` draws its
compute ceiling from a 4096x4096x4096 GEMM -- 204 TFLOP/s on the A100. That is
the friendliest shape a tensor core will ever see: square, enormous, perfectly
tiled, every operand reused thousands of times. Our attention kernel does
nothing that resembles it. Its inner loop is two small `tl.dot` calls:

    qk  = q @ k^T      [GROUP, D] x [D, BLOCK_SIZE]      64 x 64 x 128
    acc = p  @ v       [GROUP, BLOCK_SIZE] x [BLOCK_SIZE, D]   64 x 128 x 64

A 64-row tile cannot amortise what a 4096-row tile amortises, so a roofline
drawn against the GEMM number makes every attention kernel in the figure look
bad by the same wrong factor, and makes "21% of roof" read as headroom that is
not actually there.

WHAT THIS MEASURES INSTEAD. A Triton kernel doing only those two dots, in a
loop, with nothing else: no gather, no masking, no partial stores. Everything
the real kernel does *besides* arithmetic is removed, so what is left is the
arithmetic ceiling at this tile shape and this occupancy.

TWO ROOFS, NOT ONE, and the gap between them is a result. A roofline's flat roof
classically means "what if memory were free", which argues for timing the dots
alone. But the online softmax is unavoidable algorithm work, and it runs on the
CUDA cores rather than the tensor cores -- FlashAttention-2's paper makes
exactly this point: non-matmul FLOPs are few but cost far more per FLOP. So both
are measured:

    dots        the two matmuls alone
    dots+softmax  the same, plus max / exp2 / sum / rescale

The pair brackets the true ceiling, and their difference prices the softmax at
our tile shape -- currently an unattributed part of the roofline gap.

MEASURED, AND THE PREMISE ABOVE IS WRONG ON A SMALL CARD (2026-09-21, sm_86).
The paragraph that opens this docstring predicts the tile roof lands *below* the
GEMM. On the RTX 3050 Ti it lands above it:

    bf16, 4096^3 GEMM     13.04 TFLOP/s
    bf16, our tile        14.23 TFLOP/s   <- 1.09x FASTER

The reason is the card, not the tile. A 4096^3 bf16 GEMM touches about 100 MB
of operands against roughly 2 MB of L2 here, so on 4 GB the "friendliest shape
a tensor core will ever see" is itself memory bound. This kernel cycles K/V
through an L2-resident circular buffer on purpose, so it is the only one of the
two measuring arithmetic at all. Expect the predicted ordering to return on a
card whose L2 and bandwidth can feed the big GEMM (A100, H100); it has not been
measured on either. Not for want of opportunity: this module landed 2026-09-14
(`4916c29`), before the A100 was rented on 09-18 and before the H100 on 09-20,
and was simply never run on either. Both cards are now deleted, so the ordering
above stands unverified on anything larger than a laptop.

WHAT THIS DID NOT CHANGE, WHICH IS THE POINT WORTH KEEPING. Correcting the flat
roof moved no conclusion in any document. Every kernel the write-ups argue about
-- `ours`, `flash_moba`, both naive paths -- is *memory* bound, and a memory
bound point is measured against the slanted roof, `intensity * bandwidth`, which
the flat roof does not enter. At S=8192 `ours` reads 75.8 FLOP/byte and a roof
of 12.47 TFLOP/s, and 75.8 * 153.3 GiB/s = 12.48: the flat roof is absent from
that number. The flat roof governs only the ridge point and the compute-bound
dense baselines (`sdpa_dense`, `fa2_minimal`). `ours` at 75.8 sits below the
ridge either way -- 86.5 with the measured roof, 79.2 with the GEMM -- so even
the classification is unchanged.

WHAT KEEPS THE COMPILER HONEST. Two hazards, both of which would produce a
flattering number:

* **Hoisting.** If the operands never change, Triton can lift the dot out of the
  loop and compute it once. The K/V pointers therefore advance through a small
  circular buffer, so the loads vary per iteration while still hitting L2 --
  keeping the measurement about arithmetic rather than HBM.
* **Dead code.** A result nobody reads can be deleted entirely. The accumulator
  is stored at the end, so every iteration is load-bearing.

USAGE
    uv run python -m bench.tile_ceiling
    uv run python -m bench.tile_ceiling --group 64 --block-size 128 --head-dim 64
"""

import argparse

import torch
import triton
import triton.language as tl
import triton.testing

from bench.suite import cool_down_device, warm_up_device

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The tile the forward actually launches: GROUP queries against one BLOCK_SIZE
# key block at HEAD_DIM. Defaults match `tuned_configs.DEFAULT` and the shape
# every published number in this repository uses.
GROUP, BLOCK_SIZE, HEAD_DIM = 64, 128, 64

# Iterations per program. Large enough that launch overhead is noise, small
# enough that the whole sweep is seconds.
ITERS = 512

# How many K/V tiles the circular buffer holds. Small enough to sit in L2 -- the
# point is to defeat hoisting, not to measure memory.
TILES = 8

# Programs launched. One per SM times a few waves, so the card is saturated the
# way it is during a real attend pass.
WAVES = 4


# ---------------------------------------------------------------------------
# Device code
# ---------------------------------------------------------------------------


@triton.jit
def tile_ceiling_kernel(K, V, Out,
                        GROUP: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                        D: tl.constexpr, ITERS: tl.constexpr,
                        TILES: tl.constexpr, SOFTMAX: tl.constexpr):
    """Runs the forward's two dots `ITERS` times and stores the accumulator.

    Args:
        K: Key tiles, [TILES, BLOCK_SIZE, D].
        V: Value tiles, same shape.
        Out: Per-program accumulator, [programs, GROUP, D].
        GROUP: Rows of the query tile.
        BLOCK_SIZE: Keys per block.
        D: Head dimension.
        ITERS: Loop trips.
        TILES: Tiles in the circular buffer.
        SOFTMAX: Whether to include the online-softmax arithmetic.
    """
    pid = tl.program_id(0)

    offs_g = tl.arange(0, GROUP)
    offs_n = tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, D)

    # A query tile that stays in registers, as it does in the real kernel: one
    # program owns one chunk of gathered queries for the whole inner loop.
    q = tl.full((GROUP, D), 1.0, dtype=tl.bfloat16)

    acc = tl.zeros((GROUP, D), dtype=tl.float32)
    m_running = tl.full((GROUP,), float("-inf"), dtype=tl.float32)
    l_running = tl.zeros((GROUP,), dtype=tl.float32)

    for i in range(ITERS):
        # Walk the buffer so the operands differ between iterations and the
        # compiler cannot hoist the dot out of the loop.
        tile = i % TILES
        base = tile * BLOCK_SIZE * D
        k = tl.load(K + base + offs_n[:, None] * D + offs_d[None, :])
        v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :])

        qk = tl.dot(q, tl.trans(k))

        if SOFTMAX:
            # The same online softmax the forward runs, in base 2 for the same
            # reason: the SFU has a native 2^x and no e^x.
            m_new = tl.maximum(m_running, tl.max(qk, axis=1))
            alpha = tl.exp2(m_running - m_new)
            p = tl.exp2(qk - m_new[:, None])
            l_running = l_running * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_running = m_new
        else:
            acc += tl.dot(qk.to(v.dtype), v)

    out = Out + pid * GROUP * D + offs_g[:, None] * D + offs_d[None, :]
    tl.store(out, acc)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


def measure(group: int, block_size: int, head_dim: int, softmax: bool,
            programs: int) -> tuple[float, float]:
    """Times the microkernel and returns its achieved throughput.

    Args:
        group: Rows of the query tile.
        block_size: Keys per block.
        head_dim: Head dimension.
        softmax: Whether to include the online-softmax arithmetic.
        programs: Grid size.

    Returns:
        A tuple of (milliseconds, TFLOP/s). Only the two matmuls are counted as
        FLOPs in both variants, so the softmax variant's lower number is the
        price of the softmax expressed in matmul terms -- which is the
        comparison the roofline needs, since the roof is a matmul rate.
    """
    generator = torch.Generator(device="cuda").manual_seed(0)
    k = torch.randn(TILES, block_size, head_dim, generator=generator,
                    device="cuda", dtype=torch.bfloat16)
    v = torch.randn(TILES, block_size, head_dim, generator=generator,
                    device="cuda", dtype=torch.bfloat16)
    out = torch.empty(programs, group, head_dim, device="cuda",
                      dtype=torch.float32)

    def run():
        tile_ceiling_kernel[(programs,)](
            k, v, out,
            GROUP=group, BLOCK_SIZE=block_size, D=head_dim, ITERS=ITERS,
            TILES=TILES, SOFTMAX=softmax,
        )

    run()
    cool_down_device()
    warm_up_device()
    ms = min(triton.testing.do_bench(run, quantiles=[0.5, 0.2, 0.8])[0]
             for _ in range(3))

    # 2 FLOPs per MAC, two matmuls: [G,D]x[D,N] and [G,N]x[N,D].
    flops = programs * ITERS * 2 * 2 * group * block_size * head_dim
    return ms, flops / (ms * 1e-3) / 1e12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=int, default=GROUP)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device; a ceiling is measured, never modelled")
        return

    props = torch.cuda.get_device_properties(0)
    programs = props.multi_processor_count * WAVES
    print(f"{props.name}: {props.multi_processor_count} SMs, "
          f"{programs} programs ({WAVES} waves)")
    print(f"tile {args.group}x{args.block_size}x{args.head_dim}, "
          f"{ITERS} iterations per program\n")

    print(f"  {'variant':<16} {'ms':>8} {'TFLOP/s':>10}")
    print("  " + "-" * 36)
    results = {}
    for label, softmax in (("dots", False), ("dots+softmax", True)):
        ms, tflops = measure(args.group, args.block_size, args.head_dim,
                             softmax, programs)
        results[label] = tflops
        print(f"  {label:<16} {ms:>8.3f} {tflops:>10.2f}")

    gap = results["dots"] / results["dots+softmax"]
    print(f"\n  the softmax costs {gap:.2f}x at this tile shape")
    print("  Use `dots+softmax` as the roofline's flat roof: it is the rate an")
    print("  attention kernel can actually reach, where the 4096^3 GEMM figure")
    print("  is a rate nothing in this pipeline could approach.")


if __name__ == "__main__":
    main()
