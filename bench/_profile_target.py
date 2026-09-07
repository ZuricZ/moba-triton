"""The workload Nsight Compute runs. Not useful on its own.

Private -- the leading underscore is the module-private marker this project
reserves for exactly that, and nothing but `bench/profile.py` should import it.

It exists as a separate module because `ncu` profiles a *process*, so the thing
being measured has to be a script it can launch. Keeping it minimal is the
point: no benchmark loop, no warm-up, no timing. One compile pass to get the
kernels built, then one measured pass. Anything else would be replayed by the
profiler along with the work, inflating every counter with setup.
"""

import argparse

import torch

from implementations import get_backend
from moba.config import MoBAConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--backend", default="ours")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    cfg = MoBAConfig(block_size=args.block_size, top_k=args.top_k,
                     dtype=torch.bfloat16)
    backend = get_backend(args.backend)
    ok, reason = backend.is_available(cfg)
    if not ok:
        raise SystemExit(f"{args.backend} unavailable: {reason}")

    generator = torch.Generator(device="cuda").manual_seed(0)
    shape = (args.batch, args.seq_len, args.heads, args.head_dim)
    q, k, v = (torch.randn(shape, generator=generator, device="cuda",
                           dtype=cfg.dtype) for _ in range(3))

    # Triton compiles on first call and autotuning would otherwise be counted
    # as part of the measurement. This pass is thrown away; only the second is
    # what the profiler is meant to see. `ncu` still records both, which is why
    # profile.py sums repeated launches of the same kernel rather than taking
    # the first.
    backend.forward(q, k, v, cfg)
    torch.cuda.synchronize()

    backend.forward(q, k, v, cfg)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
