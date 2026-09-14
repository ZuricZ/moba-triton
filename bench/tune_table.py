"""Measures the launch configurations that `tuned_configs.TUNED` ships.

This is where the tuning that used to happen on a user's first inference call
now happens instead: once, deliberately, on a machine someone is watching, with
the thermal protocol `bench/suite.py` provides. The output is table source to
paste into `src/moba/kernels/tuned_configs.py`.

HOW IT WORKS. Runs with `MOBA_AUTOTUNE=1` set before the kernels are imported,
which swaps `triton.heuristics` back for `triton.autotune` (see
`moba.kernels.tuning.launch_policy`). Each shape is launched once so the
autotuner races its configs, and the winner is read out of
`kernel.cache`. Nothing here re-implements the search -- Triton already has one,
and a second would drift from it.

WHY THE ENV VAR AND NOT AN ARGUMENT. The policy is chosen by a decorator, and
decorators run at import. A flag parsed in `main()` would arrive after the
kernels were already wrapped. Setting it in `os.environ` at the top of this
module, before the imports that matter, is the only ordering that works -- and
is why the imports below are deliberately not at the top of the file.

USAGE
    uv run python -m bench.tune_table                # measure, print table
    uv run python -m bench.tune_table --check        # what this card is missing
    uv run python -m bench.tune_table --seq-lens 1024,4096

WHAT THIS DOES NOT DO. It does not write to `tuned_configs.py`. A benchmark
that edits the source it is measuring can overwrite a good table with one taken
on a hot card, and this project has already lost 14% once to exactly that
(`moba_fwd`'s note on `cache_results`). Read the output, sanity-check it against
what is there, paste it.
"""

import os

# Before importing anything under `moba.kernels`, for the reason above.
os.environ["MOBA_AUTOTUNE"] = "1"

import argparse                                         # noqa: E402
import textwrap                                         # noqa: E402

import torch                                            # noqa: E402

from bench.suite import cool_down_device, warm_up_device  # noqa: E402
from moba.config import MoBAConfig                      # noqa: E402
from moba.kernels import moba_bwd, moba_fwd             # noqa: E402
from moba.kernels import tuned_configs                  # noqa: E402
from moba.kernels.moba_fwd import attend_partials, combine_partials  # noqa: E402
from moba.kernels.moba_bwd import moba_backward         # noqa: E402
from moba.kernels.routing import build_csc_fused        # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BATCH, HEADS, HEAD_DIM = 1, 4, 64
BLOCK_SIZE, TOP_K = 128, 8


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------


def winner(kernel) -> dict | None:
    """Reads the configuration the autotuner settled on.

    Args:
        kernel: An autotuned Triton kernel that has been launched at least once.

    Returns:
        A dict of GROUP, num_warps and num_stages, or None if nothing was
        tuned -- which means the kernel was never reached, not that it has no
        preference.
    """
    cache = getattr(kernel, "cache", None)
    if not cache:
        return None
    config = list(cache.values())[-1]
    return {
        "GROUP": config.kwargs["GROUP"],
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
    }


def tune_shape(seq_len: int, cfg: MoBAConfig) -> dict:
    """Tunes both passes at one sequence length.

    The card is settled before each pass, because the autotuner's own timings
    decide the winner and a hot card can change it -- measured at 14% on this
    project's laptop, enough to pick a different height.

    Args:
        seq_len: Sequence length to tune at.
        cfg: The MoBA configuration.

    Returns:
        A dict with "fwd" and "bwd" keys, each a config dict or None.
    """
    generator = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(BATCH, seq_len, HEADS, HEAD_DIM, generator=generator,
                           device="cuda", dtype=cfg.dtype) for _ in range(3))
    dout = torch.randn_like(q)

    index = build_csc_fused(q, k, cfg)

    cool_down_device()
    warm_up_device()
    part_acc, part_m, part_l = attend_partials(q, k, v, cfg, index)
    out, lse = combine_partials(part_acc, part_m, part_l, q.dtype,
                                return_lse=True)

    cool_down_device()
    warm_up_device()
    moba_backward(dout, q, k, v, out, lse, part_m, cfg, index)

    result = {"fwd": winner(moba_fwd.moba_fwd_kernel),
              "bwd": winner(moba_bwd.moba_bwd_kernel)}
    del q, k, v, dout, part_acc, part_m, part_l, out, lse, index
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Emitting
# ---------------------------------------------------------------------------


def emit(rows: dict, arch: str, cfg: MoBAConfig) -> str:
    """Renders measured configs as source for `tuned_configs.TUNED`.

    Args:
        rows: {(pass_name, bracket): config dict}.
        arch: Architecture tag such as "sm_80".
        cfg: The configuration these were measured at.

    Returns:
        Python source, indented to sit inside the TUNED literal.
    """
    lines = [f"    # {arch}, block_size {cfg.block_size}, D {HEAD_DIM}, "
             f"top_k {cfg.top_k}, H {HEADS}/{HEADS}.",
             f"    # Measured by bench/tune_table.py on "
             f"{torch.cuda.get_device_name(0)}."]
    for (pass_name, bracket), chosen in sorted(rows.items()):
        key_tile = moba_bwd.BWD_KEY_TILE if pass_name == "bwd" else 0
        key = (arch, pass_name, cfg.block_size, key_tile, HEAD_DIM, cfg.top_k,
               HEADS, HEADS, bracket)
        lines.append(f"    {key!r}:")
        lines.append(f"        {chosen!r},")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", default=None,
                        help="comma separated; defaults to the table brackets")
    parser.add_argument("--check", action="store_true",
                        help="report what this card is missing, measure nothing")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device; this measures, it cannot be modelled")
        return

    arch = tuned_configs.capability_tag()
    if args.check:
        print(f"{arch} ({torch.cuda.get_device_name(0)})")
        for pass_name in ("fwd", "bwd"):
            gaps = tuned_configs.missing(pass_name, arch)
            if gaps:
                brackets = ", ".join(str(g[-1]) for g in gaps)
                print(f"  {pass_name}: no entry at {brackets} -> DEFAULT "
                      f"{tuned_configs.DEFAULT}")
            else:
                print(f"  {pass_name}: complete")
        return

    cfg = MoBAConfig(block_size=BLOCK_SIZE, top_k=TOP_K, dtype=torch.bfloat16)
    seq_lens = ([int(x) for x in args.seq_lens.split(",")]
                if args.seq_lens else list(tuned_configs.SEQ_BRACKETS))

    print(f"tuning on {arch} ({torch.cuda.get_device_name(0)}), "
          f"block_size={cfg.block_size} top_k={cfg.top_k} D={HEAD_DIM}")
    rows = {}
    for seq_len in seq_lens:
        try:
            found = tune_shape(seq_len, cfg)
        except torch.OutOfMemoryError:
            print(f"  S={seq_len:<7} out of memory; skipped")
            continue
        for pass_name, chosen in found.items():
            if chosen is None:
                continue
            rows[(pass_name, tuned_configs.seq_bracket(seq_len))] = chosen
            print(f"  S={seq_len:<7} {pass_name}  {chosen}")

    if not rows:
        print("nothing tuned")
        return

    print("\npaste into src/moba/kernels/tuned_configs.py, inside TUNED:\n")
    print(emit(rows, arch, cfg))
    print(textwrap.dedent("""
        Check before pasting: a config that differs from DEFAULT on only one
        bracket is more likely a noisy measurement than a real boundary. Re-run
        that bracket before believing it."""))


if __name__ == "__main__":
    main()
