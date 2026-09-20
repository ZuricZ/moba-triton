"""Is the sparse-backend comparison fair, or is the shape doing the work?

Beating FlashMoBA's CUDA kernel -- and Triton-MoBA and MoonshotAI's fused path
by more -- deserves scrutiny before it is written down. Two ways the result
could be an artefact rather than a finding:

1. A backend is not computing the same thing. If it silently produces garbage
   on this configuration, the timing is meaningless.
2. The shape is unfair. The sweep uses B=1, H=4, D=64 -- a small workload.
   FlashMoBA's own benchmarks use realistic model shapes, and its
   `decide_lg_block_m` picks query tiles of 512 to 1024. At H=4 on 108 SMs that
   may leave most of the GPU idle for reasons that have nothing to do with the
   algorithm.

TWO PASSES, ONE TABLE. `moonshot_efficient` needs flash-attn 2.6.x and
`triton_moba` needs >= 2.7, and no build satisfies both (`docs/environment.md`).
So this runs once per flash-attn build and each run *persists* its numbers;
`--merge` then reads every pass and prints the combined table, labelling which
build produced each column.

That persistence is the point rather than a convenience. The first version of
this script printed to stdout and nothing else, so composing two passes meant a
human reading two logs side by side and retyping -- and it called
`backend.forward` without consulting `is_available()` first, so the backend that
could not run in this build raised `RuntimeError` mid-measurement instead of
being reported as unavailable with a reason. A build where three of four
backends are expected to be absent has to treat absence as data.
"""

import argparse
import glob
import json
import os

import torch
import triton.testing

from bench.suite import cool_down_device, warm_up_device
from implementations import get_backend
from moba.config import MoBAConfig
from moba.naive import moba_attention_naive

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CFG = MoBAConfig(block_size=128, top_k=8, dtype=torch.bfloat16)

# The sparse competitors, measured against `ours` and the oracle. `sdpa_dense`
# and `fa2_minimal` are dense baselines rather than MoBA implementations, so a
# fair-vs-us question does not apply to them and they stay out of this file.
SPARSE_BACKENDS = ("flash_moba", "triton_moba", "moonshot_efficient")

# Shapes that ask whether the win survives a realistic model, not just the
# sweep's narrow one.
SHAPES = (
    (1, 8192, 4, 64, "sweep shape"),
    (1, 8192, 32, 128, "GPT-3 13B-ish"),
    (4, 8192, 32, 128, "same, batch 4"),
    (1, 16384, 32, 128, "long, wide"),
)

from bench import paths


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------


def flash_attn_tag() -> str:
    """Returns the installed flash-attn version, for naming this pass's file.

    Returns:
        For example "2.6.3", or "none" when flash-attn is not importable --
        which is itself a valid pass, measuring only the backends that do not
        need it.
    """
    try:
        import flash_attn
        return getattr(flash_attn, "__version__", "unknown")
    except Exception:                              # noqa: BLE001 - reports
        return "none"


def make(batch, seq_len, heads, head_dim):
    """Returns q, k, v for one shape, from a fixed seed."""
    g = torch.Generator(device="cuda").manual_seed(0)
    return [torch.randn(batch, seq_len, heads, head_dim, generator=g,
                        device="cuda", dtype=CFG.dtype) for _ in range(3)]


def time_backend(name, q, k, v) -> tuple[float | None, str]:
    """Times one backend, or reports why it could not run.

    Availability is checked *before* the call rather than discovered by
    catching what the call raises. A backend that needs the other flash-attn
    build is a fact about this environment, and the registry already knows it;
    letting it raise instead turns an expected absence into what looks like a
    broken kernel.

    Args:
        name: Backend name.
        q: Queries.
        k: Keys.
        v: Values.

    Returns:
        A tuple of (milliseconds, reason). Exactly one is meaningful: on
        success the reason is empty, and on refusal the time is None.
    """
    backend = get_backend(name)
    ok, why = backend.is_available(CFG)
    if not ok:
        return None, why

    fn = lambda: backend.forward(q, k, v, CFG)          # noqa: E731
    try:
        fn()
        cool_down_device()
        warm_up_device()
        return min(triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])[0]
                   for _ in range(3)), ""
    except Exception as exc:                       # noqa: BLE001 - third party
        # `is_available` said yes and the call still failed: a shape limit, an
        # allocation, an API drift the gate does not model. Worth a row saying
        # so rather than ending the run.
        return None, f"{type(exc).__name__}: {str(exc)[:60]}"


def measure_correctness() -> dict:
    """Returns each backend's error against the PyTorch oracle, by seq_len."""
    out = {}
    for seq_len in (1024, 4096):
        q, k, v = make(1, seq_len, 4, 64)
        ref = moba_attention_naive(q.float(), k.float(), v.float(), CFG)
        scale = ref.abs().max().item()
        row = {}
        for name in ("ours",) + SPARSE_BACKENDS:
            backend = get_backend(name)
            ok, why = backend.is_available(CFG)
            if not ok:
                row[name] = {"error": None, "reason": why}
                continue
            try:
                got = backend.forward(q, k, v, CFG).float()
                row[name] = {
                    "error": (got - ref).abs().max().item() / scale,
                    "reason": "",
                }
            except Exception as exc:               # noqa: BLE001
                row[name] = {"error": None,
                             "reason": f"{type(exc).__name__}: {exc!s:.60}"}
        out[str(seq_len)] = row
        del q, k, v, ref
        torch.cuda.empty_cache()
    return out


def measure_shapes() -> dict:
    """Returns each backend's latency at every shape, keyed by shape label."""
    out = {}
    for batch, seq_len, heads, head_dim, label in SHAPES:
        key = f"B{batch} S{seq_len} H{heads} D{head_dim}"
        try:
            q, k, v = make(batch, seq_len, heads, head_dim)
        except torch.OutOfMemoryError:
            out[key] = {"label": label,
                        "backends": {"*": {"ms": None,
                                           "reason": "OOM building inputs"}}}
            continue
        row = {}
        for name in ("ours",) + SPARSE_BACKENDS:
            ms, reason = time_backend(name, q, k, v)
            row[name] = {"ms": ms, "reason": reason}
        out[key] = {"label": label, "backends": row}
        del q, k, v
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def load_passes() -> dict:
    """Reads every persisted pass, newest wins on a tie.

    Returns:
        {flash_attn_version: pass dict}.
    """
    passes = {}
    for path in sorted(glob.glob(os.path.join(
            paths.card_dir(kind="data"), "fairness-fa*.json"))):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        passes[data["flash_attn"]] = data
    return passes


def merged_report(passes: dict) -> str:
    """Renders one table from however many passes have been run.

    A backend measured in more than one pass is reported from the pass where it
    actually ran. Where two passes both produced a number -- which happens for
    `ours` and `flash_moba`, neither of which uses flash-attn -- their spread is
    printed, because that spread is a free estimate of run-to-run
    reproducibility across a process restart.

    Args:
        passes: As `load_passes` returns.

    Returns:
        A printable report.
    """
    if not passes:
        return "no passes on disk; run this module under each flash-attn build"

    lines = ["=" * 78,
             f"MERGED over flash-attn {', '.join(sorted(passes))}",
             "=" * 78, ""]

    names = ("ours",) + SPARSE_BACKENDS
    lines.append("Correctness -- max error against the PyTorch oracle")
    seq_lens = sorted({s for p in passes.values() for s in p["correctness"]},
                      key=int)
    for seq_len in seq_lens:
        cells = []
        for name in names:
            best = _pick(passes, "correctness", seq_len, name, "error")
            cells.append(f"{name} {best:.4f}" if best is not None
                         else f"{name} -")
        lines.append(f"  S={seq_len:<6} " + "   ".join(cells))
    lines.append("")

    lines.append("Latency, and the ratio to ours")
    header = f"  {'shape':<24} {'ours':>9}"
    for name in SPARSE_BACKENDS:
        header += f" {name:>20}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    shapes = sorted({s for p in passes.values() for s in p["shapes"]})
    for shape in shapes:
        ours = _pick(passes, "shapes", shape, "ours", "ms")
        cells = f"  {shape:<24} " + (f"{ours:>9.3f}" if ours else f"{'-':>9}")
        for name in SPARSE_BACKENDS:
            theirs = _pick(passes, "shapes", shape, name, "ms")
            cells += (f" {theirs / ours:>19.2f}x" if theirs and ours
                      else f" {'-':>20}")
        lines.append(cells)

    lines.append("")
    lines.append("Absent backends, and why -- one per pass that refused them:")
    for version, data in sorted(passes.items()):
        for shape, row in data["shapes"].items():
            for name, cell in row["backends"].items():
                if cell["ms"] is None and cell.get("reason"):
                    lines.append(f"  fa{version}  {shape:<24} {name:<20} "
                                 f"{cell['reason'][:60]}")
            break        # one shape is enough; the reason does not vary

    spread = _reproducibility(passes)
    if spread:
        lines.append("")
        lines.append("Same backend, both passes -- run-to-run spread across a "
                     "flash-attn switch:")
        for name, value in sorted(spread.items()):
            lines.append(f"  {name:<20} {value:.1%}")
    return "\n".join(lines)


def _pick(passes, section, key, name, field):
    """Returns the first non-None value for one cell across all passes."""
    for data in passes.values():
        row = data[section].get(key)
        if not row:
            continue
        cells = row["backends"] if section == "shapes" else row
        cell = cells.get(name)
        if cell and cell.get(field) is not None:
            return cell[field]
    return None


def _reproducibility(passes) -> dict:
    """Returns the worst cross-pass spread for backends measured in both."""
    if len(passes) < 2:
        return {}
    worst = {}
    for name in ("ours",) + SPARSE_BACKENDS:
        for shape in {s for p in passes.values() for s in p["shapes"]}:
            values = []
            for data in passes.values():
                row = data["shapes"].get(shape)
                cell = row and row["backends"].get(name)
                if cell and cell.get("ms"):
                    values.append(cell["ms"])
            if len(values) > 1:
                gap = (max(values) - min(values)) / min(values)
                worst[name] = max(worst.get(name, 0.0), gap)
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge", action="store_true",
                        help="print the combined table from passes already on "
                             "disk, measuring nothing")
    args = parser.parse_args()

    if args.merge:
        print(merged_report(load_passes()))
        return

    if not torch.cuda.is_available():
        print("no CUDA device")
        return

    version = flash_attn_tag()
    print(f"pass: flash-attn {version} on "
          f"{torch.cuda.get_device_name(0)}\n")

    data = {"flash_attn": version,
            "device": torch.cuda.get_device_name(0),
            "correctness": measure_correctness(),
            "shapes": measure_shapes()}

    path = os.path.join(paths.card_dir(kind="data", create=True),
                        f"fairness-fa{version}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    print(f"wrote {path}\n")
    print(merged_report(load_passes()))


if __name__ == "__main__":
    main()
