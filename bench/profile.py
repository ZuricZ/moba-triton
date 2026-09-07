"""Measured DRAM traffic and occupancy, from Nsight Compute's hardware counters.

WHY THIS EXISTS. `bench/roofline.py` places each backend using a *model* of the
bytes its algorithm must move. That model is reasoning, and reasoning about
memory traffic is exactly the thing that turns out to be wrong -- it already
did once, when it assumed the partial accumulator was float32 and doubled the
largest term. This module asks the hardware instead.

It also answers a question the roofline raised and cannot settle. `ours` sits
at 75.8 FLOP/byte against a ridge of 86, so it is barely memory bound -- and
yet it reaches only ~44% of the throughput that intensity allows. Neither
ceiling explains the gap. The candidates are occupancy, launch overhead across
five pipeline stages, and stalls that a roofline cannot see. All three are
counters.

WHAT `ncu` COSTS. It serializes kernels and replays each one several times to
collect counters, so a profiled run is many times slower than a timed one and
its wall-clock numbers are meaningless. Never read a duration from here; read
bytes and occupancy, and take timings from `bench/compare.py`.

PERMISSIONS. Counter collection needs `CAP_PERFMON` or root. On a consumer
laptop it usually returns ERR_NVGPUCTRPERM; on a rented datacenter box in a
container it usually just works. When it fails, this module says so plainly
rather than reporting zeros -- a silent zero is how a missing measurement
becomes a believed one.
"""

import argparse
import csv
import io
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

# The counters worth having, and why each. Kept short deliberately: every
# extra metric is another replay pass, and a profile that takes ten minutes
# gets run once and then not again.
METRICS = [
    # Bytes actually moved between DRAM and the chip -- the number the roofline
    # models. Read and write are separate because our combine pass is
    # write-heavy in one kernel and read-heavy in the next, and a single total
    # would hide that.
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    # Occupancy: warps resident against warps the SM could hold. Low occupancy
    # with low DRAM traffic is the signature of a kernel that is neither
    # compute nor bandwidth bound -- it is simply not filling the machine.
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    # Tensor-core utilization, to tell "not enough work issued" apart from
    # "work issued but stalled".
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    # How long each kernel actually ran, for attributing traffic per stage.
    "gpu__time_duration.sum",
]

SHORT = {
    "dram__bytes_read.sum": "DRAM read",
    "dram__bytes_write.sum": "DRAM write",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy %",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": "tensor %",
    "gpu__time_duration.sum": "duration",
}

BYTES_PER_MIB = 1 << 20


@dataclass
class KernelProfile:
    """One kernel launch, as Nsight Compute saw it."""

    name: str
    metrics: dict[str, float]

    @property
    def short_name(self) -> str:
        """The kernel name without C++ template noise."""
        base = self.name.split("(")[0]
        return base.rsplit("::", 1)[-1][:44]

    def bytes_moved(self) -> float:
        return (self.metrics.get("dram__bytes_read.sum", 0.0)
                + self.metrics.get("dram__bytes_write.sum", 0.0))


# ---------------------------------------------------------------------------
# Running the profiler
# ---------------------------------------------------------------------------


def ncu_path() -> str | None:
    """Finds `ncu`, including the usual place it hides off PATH."""
    found = shutil.which("ncu")
    if found:
        return found
    for root in ("/usr/local/cuda", "/usr/local/cuda-12.6", "/opt/nvidia/nsight-compute"):
        candidate = os.path.join(root, "bin", "ncu")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def profile(script: str, args: list[str] | None = None,
            kernel_filter: str = "") -> tuple[list[KernelProfile], str]:
    """Runs `script` under Nsight Compute and returns one entry per kernel.

    Args:
        script: Python module path to run with `-m`, or a file path.
        args: Arguments passed through to the script.
        kernel_filter: Regex; only matching kernels are profiled. Narrowing
            this is the difference between a one-minute profile and a ten-
            minute one, because each kernel is replayed several times.

    Returns:
        A tuple of (profiles, diagnostic). The list is empty when profiling
        failed, and the diagnostic then says why in the profiler's own words.
    """
    ncu = ncu_path()
    if ncu is None:
        return [], ("ncu not found. It ships with the CUDA toolkit but is "
                    "usually off PATH; try /usr/local/cuda-*/bin/ncu.")

    command = [
        ncu, "--csv", "--target-processes", "all",
        "--metrics", ",".join(METRICS),
    ]
    if kernel_filter:
        command += ["--kernel-name", f"regex:{kernel_filter}"]
    command += [sys.executable, "-m", script] + (args or [])

    result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0 or "==ERROR==" in result.stderr:
        detail = (result.stderr or result.stdout).strip().splitlines()
        hint = ""
        if "ERR_NVGPUCTRPERM" in result.stderr:
            hint = (" -- counter collection is permission-gated. On Linux: "
                    "sudo, or set NVreg_RestrictProfilingToAdminUsers=0. "
                    "It is normally allowed inside a rented GPU container.")
        return [], "\n".join(detail[-6:]) + hint
    return _parse(result.stdout), ""


def _parse(text: str) -> list[KernelProfile]:
    """Turns ncu's CSV into one record per kernel.

    ncu emits one *row per metric per kernel*, plus banner lines before the
    header, so this pivots rows into kernels rather than reading a table.
    """
    lines = [ln for ln in text.splitlines() if ln.startswith('"')]
    if not lines:
        return []
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    order: list[str] = []
    collected: dict[str, dict[str, float]] = {}
    for row in rows:
        name = row.get("Kernel Name") or ""
        metric = row.get("Metric Name") or ""
        raw = (row.get("Metric Value") or "").replace(",", "")
        if not name or not metric:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if name not in collected:
            collected[name] = {}
            order.append(name)
        # ncu sums repeated launches of the same kernel across the run.
        collected[name][metric] = collected[name].get(metric, 0.0) + value
    return [KernelProfile(name, collected[name]) for name in order]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_profiles(profiles: list[KernelProfile], modelled_mib: float | None) -> None:
    """Prints per-kernel counters and, when given, the model's error."""
    if not profiles:
        print("no kernels profiled")
        return

    total = sum(p.bytes_moved() for p in profiles)
    print(f"\n  {'kernel':<46} {'read':>9} {'write':>9} "
          f"{'occ %':>7} {'tensor %':>9}")
    print("  " + "-" * 84)
    for p in sorted(profiles, key=lambda p: -p.bytes_moved()):
        m = p.metrics
        print(f"  {p.short_name:<46} "
              f"{m.get('dram__bytes_read.sum', 0) / BYTES_PER_MIB:>8.1f}M "
              f"{m.get('dram__bytes_write.sum', 0) / BYTES_PER_MIB:>8.1f}M "
              f"{m.get('sm__warps_active.avg.pct_of_peak_sustained_active', 0):>7.1f} "
              f"{m.get('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active', 0):>9.1f}")
    print("  " + "-" * 84)
    print(f"  {'total DRAM traffic':<46} {total / BYTES_PER_MIB:>18.1f} MiB")

    if modelled_mib is not None:
        measured = total / BYTES_PER_MIB
        ratio = measured / modelled_mib if modelled_mib else float("nan")
        print(f"  {'modelled by bench/roofline.py':<46} "
              f"{modelled_mib:>18.1f} MiB")
        print(f"  {'measured / modelled':<46} {ratio:>18.2f}x")
        # Both directions are informative and mean different things, which is
        # why this does not just report an absolute error.
        if ratio < 0.85:
            print("\n  Measured traffic is below the model: caches are absorbing "
                  "re-reads\n  the model counts in full -- most likely the "
                  "gathered-Q term.")
        elif ratio > 1.15:
            print("\n  Measured traffic exceeds the model: something moves bytes "
                  "the model\n  does not know about -- spills, write allocation, "
                  "or a stage not counted.")
        else:
            print("\n  Model and hardware agree within 15%.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile a MoBA forward pass with Nsight Compute.")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--backend", default="ours")
    parser.add_argument("--kernel", default="",
                        help="regex of kernel names to profile; empty means all")
    args = parser.parse_args()

    if ncu_path() is None:
        print(__doc__.strip().splitlines()[0])
        print("\nncu not found -- nothing to do. It ships with the CUDA "
              "toolkit and is usually off PATH.")
        return

    print(f"profiling {args.backend} at S={args.seq_len} "
          f"(ncu replays kernels; wall-clock here is meaningless)")
    profiles, diagnostic = profile(
        "bench._profile_target",
        ["--seq-len", str(args.seq_len), "--backend", args.backend],
        kernel_filter=args.kernel,
    )
    if diagnostic:
        print(f"\nprofiling failed:\n  {diagnostic}")
        return

    modelled = None
    try:
        from bench.roofline import BATCH, CFG, HEAD_DIM, HEADS, model_traffic
        traffic = model_traffic(args.backend, BATCH, args.seq_len, HEADS,
                                HEAD_DIM, CFG)
        if traffic is not None:
            modelled = traffic.total / BYTES_PER_MIB
    except Exception:                       # noqa: BLE001 - reporting only
        pass
    print_profiles(profiles, modelled)


if __name__ == "__main__":
    main()
