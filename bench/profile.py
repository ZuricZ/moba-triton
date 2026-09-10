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

PERMISSIONS, AND WHY THERE ARE TWO TIERS. Counter collection needs
`CAP_PERFMON` or admin, and it is refused far more often than expected --
verified refused on this laptop *and* on a rented A100 container, which lacked
both `CAP_SYS_ADMIN` and `CAP_PERFMON` while its host driver restricted
profiling to admins. So `ncu` cannot be the only answer.

`torch.profiler` reaches a different API. CUPTI splits into *metrics* (the
hardware counters `ncu` reads, permission-gated) and *activity* (kernel
records, normally not gated), and the activity records already carry the
figures that matter for the open question:

    est. achieved occupancy %   registers per thread   grid / block
    occupancy (theoretical)     shared memory          warps per SM

That is not DRAM traffic, so it cannot check the roofline's traffic model. But
occupancy is the leading suspect for the missing throughput, and this tier
reaches it where `ncu` is refused. `--tier torch` forces it; `auto` tries
counters first and falls back with a note saying which ran.

Never read a duration from either tier. `ncu` replays kernels to collect
counters; `torch.profiler` adds tracing overhead. Timings come from
`bench/compare.py`.
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
# Tier 2: torch.profiler, for boxes where counters are refused
# ---------------------------------------------------------------------------


@dataclass
class KernelOccupancy:
    """One kernel, aggregated over its launches, as CUPTI's activity API saw it.

    Attributes:
        name: Mangled kernel name.
        launches: How many times it ran.
        microseconds: Total time across those launches. Indicative only --
            tracing adds overhead, so ranking is meaningful and absolutes are
            not.
        achieved: CUPTI's estimated achieved occupancy, percent. Reported as
            zero when its calculator gives up; see `limiter`.
        registers: Registers per thread.
        shared: Static shared memory per block, bytes.
        block_threads: Threads per block.
        limiter: What CUPTI says caps residency -- "REGS", "SMEM", "WARPS",
            "BLOCKS". The single most useful field here, because the fixes
            differ: registers want a smaller tile, shared memory wants a
            narrower one, warps want more work per thread.
        limits: Blocks per SM each resource allows, keyed by resource.
        grid: Blocks launched.
    """

    name: str
    launches: int = 0
    microseconds: float = 0.0
    achieved: float = 0.0
    registers: int = 0
    shared: int = 0
    block_threads: int = 0
    limiter: str = "?"
    limits: dict = None
    grid: tuple = ()

    @property
    def short_name(self) -> str:
        base = self.name.split("(")[0]
        return base.rsplit("::", 1)[-1][:38]

    @property
    def blocks_per_sm(self) -> int:
        """Blocks that can be resident, the minimum over every resource.

        `blockLimitSharedMem` is zero whenever a kernel opts in to more than
        the 48 KiB static limit -- CUPTI's calculator does not model the opt-in
        that Triton performs, so it concludes nothing fits, which is plainly
        false for a kernel that ran. Zeros are therefore ignored rather than
        treated as a hard limit of none.
        """
        if not self.limits:
            return 0
        positive = [v for v in self.limits.values() if v > 0]
        return min(positive) if positive else 0

    @property
    def warps_resident(self) -> float:
        """Warps per SM this launch configuration can keep resident."""
        return self.blocks_per_sm * (self.block_threads / 32.0)


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def torch_profile(seq_len: int, backend_name: str, cfg,
                  backward: bool = False) -> tuple[list[KernelOccupancy], str]:
    """Profiles one forward (or training step) with `torch.profiler`.

    Runs in-process rather than under a subprocess: there is no counter
    collection to isolate, and CUPTI's activity records come back through the
    trace regardless.

    Args:
        seq_len: Sequence length to run.
        backend_name: Registry name of the backend to profile.
        cfg: `MoBAConfig` to run it with.
        backward: Profile `forward().backward(dO)` instead of the forward.

    Returns:
        (records, diagnostic). The list is empty when profiling produced no
        kernel events, and the diagnostic then says so.
    """
    import json
    import tempfile

    import torch
    from torch.profiler import ProfilerActivity, profile as torch_profiler

    from implementations import get_backend

    backend = get_backend(backend_name)
    ok, reason = backend.is_available(cfg)
    if not ok:
        return [], f"{backend_name} unavailable: {reason}"

    generator = torch.Generator(device="cuda").manual_seed(0)
    shape = (1, seq_len, 4, 64)
    q, k, v = (torch.randn(shape, generator=generator, device="cuda",
                           dtype=cfg.dtype, requires_grad=backward)
               for _ in range(3))
    call = (getattr(backend, "forward_grad", backend.forward) if backward
            else backend.forward)

    # Compile and autotune first. Both would otherwise land inside the trace
    # and swamp it -- autotuning alone runs the kernel a dozen times.
    out = call(q, k, v, cfg)
    grad = torch.randn_like(out) if backward else None
    torch.cuda.synchronize()

    def once():
        if backward:
            call(q, k, v, cfg).backward(grad)
        else:
            call(q, k, v, cfg)

    with torch_profiler(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            once()
        torch.cuda.synchronize()

    path = os.path.join(tempfile.gettempdir(), f"moba_trace_{os.getpid()}.json")
    prof.export_chrome_trace(path)
    try:
        with open(path, encoding="utf-8") as handle:
            events = json.load(handle).get("traceEvents", [])
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    collected: dict[str, KernelOccupancy] = {}
    for event in events:
        if event.get("cat") not in ("kernel", "Kernel"):
            continue
        args = event.get("args", {})
        record = collected.setdefault(event.get("name", "?"),
                                      KernelOccupancy(event.get("name", "?")))
        record.launches += 1
        record.microseconds += _as_float(event.get("dur"))
        # These are per launch and constant across launches of one kernel, so
        # the last wins rather than being averaged.
        record.achieved = _as_float(args.get("est. achieved occupancy %"))
        record.registers = int(_as_float(args.get("registers per thread")))
        record.shared = int(_as_float(args.get("shared memory")))
        block = args.get("block") or ()
        record.block_threads = int(_as_float(block[0])) if block else 0
        record.grid = tuple(args.get("grid", ()) or ())

        # `occupancy` is a nested object, not a number -- the field name reads
        # like a percentage and is a whole occupancy calculation. Reading it as
        # a float silently produced 0.0 for every kernel.
        occ = args.get("occupancy") or {}
        if isinstance(occ, dict):
            record.limiter = str(occ.get("limitingFactors", "?"))
            record.limits = {
                "REGS": int(_as_float(occ.get("blockLimitRegs"))),
                "SMEM": int(_as_float(occ.get("blockLimitSharedMem"))),
                "WARPS": int(_as_float(occ.get("blockLimitWarps"))),
                "BLOCKS": int(_as_float(occ.get("blockLimitBlocks"))),
            }

    if not collected:
        return [], ("no CUDA kernel events in the trace; CUPTI produced "
                    "nothing, which usually means activity tracing is also "
                    "restricted here")
    return list(collected.values()), ""


def print_occupancy(records: list[KernelOccupancy]) -> None:
    """Prints per-kernel residency and what limits it, busiest first."""
    import torch

    props = torch.cuda.get_device_properties(0)
    max_warps = props.max_threads_per_multi_processor // 32
    total = sum(r.microseconds for r in records) or 1.0

    print("")
    print(f"  {props.name}: {max_warps} warps per SM is 100% occupancy")
    print("")
    print(f"  {'kernel':<38} {'%time':>6} {'regs':>5} {'smem':>7} "
          f"{'blk/SM':>7} {'warps':>6} {'occ':>6}  limited by")
    print("  " + "-" * 96)
    for r in sorted(records, key=lambda r: -r.microseconds):
        warps = r.warps_resident
        pct = 100.0 * warps / max_warps if max_warps else 0.0
        detail = ""
        if r.limits:
            detail = " ".join(f"{k}={v}" for k, v in r.limits.items() if v > 0)
        print(f"  {r.short_name:<38} {100 * r.microseconds / total:>5.1f}% "
              f"{r.registers:>5} {r.shared:>7} {r.blocks_per_sm:>7} "
              f"{warps:>6.0f} {pct:>5.0f}%  {r.limiter:<6} {detail}")
    print("  " + "-" * 96)
    print("  blk/SM is the smallest positive block limit over registers, "
          "shared memory, warps")
    print("  and blocks; warps and occ follow from it. CUPTI reports "
          "blockLimitSharedMem=0 for")
    print("  any kernel opting in past the 48 KiB static limit -- it does not "
          "model the opt-in")
    print("  Triton performs -- so zeros are ignored rather than believed. Its "
          "own achieved")
    print("  estimate goes to 0 for the same reason, which is why it is not "
          "the headline here.")
    print("  Times are indicative only: tracing adds overhead, so read the "
          "ranking.")


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


def _run_torch_tier(args) -> None:
    """Runs the occupancy tier and prints it."""
    import torch

    from moba.config import MoBAConfig

    cfg = MoBAConfig(block_size=128, top_k=8, dtype=torch.bfloat16)
    records, diagnostic = torch_profile(args.seq_len, args.backend, cfg,
                                        backward=args.backward)
    if diagnostic:
        print(f"  torch.profiler produced nothing: {diagnostic}")
        return
    print_occupancy(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile a MoBA pass: DRAM counters, or occupancy.")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--backend", default="ours")
    parser.add_argument("--kernel", default="",
                        help="regex of kernel names to profile; empty means all")
    parser.add_argument("--backward", action="store_true",
                        help="profile the training step (torch tier only)")
    parser.add_argument("--tier", choices=("auto", "ncu", "torch"),
                        default="auto",
                        help="auto tries ncu counters, falls back to occupancy")
    args = parser.parse_args()

    if args.tier == "torch":
        print(f"occupancy tier: {args.backend} at S={args.seq_len}")
        _run_torch_tier(args)
        return

    if ncu_path() is None:
        print("ncu not found -- it ships with the CUDA toolkit and is usually "
              "off PATH.")
        if args.tier == "auto":
            print("falling back to the occupancy tier.\n")
            _run_torch_tier(args)
        return

    print(f"profiling {args.backend} at S={args.seq_len} "
          f"(ncu replays kernels; wall-clock here is meaningless)")
    profiles, diagnostic = profile(
        "bench._profile_target",
        ["--seq-len", str(args.seq_len), "--backend", args.backend],
        kernel_filter=args.kernel,
    )
    if diagnostic:
        print(f"\ncounter collection failed:\n  {diagnostic}")
        if args.tier == "auto":
            # The common case, not an edge case: counters were refused on both
            # machines this project has run on. Occupancy still answers the
            # question the roofline raised, so fall through rather than stop.
            print("\nfalling back to the occupancy tier (CUPTI activity "
                  "records, a different permission).")
            _run_torch_tier(args)
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
