"""GPU capability probing and result metadata.

Every benchmark row carries a DeviceInfo. That is what allows results measured
on this 4 GB laptop and on a rented A100/H100 to merge into one comparable
table rather than becoming two artifacts nobody can line up later.
"""

from dataclasses import dataclass

import torch

# Ampere. Below this there are no bf16 tensor cores and no cp.async, which
# together rule out the kernel design this project is built around.
MIN_CAPABILITY = (8, 0)

# --- Hardware facts -------------------------------------------------------
#
# These are properties of the GPU, not tuning choices. They live here so that
# a kernel, a benchmark and a test cannot disagree about them -- which they
# did: bench/load_balance.py hardcoded a tile height the kernel had stopped
# using, so the simulation was predicting a different kernel.

# Smallest matrix dimension `tl.dot` accepts. Ampere's MMA atom is 16x8x16, so
# an operand narrower than this cannot fill the tensor core at all.
TENSOR_CORE_TILE = 16

# Memory arrives in cache lines. This is the unit that makes a gathered row
# cost the same as a contiguous one, provided the row is at least this wide --
# the measured basis for decision D7 (`docs/gather-cost-measurements.md`).
CACHE_LINE_BYTES = 128

# Threads per warp -- the SIMT execution width. A tile is divided across
# `32 * num_warps` threads, so this is what converts a tile shape into
# registers per thread.
WARP_SIZE = 32

# Hard per-thread register limit on Ampere. Beyond this the compiler spills to
# HBM-backed local memory, silently, with no launch error.
MAX_REGISTERS_PER_THREAD = 255

# Used only when the device query fails. Matches sm_86's opt-in maximum.
FALLBACK_SHARED_MEMORY_BYTES = 101376

BYTES_PER_MIB = 1024 * 1024
MICROSECONDS_PER_MILLISECOND = 1000


def shared_memory_budget(device=None) -> int:
    """Shared memory available to one program, in bytes.

    Must be the *opt-in* maximum, not `shared_memory_per_block`. The latter is
    the 48 KiB static default every kernel gets without asking; a kernel raises
    its own limit via `cudaFuncSetAttribute`, which Triton does. On sm_86 the
    two differ by more than a factor of two -- 49152 against 101376 -- so using
    the default rejects configurations that in fact launch fine.

    Args:
        device: CUDA device to query; the current device when omitted.

    Returns:
        Bytes of shared memory available per program.
    """
    props = torch.cuda.get_device_properties(device)
    for attr in ("shared_memory_per_block_optin", "shared_memory_per_block"):
        value = getattr(props, attr, None)
        if value:
            return int(value)
    return FALLBACK_SHARED_MEMORY_BYTES


def multiprocessor_count(device=None) -> int:
    """Number of SMs on the device.

    Args:
        device: CUDA device to query; the current device when omitted.

    Returns:
        SM count.
    """
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def round_up_to_tensor_core_tile(x: int) -> int:
    """Rounds up to a multiple of `TENSOR_CORE_TILE`, with it as the floor.

    Args:
        x: Any non-negative size.

    Returns:
        The smallest multiple of TENSOR_CORE_TILE that is >= max(x, TILE).
    """
    tile = TENSOR_CORE_TILE
    return max(tile, ((x + tile - 1) // tile) * tile)


def check_gathered_head_dim(head_dim: int, itemsize: int) -> tuple[bool, str]:
    """Reports whether a head dimension makes row gathering efficient.

    A gathered row costs one cache-line fetch regardless of where it sits, so
    gathering is nearly free *provided a row fills a line*. Below that, a
    gathered row pulls a whole line and uses part of it, and the measured 1.03x
    penalty no longer applies.

    Args:
        head_dim: Head dimension.
        itemsize: Bytes per element of the input dtype.

    Returns:
        A tuple of (ok, reason). `reason` is empty when ok.
    """
    if head_dim & (head_dim - 1):
        return False, f"head_dim {head_dim} must be a power of two"
    row_bytes = head_dim * itemsize
    if row_bytes < CACHE_LINE_BYTES:
        return False, (
            f"head_dim {head_dim} gives a {row_bytes}-byte row, below the "
            f"{CACHE_LINE_BYTES}-byte cache line; gathered rows would waste "
            "the remainder of every line fetched"
        )
    return True, ""


def supports_dtype(
    dtype: torch.dtype, capability: tuple[int, int]
) -> tuple[bool, str]:
    """Reports whether a compute dtype is usable on a given GPU architecture.

    Returns a reason rather than raising, because callers use this to decide
    whether to *skip* work, not to abort. A skipped backend must be able to
    explain itself in a results table.

    Args:
        dtype: The candidate compute dtype.
        capability: CUDA compute capability as (major, minor).

    Returns:
        A tuple of (supported, reason). `reason` is empty when supported.
    """
    found = f"sm_{capability[0]}{capability[1]}"
    if dtype is torch.bfloat16 and capability < (8, 0):
        return False, f"bfloat16 requires sm_80+ (Ampere), found {found}"
    if dtype is torch.float16 and capability < (7, 0):
        return False, f"float16 tensor cores require sm_70+, found {found}"
    if dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        return False, f"unsupported dtype {dtype}"
    return True, ""


@dataclass(frozen=True)
class DeviceInfo:
    """Identifying metadata for the GPU a result was measured on."""

    name: str
    capability: tuple[int, int]
    total_memory_mb: int
    torch_version: str
    triton_version: str
    driver: str

    @classmethod
    def probe(cls, index: int = 0) -> "DeviceInfo":
        """Reads device metadata from the running CUDA context.

        Args:
            index: CUDA device index.

        Returns:
            A populated DeviceInfo.

        Raises:
            RuntimeError: If no CUDA device is available.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA device available")
        props = torch.cuda.get_device_properties(index)
        try:
            import triton

            triton_version = triton.__version__
        except ImportError:
            triton_version = "unavailable"
        return cls(
            name=props.name,
            capability=(props.major, props.minor),
            total_memory_mb=props.total_memory // BYTES_PER_MIB,
            torch_version=torch.__version__,
            triton_version=triton_version,
            driver=torch.version.cuda or "unknown",
        )

    def as_row(self) -> dict[str, str | int]:
        """Returns this device's metadata as flat columns for a results table.

        These key names are splatted directly into BenchRow, so they are an
        interface: renaming one breaks the benchmark suite.

        Returns:
            A dict of column name to value.
        """
        return {
            "gpu_name": self.name,
            "capability": f"sm_{self.capability[0]}{self.capability[1]}",
            "total_memory_mb": self.total_memory_mb,
            "torch_version": self.torch_version,
            "triton_version": self.triton_version,
            "cuda_version": self.driver,
        }
