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
            total_memory_mb=props.total_memory // (1024 * 1024),
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
