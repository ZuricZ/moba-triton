"""Smoke tests: the package imports and the toolchain actually works.

These run first and fail loudest. Every later test assumes a GPU that meets
the project's architecture floor and a Triton that can compile for it; if
that assumption is wrong, the failure should say so here rather than surface
as a baffling error inside a kernel.
"""

import pytest
import torch
import triton
import triton.language as tl

import moba


def test_package_imports():
    assert moba.__version__ == "0.1.0"


def test_cuda_available():
    assert torch.cuda.is_available(), (
        f"No CUDA device visible from WSL. torch was built for CUDA "
        f"{torch.version.cuda}; a '+cu130' build cannot initialise against "
        f"this machine's 12.7 driver."
    )


def test_gpu_meets_compute_capability_floor():
    major, minor = torch.cuda.get_device_capability(0)
    assert (major, minor) >= (8, 0), (
        f"Project requires sm_80+ (Ampere) for bf16 tensor cores and "
        f"cp.async; found sm_{major}{minor}"
    )


def test_bfloat16_supported():
    x = torch.ones(8, 8, device="cuda", dtype=torch.bfloat16)
    assert (x @ x).dtype is torch.bfloat16


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Adds two vectors. The simplest possible proof that Triton compiles.

    One program instance handles BLOCK contiguous elements. `mask` guards the
    tail when n is not a multiple of BLOCK -- without it the final program
    would read and write out of bounds.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def test_triton_kernel_compiles_and_runs():
    """Proves the Triton compiler produces working code for this GPU.

    `import triton` succeeding only proves the package is installed. This
    exercises the whole path: JIT compilation, PTX generation for sm_86, and
    execution. A deliberately non-power-of-two length exercises the mask.
    """
    n = 1000
    x = torch.rand(n, device="cuda", dtype=torch.float32)
    y = torch.rand(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    # `grid` is how many independent program instances to launch. Each handles
    # BLOCK elements, so we need ceil(n / BLOCK) of them.
    block = 256
    grid = (triton.cdiv(n, block),)
    _add_kernel[grid](x, y, out, n, BLOCK=block)

    torch.testing.assert_close(out, x + y)


def test_triton_version_is_reported():
    assert triton.__version__, "triton must report a version"
