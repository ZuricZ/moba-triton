# Variable-Length (Packed Sequence) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the MoBA kernels attend over a batch of sequences of *different*
lengths packed end to end, addressed by a `cu_seqlens` array, with the existing
padded rectangular path unchanged and byte-identical.

**Architecture:** Packed mode keeps the four-dimensional layout and pins
`B = 1`, so `[1, H, T, D]` holds every token of every sequence. **No tensor
changes shape family and no kernel's address arithmetic changes at all.** What
changes is that blocks are cut *per sequence* rather than every `block_size`
tokens, so a block's key range comes from two small lookup arrays instead of
`blk * BLOCK_SIZE`. A host-built tile→sequence map guarantees no query tile ever
straddles a sequence boundary, which keeps the gate's streaming top-k
untouched.

**Tech Stack:** Triton 3.7.1, PyTorch 2.13.0+cu126, Python 3.13, pytest.

**Spec:** No separate design doc; decisions were settled in conversation and are
recorded under "Design decisions" below. Background:
`docs/superpowers/specs/2026-08-18-moba-kernel-design.md` (D6 KV-block-major,
D8 longest-first dispatch, D10 fixed-slot partials) and
`docs/superpowers/plans/2026-08-23-gqa-support.md` (the head-axis split this
builds on).

---

## Global Constraints

- **Google Style** for all code and docstrings (Args/Returns/Raises, 4-space
  indent). Project-wide, per `CLAUDE.md`.
- **Never edit `references/`.** Vendored, read-only, gitignored.
- **No `Co-Authored-By:` lines** referencing Claude or Anthropic in commits.
- **Run everything through WSL**, never native Windows:
  ```bash
  wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; \
    export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; \
    cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run <command>'
  ```
- **`git` runs from PowerShell**, not WSL — git identity is set on Windows only.
- **4 GiB VRAM.** Keep test totals small: `total_tokens` ≤ 2048 for correctness
  work, sequences in the tens to low hundreds.
- **Correctness policy is `moba.tolerance`**, never an absolute threshold: pass
  if our error is within `TOLERANCE_FACTOR` (2.0) of the same-dtype oracle's own
  deviation from an fp64 truth.
- **Measured hardware:** RTX 3050 Ti Laptop, sm_86, 20 SMs, 101376 B shared
  memory per block.
- **The padded path must not change.** 379 tests and every published benchmark
  number depend on it. Any diff that alters padded behaviour is a bug in this
  work, not an improvement.

---

## Design decisions

Each was chosen against a named alternative, several of them after reading the
references. Implementers should not silently re-decide them.

**D-VL-1 — Packed mode keeps four dimensions and pins `B = 1`.**
`q` is `[1, H_q, T, D]`, not `[T, H_q, D]`. Borrowed from fla-org's NSA, which
asserts exactly this (`ops/naive.py:84`: *"batch size must be 1 when cu_seqlens
are provided"*). *Rejected:* FlashAttention's `[T, H, D]` convention. It is the
better public convention and the wrong internal one **for us**: our kernels
address queries as `pid_bh * S * D + row * D`, and with `B = 1` that is `head *
T * D + token * D` — already correct, character for character. Adopting
`[T, H, D]` would rewrite the address arithmetic of six kernels to buy nothing.
Conversion at the API edge is a transpose the caller can do if they want FA's
layout.

**D-VL-2 — One entry point, `cu_seqlens=None` meaning padded.**
`moba_attention(q, k, v, cfg, cu_seqlens=None)`. *Rejected:* a separate
`moba_attention_varlen`. FlashMoBA's own C++ does it the single-entry way —
`Varlen` is a template parameter and `cu_seqlens_q == nullptr` selects padded
(`csrc/flash_moba/src/block_info.h:19-25`). Also rejected: making packed the
only internal form and having padded callers pack first, which costs a copy of
Q, K and V — ~25 MB at S=8192, ≈300 µs against a 2280 µs forward, about 13%.

**D-VL-3 — One `SeqLayout` helper answers every raggedness question.**
Modelled on FlashMoBA's `BlockInfo` struct, whose whole job is to yield
`actual_seqlen_q` and the row offset for both modes so that downstream code
never asks which mode it is in. *Rejected:* `if VARLEN:` branches scattered
through six kernels. The difference must live in one place or it will be got
wrong in six.

**D-VL-3b — `SeqLayout` is built on the CPU and transferred once.**
`from_cu_seqlens` takes a **CPU** tensor or a plain list of ints, does every
`int()` extraction and every derivation there, and moves the four device-side
arrays over in one transfer. *Rejected:* building it from a CUDA `cu_seqlens`.
That reads `int(cu[-1])`, `int(cu_blocks[-1])` and `int(lengths.max())` off the
device — **three host synchronizations**, measured at ~230 us each on this WSL
box (`index_fused.py` module docstring), against a 2280 us forward. Roughly 30%
of the pass, reintroducing precisely what the fused index builder was written to
remove.

This is not a workaround. `cu_seqlens` *originates* on the host: the data loader
knows the lengths before it packs anything. Keeping the layout there is the same
rule that puts it in PyTorch rather than Triton -- **work that scales with the
number of sequences belongs on the host; work that scales with tokens belongs on
the device.** A `SeqLayout` is also then cacheable across calls with the same
batch shape.

If a caller only has a CUDA `cu_seqlens`, they pay one sync to move it and are
told so by the docstring, rather than paying three silently.

**D-VL-4 — Blocks are cut per sequence; a block never straddles a boundary.**
Not a preference. Cutting on the packed buffer would put tokens from two
documents in one block, and a query selecting it would attend across the
boundary — well-formed, plausible, silent. Each sequence gets
`ceil(len_b / block_size)` blocks, and each may end short. Block ids are
**global** across the packed batch, with `cu_blocks` giving each sequence's
first.

**D-VL-5 — A host-built tile→sequence map; query tiles never straddle.**
Borrowed from NSA (`ops/parallel.py:64`), whose `token_indices` array holds one
`(sequence_id, local_offset)` pair per tile and is indexed by the grid.
*Rejected:* looping over the union of the tile's sequences' candidate ranges
and masking. The union wastes most lanes on every iteration, and — the part
that is easy to miss — **the causal mask alone would not make it correct**. A
block belonging to an earlier sequence is causally *before* the current tile's
tokens, so `key_pos <= rows` passes; the union form needs an extra
same-sequence comparison in the hottest loop, guarding a case the map removes
for free. Ragged work is cheaper to fix on the host than in the kernel.

**D-VL-6 — The gate ranks in sequence-local block indices and adds an offset.**
Inside `_moba_gate_fwd`, `cols` stays `0 .. n_blocks_of_this_sequence`, and the
sequence's block offset `boc` is added when loading `Kbar` and when emitting the
chosen ids. Because D-VL-5 guarantees a tile belongs to one sequence, `bos`,
`boc` and the sequence length are **scalars**, not per-lane vectors — which is
exactly what keeps the streaming top-k, its tie-break, and its horizon
computation a zero-line diff. *Rejected:* ranking in global indices, which would
make the loop's start a per-lane value and the horizon logic per-lane with it.

**D-VL-7 — The causal mask is not modified, and the gate is what makes that safe.**
`qk += tl.where(key_pos <= rows, 0.0, -inf)` compares packed token indices.
Within a sequence, packed order *is* position order, so the comparison is
correct **provided a query is never offered a block from another sequence**.
That invariant is enforced entirely by the gate (D-VL-6). This is the one place
in the design where correctness of an untouched line is delegated to a distant
one; it must be commented on both sides and tested directly
(`test_no_selected_block_crosses_a_sequence_boundary`).

**D-VL-8 — `_moba_combine`, `_moba_dq_reduce` and `_moba_bwd_preprocess` are not modified.**
All three stream over `[B, H_q, S, ...]` and launch over `b * h_q`. With
`B = 1` and `S = T` they are already correct. As under GQA: an implementer who
finds themselves editing these has an upstream problem.

### What changes, and what does not

| | padded | packed | changed? |
|---|---|---|---|
| `q` | `[B, H_q, S, D]` | `[1, H_q, T, D]` | **same shape family** |
| `k`, `v` | `[B, H_kv, S, D]` | `[1, H_kv, T, D]` | same |
| `part_m`, `part_l`, `part_acc` | `[B, H_q, S, K(,D)]` | `[1, H_q, T, K(,D)]` | same |
| `lse`, `delta` | `[B, H_q, S]` | `[1, H_q, T]` | same |
| CSC arrays | `[B, H_kv, N]` | `[1, H_kv, N_total]` | same |
| attention grid | `N × B × H_kv` | `N_total × 1 × H_kv` | same |
| **kernel address arithmetic** | — | — | **none** |
| **tile shapes, registers, SRAM** | — | — | **none** |
| **tuning constants** | — | — | **none** |
| a block's key range | `blk * BLOCK_SIZE + arange`, `< S` | `BlockStart[blk] + arange`, `< BlockEnd[blk]` | **two loads** |
| gate candidate range | `0 .. own_block` | `0 .. own_block` **local**, `+boc` on access | **one offset** |
| gate grid | `cdiv(S, BLOCK_M) × B × H_q` | `n_tiles × H_q`, tiles from a map | **new map** |

### New metadata, all derived from `cu_seqlens`

For `cu_seqlens = [0, 5, 8, 12]` with `block_size = 2`:

```
packed token:  0  1  2  3  4 | 5  6  7 | 8  9 10 11
blocks:       [0,1][2,3][4]  |[5,6][7] |[8,9][10,11]
block id:       0    1   2   |  3   4  |  5     6

cu_blocks   = [0, 3, 5, 7]          # per sequence, B+1
block_start = [0, 2, 4, 5, 7, 8, 10]    # per block, N_total
block_end   = [2, 4, 5, 7, 8, 10, 12]   # per block, N_total
seq_of_block= [0, 0, 0, 1, 1, 2, 2]     # per block, N_total
```

`block_end - block_start` is the true occupancy, which
`_block_representatives_fwd` needs so a short tail is not divided by
`block_size`.

---

## File Structure

**Create:**
- `src/moba/seqlayout.py` — `SeqLayout`, the single source of truth for
  raggedness. Builds and validates every array above plus the tile map. The
  `BlockInfo` analogue.
- `tests/test_seqlayout.py` — unit tests, CPU where possible.
- `tests/test_varlen.py` — end-to-end packed correctness, forward and backward.
- `docs/varlen-measurements.md` — padding waste removed, grid effect, and a
  head-to-head against nested-tensor dense SDPA.

**Modify:**
- `src/moba/ref/naive.py` — oracle accepts `cu_seqlens`.
- `src/moba/kernels/gate.py` — `_block_representatives_fwd` and
  `_moba_gate_fwd` gain block ranges and the tile map.
- `src/moba/kernels/index_fused.py`, `index.py` — `num_blocks` becomes
  `N_total`; verify nothing else needs touching.
- `src/moba/kernels/moba_fwd.py` — `_validate`, `_moba_attn_fwd` key range.
- `src/moba/kernels/moba_bwd.py` — `_moba_bwd` key range.
- `src/moba/kernels/autograd.py` — plumb `cu_seqlens`.
- `src/moba/backends/sdpa_dense.py` — nested-tensor varlen baseline.
- `bench/suite.py` — a packed row.

**Untouched by design:** `_moba_combine`, `_moba_dq_reduce`,
`_moba_bwd_preprocess`, `moba/config.py`, `moba/tolerance.py`,
`moba/device.py`, `moba/heads.py`.

---

## Task 1: `SeqLayout` and the oracle

Pure PyTorch, no kernels. Establishes the definition everything else is checked
against and gets "packed with one sequence equals padded with batch 1" nailed
down before any Triton moves.

**Files:**
- Create: `src/moba/seqlayout.py`, `tests/test_seqlayout.py`
- Modify: `src/moba/ref/naive.py`
- Test: `tests/test_seqlayout.py`

**Interfaces:**
- Produces:
  - `SeqLayout.from_cu_seqlens(cu_seqlens: torch.Tensor, block_size: int, tile: int) -> SeqLayout`
  - `SeqLayout.uniform(batch: int, seq_len: int, block_size: int, tile: int) -> SeqLayout`
  - Fields: `cu_seqlens [B+1]`, `cu_blocks [B+1]`, `block_start [N]`,
    `block_end [N]`, `seq_of_block [N]`, `tile_map [n_tiles, 2]`,
    `total_tokens: int`, `total_blocks: int`, `num_sequences: int`,
    `max_seqlen: int`
  - `moba.ref.naive.moba_attention_naive(..., cu_seqlens=None)` and
    `moba_selection(..., cu_seqlens=None)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_seqlayout.py`:

```python
"""Ragged-batch metadata: everything derived from `cu_seqlens`, once.

`SeqLayout` is the only place in the codebase that knows sequences can have
different lengths. Every kernel reads its arrays and none of them recompute a
boundary, which is the point -- a boundary computed in two places is a boundary
that will disagree in one of them.

Modelled on FlashMoBA's `BlockInfo`
(`references/flash-moba/csrc/flash_moba/src/block_info.h`), whose whole job is
to answer "where does my sequence start and how long is it" so downstream code
never asks which mode it is in.
"""

import pytest
import torch

from moba.seqlayout import SeqLayout


def test_the_worked_example_from_the_plan(device):
    """Lengths 5, 3, 4 at block_size 2 -- countable by hand."""
    cu = torch.tensor([0, 5, 8, 12], dtype=torch.int32, device=device)
    lay = SeqLayout.from_cu_seqlens(cu, block_size=2, tile=4)

    assert lay.num_sequences == 3
    assert lay.total_tokens == 12
    assert lay.total_blocks == 7          # 3 + 2 + 2
    assert lay.max_seqlen == 5

    assert lay.cu_blocks.tolist() == [0, 3, 5, 7]
    assert lay.block_start.tolist() == [0, 2, 4, 5, 7, 8, 10]
    assert lay.block_end.tolist() == [2, 4, 5, 7, 8, 10, 12]
    assert lay.seq_of_block.tolist() == [0, 0, 0, 1, 1, 2, 2]


def test_no_block_straddles_a_sequence_boundary(device):
    """The one thing that must never happen.

    A block holding tokens from two documents would let a query attend across
    the boundary. It would not crash: the shapes stay right and the numbers
    stay finite.
    """
    cu = torch.tensor([0, 5, 8, 12, 13], dtype=torch.int32, device=device)
    lay = SeqLayout.from_cu_seqlens(cu, block_size=2, tile=4)

    for blk in range(lay.total_blocks):
        lo = int(lay.block_start[blk])
        hi = int(lay.block_end[blk])
        seq = int(lay.seq_of_block[blk])
        assert int(lay.cu_seqlens[seq]) <= lo < hi <= int(lay.cu_seqlens[seq + 1])


def test_every_token_belongs_to_exactly_one_block(device):
    """Blocks partition the packed buffer -- no gap, no overlap."""
    cu = torch.tensor([0, 5, 8, 12, 13], dtype=torch.int32, device=device)
    lay = SeqLayout.from_cu_seqlens(cu, block_size=2, tile=4)

    covered = []
    for blk in range(lay.total_blocks):
        covered.extend(range(int(lay.block_start[blk]),
                             int(lay.block_end[blk])))
    assert covered == list(range(lay.total_tokens))


def test_no_query_tile_straddles_a_sequence_boundary(device):
    """`tile_map` is what makes `bos` a scalar inside the gate.

    A tile spanning two sequences would give its lanes different candidate
    ranges, and a Triton program has one loop bound for all its lanes.
    """
    cu = torch.tensor([0, 5, 8, 12], dtype=torch.int32, device=device)
    lay = SeqLayout.from_cu_seqlens(cu, block_size=2, tile=4)

    # ceil(5/4) + ceil(3/4) + ceil(4/4) = 2 + 1 + 1
    assert lay.tile_map.shape == (4, 2)
    for seq, off in lay.tile_map.tolist():
        start = int(lay.cu_seqlens[seq]) + off
        length = int(lay.cu_seqlens[seq + 1]) - int(lay.cu_seqlens[seq])
        assert 0 <= off < length
        assert start < int(lay.cu_seqlens[seq + 1])


def test_uniform_layout_matches_a_padded_batch(device):
    """`uniform` is the packed description of the rectangular case.

    Used by tests to run one code path against the other; the padded kernels
    never construct it.
    """
    lay = SeqLayout.uniform(batch=3, seq_len=4, block_size=2, tile=4)
    assert lay.cu_seqlens.tolist() == [0, 4, 8, 12]
    assert lay.total_blocks == 6
    assert lay.block_start.tolist() == [0, 2, 4, 6, 8, 10]


def test_a_sequence_shorter_than_one_block_gets_one_short_block(device):
    cu = torch.tensor([0, 1], dtype=torch.int32, device=device)
    lay = SeqLayout.from_cu_seqlens(cu, block_size=8, tile=4)
    assert lay.total_blocks == 1
    assert lay.block_start.tolist() == [0]
    assert lay.block_end.tolist() == [1]


def test_building_a_layout_does_not_synchronize_with_the_host(device):
    """Every scalar must come from CPU data, never off the device.

    Reading `int(cu[-1])` from a CUDA tensor costs ~230 us on this box, and
    there are three such reads to make. `index_fused` was written specifically
    to remove synchronizations of that kind; reintroducing them here would cost
    about 30% of a forward pass and look like a kernel regression.
    """
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        SeqLayout.from_cu_seqlens([0, 20, 32, 48], block_size=4, tile=32,
                                  device=device)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.parametrize("bad", [
    [1, 5],          # does not start at zero
    [0, 5, 3],       # not monotone
    [0, 0, 5],       # empty sequence
])
def test_malformed_cu_seqlens_are_rejected(device, bad):
    cu = torch.tensor(bad, dtype=torch.int32, device=device)
    with pytest.raises(ValueError):
        SeqLayout.from_cu_seqlens(cu, block_size=2, tile=4)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_seqlayout.py -q` (WSL prefix).
Expected: `ModuleNotFoundError: No module named 'moba.seqlayout'`.

- [ ] **Step 3: Write `src/moba/seqlayout.py`**

```python
"""Everything the kernels need to know about a ragged batch, derived once.

A packed batch stores sequences of different lengths end to end in one buffer,
with `cu_seqlens` -- cumulative sequence lengths -- marking the boundaries.
There is no `S` dimension because there is no single sequence length, so the
raggedness has to live somewhere; this module is that somewhere, and the only
one. A boundary computed in two places is a boundary that will disagree.

Modelled on FlashMoBA's `BlockInfo` struct, whose entire job is to answer
"where does my sequence start, and how long is it?" for both the padded and the
packed case so that every kernel downstream reads one field and never asks
which mode it is in.

BLOCKS ARE CUT PER SEQUENCE. Cutting every `block_size` tokens of the packed
buffer would place tokens from two documents in one block, and a query
selecting that block would attend across the boundary. That failure does not
crash: the shapes stay correct and the numbers stay finite. Each sequence
therefore gets its own `ceil(len / block_size)` blocks, the last of which may be
short -- exactly the case `_block_representatives_fwd` already handles for the
tail of a padded sequence, now arising once per sequence rather than once per
batch.

QUERY TILES ARE ALSO CUT PER SEQUENCE, via `tile_map`. A Triton program has one
loop bound for all of its lanes, so a tile holding queries from two sequences
would need the union of their candidate ranges and a per-lane mask on every
iteration. Precomputing the map costs one small host-side array and makes the
launch grid exact. Borrowed from fla-org's NSA, whose `token_indices` serves the
same purpose (`references/native-sparse-attention/.../ops/parallel.py:64`).
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SeqLayout:
    """Block and tile structure of one packed batch.

    Frozen because kernels are launched against it; mutating one after an index
    was built would silently invalidate the index.

    Attributes:
        cu_seqlens: Cumulative token counts, [num_sequences + 1] int32. Sequence
            `i` occupies packed tokens `cu_seqlens[i] : cu_seqlens[i + 1]`.
        cu_blocks: Cumulative block counts, [num_sequences + 1] int32. Sequence
            `i` owns global blocks `cu_blocks[i] : cu_blocks[i + 1]`.
        block_start: First packed token of each block, [total_blocks] int32.
        block_end: One past the last packed token of each block,
            [total_blocks] int32. `block_end - block_start` is the block's true
            occupancy, which the mean-pool needs so a short tail is not divided
            by `block_size`.
        seq_of_block: Which sequence each block belongs to, [total_blocks]
            int32.
        tile_map: One `(sequence, local_token_offset)` pair per query tile,
            [num_tiles, 2] int32. Indexed by the gate's grid.
        tile: Queries per tile that `tile_map` was built for. Kept so the gate
            can assert the launch matches; a layout built for one tile size and
            launched with another produces a map that silently misaddresses
            every program.
        total_tokens: Packed token count.
        total_blocks: Block count across all sequences.
        num_sequences: Sequence count.
        max_seqlen: Longest sequence, in tokens.
    """

    cu_seqlens: torch.Tensor
    cu_blocks: torch.Tensor
    block_start: torch.Tensor
    block_end: torch.Tensor
    seq_of_block: torch.Tensor
    tile_map: torch.Tensor
    tile: int
    total_tokens: int
    total_blocks: int
    num_sequences: int
    max_seqlen: int

    @classmethod
    def from_cu_seqlens(
        cls, cu_seqlens: torch.Tensor | list[int], block_size: int, tile: int,
        device=None,
    ) -> "SeqLayout":
        """Derives the full layout from cumulative sequence lengths.

        Everything is computed on the CPU and the device arrays are transferred
        once; see D-VL-3b. Passing a CUDA `cu_seqlens` costs one synchronization
        to bring it back, which is one instead of the three that reading scalars
        off it would cost.

        Args:
            cu_seqlens: Cumulative token counts, [num_sequences + 1], on the CPU
                or as a plain list. Must start at zero, be strictly increasing,
                and hold no empty sequence.
            block_size: Keys per MoBA block.
            tile: Queries per gate program, so the tile map can be built.
            device: Where the four device-side arrays should live. Defaults to
                CPU, which is what the unit tests want.

        Returns:
            A populated SeqLayout with its arrays on `device`.

        Raises:
            ValueError: If `cu_seqlens` is malformed or the sizes are not
                positive.
        """
        if block_size <= 0 or tile <= 0:
            raise ValueError(
                f"block_size and tile must be positive, got {block_size}, {tile}"
            )
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError(
                f"cu_seqlens must be 1-D with at least two entries, got "
                f"{tuple(cu_seqlens.shape)}"
            )
        cu = cu_seqlens.to(torch.int64)
        if int(cu[0]) != 0:
            raise ValueError(f"cu_seqlens must start at 0, got {int(cu[0])}")
        lengths = cu[1:] - cu[:-1]
        if bool((lengths <= 0).any()):
            raise ValueError(
                f"cu_seqlens must be strictly increasing (no empty sequence), "
                f"got lengths {lengths.tolist()}"
            )

        n_seq = lengths.numel()
        blocks_per = (lengths + block_size - 1) // block_size
        cu_blocks = torch.zeros(n_seq + 1, dtype=torch.int64, device=device)
        cu_blocks[1:] = blocks_per.cumsum(0)
        total_blocks = int(cu_blocks[-1])

        # Block b of sequence i starts at cu_seqlens[i] + b * block_size and
        # stops at the sequence's end, which is what makes the last block short
        # rather than overrunning into the next document.
        seq_of_block = torch.repeat_interleave(
            torch.arange(n_seq, device=device), blocks_per
        )
        local_block = (torch.arange(total_blocks, device=device)
                       - cu_blocks[seq_of_block])
        block_start = cu[seq_of_block] + local_block * block_size
        block_end = torch.minimum(block_start + block_size,
                                  cu[seq_of_block + 1])

        tiles_per = (lengths + tile - 1) // tile
        seq_of_tile = torch.repeat_interleave(
            torch.arange(n_seq, device=device), tiles_per
        )
        cu_tiles = torch.zeros(n_seq + 1, dtype=torch.int64, device=device)
        cu_tiles[1:] = tiles_per.cumsum(0)
        local_tile = (torch.arange(int(cu_tiles[-1]), device=device)
                      - cu_tiles[seq_of_tile])
        tile_map = torch.stack([seq_of_tile, local_tile * tile], dim=1)

        i32 = torch.int32
        return cls(
            cu_seqlens=cu.to(i32),
            cu_blocks=cu_blocks.to(i32),
            block_start=block_start.to(i32),
            block_end=block_end.to(i32),
            seq_of_block=seq_of_block.to(i32),
            tile_map=tile_map.to(i32).contiguous().to(device),
            tile=tile,
            total_tokens=int(cu[-1]),
            total_blocks=total_blocks,
            num_sequences=n_seq,
            max_seqlen=int(lengths.max()),
        )

    @classmethod
    def uniform(
        cls, batch: int, seq_len: int, block_size: int, tile: int,
        device=None,
    ) -> "SeqLayout":
        """Builds the layout a padded batch would have, if packed.

        Test support: it lets the packed path be run against the padded one on
        the same data. The padded kernels never construct this -- they need no
        layout at all, which is the point of keeping their arithmetic untouched.

        Args:
            batch: Sequence count.
            seq_len: Length of every sequence.
            block_size: Keys per MoBA block.
            tile: Queries per gate program.
            device: Where to build the arrays.

        Returns:
            A populated SeqLayout.
        """
        cu = torch.arange(batch + 1, device=device,
                          dtype=torch.int32) * seq_len
        return cls.from_cu_seqlens(cu, block_size, tile)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_seqlayout.py -q`
Expected: 9 passed.

- [ ] **Step 5: Write the failing oracle test**

Append to `tests/test_seqlayout.py`:

```python
def test_packed_oracle_equals_per_sequence_padded_oracle(device):
    """The definition of varlen: identical to running each sequence alone.

    Checked bitwise. Slicing the packed buffer gives exactly the same numbers,
    so any difference is a boundary or indexing bug, not rounding.
    """
    from moba.config import MoBAConfig
    from moba.ref.naive import moba_attention_naive

    cfg = MoBAConfig(block_size=4, top_k=2, dtype=torch.float64)
    lens = [20, 12, 16]
    cu = torch.tensor([0, 20, 32, 48], dtype=torch.int32, device=device)
    gen = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(1, 2, 48, 16, generator=gen, device=device,
                    dtype=torch.float64)
    k = torch.randn(1, 2, 48, 16, generator=gen, device=device,
                    dtype=torch.float64)
    v = torch.randn(1, 2, 48, 16, generator=gen, device=device,
                    dtype=torch.float64)

    packed = moba_attention_naive(q, k, v, cfg, cu_seqlens=cu)

    lo = 0
    for n in lens:
        piece = moba_attention_naive(
            q[:, :, lo:lo + n], k[:, :, lo:lo + n], v[:, :, lo:lo + n], cfg)
        assert torch.equal(packed[:, :, lo:lo + n], piece)
        lo += n
```

- [ ] **Step 6: Run it to verify it fails**

Expected: `TypeError: moba_attention_naive() got an unexpected keyword
argument 'cu_seqlens'`.

- [ ] **Step 7: Add `cu_seqlens` to the oracle**

In `src/moba/ref/naive.py`, give `moba_selection` and `moba_attention_naive` a
`cu_seqlens: torch.Tensor | None = None` keyword. When set, require `B == 1` and
loop over sequences, slicing and delegating to the existing per-sequence code:

```python
    # Deliberately a Python loop. This is the oracle: one line per line of the
    # algorithm, and "a packed batch means each sequence attends alone" is the
    # definition being encoded. fla-org's NSA reference does the same
    # (`ops/naive.py:111`), and so does FSA's production path, which loops in
    # Python with a TODO to fuse it -- the fusing is the kernel's job, not the
    # definition's.
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"packed input must have batch 1, got {q.shape[0]}"
            )
        out = torch.empty_like(q)
        bounds = cu_seqlens.tolist()
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            out[:, :, lo:hi] = moba_attention_naive(
                q[:, :, lo:hi], k[:, :, lo:hi], v[:, :, lo:hi], cfg,
                query_chunk=query_chunk,
            )
        return out
```

`moba_selection` takes the same shape, returning
`[1, H, T, max_blocks_in_any_sequence]` — see Step 8 for why that shape.

- [ ] **Step 8: Give `moba_selection` the same treatment**

The packed selection mask uses **sequence-local** block ids, width
`max_blocks_in_any_sequence`. Not `total_blocks`: such a mask would be almost
entirely false, growing with the batch for no information -- at 32 sequences it
is ~32x the memory of the useful part.

`csc_to_mask` must reproduce the identical convention in Task 4, so the
conversion is written down once, here, in the docstring:

```python
    # Block ids are sequence-local in this mask and global in the CSC index.
    # The conversion is `global = cu_blocks[seq] + local`, and it lives in
    # exactly two functions -- this one producing local ids and `csc_to_mask`
    # reproducing them. Everywhere else, ids are global.
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"packed input must have batch 1, got {q.shape[0]}"
            )
        bounds = cu_seqlens.tolist()
        width = max(-(-(hi - lo) // cfg.block_size)
                    for lo, hi in zip(bounds[:-1], bounds[1:]))
        out = torch.zeros(1, q.shape[1], q.shape[2], width,
                          dtype=torch.bool, device=q.device)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            m = moba_selection(q[:, :, lo:hi], k[:, :, lo:hi], cfg)
            out[:, :, lo:hi, :m.shape[-1]] = m
        return out
```

- [ ] **Step 9: Run the oracle-facing suite**

Run: `uv run pytest tests/test_seqlayout.py tests/test_semantics.py tests/test_smoke.py -q`
Expected: all pass. Existing tests pass no `cu_seqlens`, so they take the
unchanged branch.

- [ ] **Step 10: Commit**

```bash
git add src/moba/seqlayout.py tests/test_seqlayout.py src/moba/ref/naive.py docs/superpowers/plans/2026-08-24-varlen-support.md
git commit -m "feat: SeqLayout, the one place that knows sequences differ"
```

---

## Task 2: Block representatives over a ragged batch

Smallest kernel change, and it isolates the "blocks are cut per sequence" idea
before anything depends on it.

**Files:**
- Modify: `src/moba/kernels/gate.py` — `_block_representatives_fwd`,
  `block_representatives_triton`
- Test: `tests/test_gate.py`

**Interfaces:**
- Consumes: `SeqLayout` (Task 1).
- Produces: `block_representatives_triton(k, cfg, layout=None)` returning
  `[1, H_kv, total_blocks, D]` when a layout is given, `[B, H, num_blocks, D]`
  otherwise.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gate.py`:

```python
def test_block_representatives_are_pooled_per_sequence(device):
    """A block must average its own sequence's keys and stop at the boundary.

    The bug this catches is a block straddling two sequences: its mean would mix
    documents, every downstream stage would accept it, and the numbers would
    stay finite. Compared against per-sequence pooling of the same keys.
    """
    from moba.seqlayout import SeqLayout

    cfg = MoBAConfig(block_size=4, top_k=2, dtype=torch.bfloat16)
    cu = torch.tensor([0, 20, 32, 48], dtype=torch.int32, device=device)
    lay = SeqLayout.from_cu_seqlens(cu, cfg.block_size, tile=32)
    gen = torch.Generator(device=device).manual_seed(0)
    k = torch.randn(1, 2, 48, 16, generator=gen, device=device,
                    dtype=torch.bfloat16)

    got = block_representatives_triton(k, cfg, layout=lay)
    assert got.shape == (1, 2, lay.total_blocks, 16)

    lo = 0
    off = 0
    for n in [20, 12, 16]:
        want = block_representatives_triton(k[:, :, lo:lo + n], cfg)
        nb = want.shape[2]
        assert torch.equal(got[:, :, off:off + nb], want)
        lo += n
        off += nb
    assert off == lay.total_blocks
```

- [ ] **Step 2: Run to verify it fails**

Expected: `TypeError: block_representatives_triton() got an unexpected keyword
argument 'layout'`.

- [ ] **Step 3: Give the kernel a block range**

Add `BlockStart, BlockEnd` pointers and `VARLEN: tl.constexpr` to
`_block_representatives_fwd`, and replace the row computation:

```python
    if VARLEN:
        # Blocks are cut per sequence, so a block's extent is looked up rather
        # than computed. The tail case this creates is the one this kernel
        # already handled for a padded sequence's last block -- it now arises
        # once per sequence instead of once per batch.
        lo = tl.load(BlockStart + blk)
        hi = tl.load(BlockEnd + blk)
    else:
        lo = blk * BLOCK_SIZE
        hi = tl.minimum(lo + BLOCK_SIZE, S)

    rows = lo + tl.arange(0, BLOCK_POW2)
    live = (rows < hi) & (tl.arange(0, BLOCK_POW2) < BLOCK_SIZE)
    occupancy = hi - lo
```

and the store, whose row stride is now `N_BLOCKS` passed in rather than
`tl.cdiv(S, BLOCK_SIZE)`:

```python
    tl.store(Kbar + pid_bh * N_BLOCKS * D + blk * D + offs_d,
             mean.to(Kbar.dtype.element_ty))
```

Pass `N_BLOCKS` as a runtime argument in both modes so the two paths differ only
in the range lookup.

- [ ] **Step 4: Update the wrapper**

```python
def block_representatives_triton(
    k: torch.Tensor, cfg: MoBAConfig, layout: "SeqLayout | None" = None
) -> torch.Tensor:
```

When `layout` is None the grid and shapes are exactly as today. When given,
require `k.shape[0] == 1`, use `n = layout.total_blocks`, grid
`(n, 1 * h)`, and pass the two arrays with `VARLEN=True`.

- [ ] **Step 5: Run the gate suite**

Run: `uv run pytest tests/test_gate.py -q`
Expected: 39 passed — 38 existing plus the new one.

- [ ] **Step 6: Commit**

```bash
git add src/moba/kernels/gate.py tests/test_gate.py
git commit -m "feat: block representatives pooled per sequence"
```

---

## Task 3: The gate over a ragged batch

The correctness linchpin. D-VL-7 delegates the causal mask's correctness here.

**Files:**
- Modify: `src/moba/kernels/gate.py` — `_moba_gate_fwd`, `gate_select`
- Test: `tests/test_gate.py`, `tests/test_varlen.py`

**Interfaces:**
- Consumes: `SeqLayout`, ragged `kbar` (Task 2).
- Produces: `gate_select(q, kbar, cfg, layout=None)` returning
  `[1, H_q, T, top_k]` int32 of **global** block ids, -1 padded.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_varlen.py` with the boundary test first — it is the whole
point of the task:

```python
"""Packed (variable-length) batches, end to end.

Anchored the same way throughout: the packed kernel must agree with the *same
code path* run on each sequence separately. Everything else -- arithmetic,
routing, tuning -- is identical on both sides, so a disagreement can only be a
boundary or indexing bug.

The failure to fear does not crash. A block that straddles a sequence boundary,
or a gate that offers a query another document's block, produces finite,
plausible numbers of exactly the right shape.

See `docs/superpowers/plans/2026-08-24-varlen-support.md`.
"""

import pytest
import torch

from moba.config import MoBAConfig
from moba.kernels.gate import block_representatives_triton, gate_select
from moba.seqlayout import SeqLayout

LENS = [20, 12, 16, 5]
CU = [0, 20, 32, 48, 53]


def _packed(device, dtype, heads=2, d=16, seed=0, total=53):
    gen = torch.Generator(device=device).manual_seed(seed)
    mk = lambda: torch.randn(1, heads, total, d, generator=gen,
                             device=device, dtype=dtype)
    return mk(), mk(), mk()


def test_no_selected_block_crosses_a_sequence_boundary(device):
    """The invariant the untouched causal mask depends on.

    `qk += tl.where(key_pos <= rows, ...)` compares packed token indices. Within
    a sequence packed order is position order, so it is correct -- but only
    because a query is never offered a block from another sequence. Nothing in
    the attention kernel checks that. This is the check.
    """
    cfg = MoBAConfig(block_size=4, top_k=3, dtype=torch.bfloat16)
    cu = torch.tensor(CU, dtype=torch.int32, device=device)
    lay = SeqLayout.from_cu_seqlens(cu, cfg.block_size, tile=32)
    q, k, _ = _packed(device, torch.bfloat16)

    kbar = block_representatives_triton(k, cfg, layout=lay)
    idx = gate_select(q, kbar, cfg, layout=lay)
    assert idx.shape == (1, 2, lay.total_tokens, cfg.top_k)

    seq_of_block = lay.seq_of_block
    for t in range(lay.total_tokens):
        seq = int((cu[1:] <= t).sum())
        chosen = idx[0, :, t]
        real = chosen[chosen >= 0]
        assert bool((seq_of_block[real.long()] == seq).all()), (
            f"token {t} (sequence {seq}) selected a block from another sequence"
        )


def test_gate_matches_running_each_sequence_alone(device):
    """Selections must be identical, not merely similar -- routing is a ranking."""
    cfg = MoBAConfig(block_size=4, top_k=3, dtype=torch.bfloat16)
    cu = torch.tensor(CU, dtype=torch.int32, device=device)
    lay = SeqLayout.from_cu_seqlens(cu, cfg.block_size, tile=32)
    q, k, _ = _packed(device, torch.bfloat16)

    packed = gate_select(q, block_representatives_triton(k, cfg, layout=lay),
                         cfg, layout=lay)

    lo, off = 0, 0
    for n in LENS:
        piece = gate_select(
            q[:, :, lo:lo + n],
            block_representatives_triton(k[:, :, lo:lo + n], cfg), cfg)
        # The packed gate emits global ids; the per-sequence one emits local.
        shifted = torch.where(piece >= 0, piece + off, piece)
        assert torch.equal(packed[:, :, lo:lo + n], shifted)
        lo += n
        off += (n + cfg.block_size - 1) // cfg.block_size


def test_a_tile_spanning_a_boundary_is_never_launched(device):
    """`tile_map` cuts tiles per sequence, so `bos` is a scalar in the kernel.

    Uses a tile size larger than several sequences, which is the arrangement
    that would straddle if the map were not there.
    """
    cfg = MoBAConfig(block_size=4, top_k=2, dtype=torch.bfloat16)
    cu = torch.tensor(CU, dtype=torch.int32, device=device)
    lay = SeqLayout.from_cu_seqlens(cu, cfg.block_size, tile=32)
    assert lay.tile_map.shape[0] == sum(
        (n + 31) // 32 for n in LENS), "one tile per sequence chunk"
    q, k, _ = _packed(device, torch.bfloat16)
    idx = gate_select(q, block_representatives_triton(k, cfg, layout=lay),
                      cfg, layout=lay)
    assert bool((idx < lay.total_blocks).all())
```

- [ ] **Step 2: Run to verify they fail**

Expected: `TypeError: gate_select() got an unexpected keyword argument 'layout'`.

- [ ] **Step 3: Give the gate kernel a sequence**

Add `TileMap, CuSeqlens, CuBlocks` pointers and `VARLEN: tl.constexpr`. Replace
the tile's row computation:

```python
    if VARLEN:
        # One tile belongs to exactly one sequence (`SeqLayout.tile_map`), so
        # every value here is a *scalar*. That is what keeps the streaming
        # top-k below unchanged: a tile spanning two sequences would make the
        # loop bound and the candidate range per-lane, and a Triton program has
        # one loop bound for all its lanes.
        seq = tl.load(TileMap + pid_m * 2)
        off = tl.load(TileMap + pid_m * 2 + 1)
        bos = tl.load(CuSeqlens + seq)
        eos = tl.load(CuSeqlens + seq + 1)
        boc = tl.load(CuBlocks + seq)            # this sequence's first block
        n_seq_blocks = tl.load(CuBlocks + seq + 1) - boc
        local = off + tl.arange(0, BLOCK_M)      # position within the sequence
        offs_m = bos + local                     # packed token index
        live = offs_m < eos
    else:
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        local = offs_m
        live = offs_m < S
        boc = 0
        n_seq_blocks = N_BLOCKS

    own_block = local // BLOCK_SIZE              # sequence-local (D-VL-6)
```

`offs_m` addresses `Q` and `BlockIdx` (packed token index); `local` drives
`own_block` and the horizon. In padded mode they are the same value, so nothing
moves.

- [ ] **Step 4: Offset the candidate loads and the emitted ids**

Two edits inside the streaming loop, and one at the emit. The loop bound uses
`n_seq_blocks` in place of `N_BLOCKS`:

```python
        in_range = cols < n_seq_blocks
        kbar = tl.load(
            Kbar + kv_bh * N_BLOCKS * D + (boc + cols)[:, None] * D
            + offs_d[None, :],
            mask=in_range[:, None], other=0.0,
        )
```

and the horizon:

```python
    horizon = tl.max(tl.where(live, own_block, 0))
    n_chunks = tl.cdiv(tl.minimum(horizon, n_seq_blocks), BLOCK_N)
```

The whole merge — tie-break, eviction, +inf retirement, seeding — is
**unchanged**: it operates on sequence-local `cols` throughout. At the emit,
add the offset once:

```python
        tl.store(BlockIdx + pid_bh * S * TOP_K + offs_m * TOP_K + j,
                 tl.where(valid, boc + emitted, -1), mask=live)
```

Note `N_BLOCKS` remains the *stride* of `Kbar` (total blocks) while
`n_seq_blocks` is the *range*. Conflating them is the likely bug here.

- [ ] **Step 5: Update `gate_select`**

Add `layout: SeqLayout | None = None`. When given, require `q.shape[0] == 1`,
set `n = layout.total_blocks`, grid `(layout.tile_map.shape[0], 1 * h)`, and
pass the three arrays with `VARLEN=True`. The `block_m` used to build the layout
must equal the one launched with — assert it rather than assume, since a
mismatch produces a silently wrong map:

```python
    if layout is not None and layout.tile != block_m:
        raise ValueError(
            f"layout was built for tile {layout.tile} but the gate launches "
            f"{block_m}; the tile map would not match the grid"
        )
```

- [ ] **Step 6: Run**

Run: `uv run pytest tests/test_gate.py tests/test_varlen.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/moba/kernels/gate.py tests/test_gate.py tests/test_varlen.py
git commit -m "feat: gate ranks within a sequence and emits global block ids"
```

---

## Task 4: The index, and confirming it needs almost nothing

**Files:**
- Modify: `src/moba/kernels/index_fused.py`, `src/moba/kernels/index.py`
- Test: `tests/test_varlen.py`

**Interfaces:**
- Produces: `build_csc_fused(q, k, cfg, layout=None)`; `CSCIndex` gains
  `layout: SeqLayout | None`; a `csc_to_mask` that returns sequence-local
  columns when a layout is present.

- [ ] **Step 1: Write the failing test**

```python
def test_index_needs_no_batch_axis_under_packing(device):
    """Global block ids mean the CSC arrays are already the right shape.

    A column is keyed by (kv_head, global_block). With B pinned to 1 the
    existing `[B, H_kv, N]` arrays hold exactly that, so the transpose code is
    untouched -- the block ids it is handed simply span the whole batch.
    """
    from moba.kernels.index_fused import build_csc_fused

    cfg = MoBAConfig(block_size=4, top_k=3, dtype=torch.bfloat16)
    cu = torch.tensor(CU, dtype=torch.int32, device=device)
    lay = SeqLayout.from_cu_seqlens(cu, cfg.block_size, tile=32)
    q, k, _ = _packed(device, torch.bfloat16)

    index = build_csc_fused(q, k, cfg, layout=lay)
    assert index.col_nnz.shape == (1, 2, lay.total_blocks)
    assert index.num_blocks == lay.total_blocks
    assert int(index.col_nnz.sum()) == int(
        (gate_select(q, block_representatives_triton(k, cfg, layout=lay),
                     cfg, layout=lay) >= 0).sum())


def test_every_column_holds_only_its_own_sequence_s_queries(device):
    """Blocks are per sequence, so a column's queries must be too."""
    from moba.kernels.index_fused import build_csc_fused

    cfg = MoBAConfig(block_size=4, top_k=3, dtype=torch.bfloat16)
    cu = torch.tensor(CU, dtype=torch.int32, device=device)
    lay = SeqLayout.from_cu_seqlens(cu, cfg.block_size, tile=32)
    q, k, _ = _packed(device, torch.bfloat16)
    index = build_csc_fused(q, k, cfg, layout=lay)

    for kv in range(2):
        for blk in range(lay.total_blocks):
            lo = int(index.col_offsets[0, kv, blk])
            cnt = int(index.col_nnz[0, kv, blk])
            rows = index.row_indices[0, kv, lo:lo + cnt]
            seq = int(lay.seq_of_block[blk])
            assert bool(((rows >= lay.cu_seqlens[seq])
                         & (rows < lay.cu_seqlens[seq + 1])).all()), (
                f"column {blk} (sequence {seq}) holds a foreign query"
            )
```

- [ ] **Step 2: Run to verify it fails**

Expected: `TypeError` on the `layout` keyword.

- [ ] **Step 3: Thread `layout` through, changing nothing else**

`build_csc_fused(q, k, cfg, layout=None)` passes the layout to
`block_representatives_triton` and `gate_select`, then calls
`build_csc_from_indices` with `num_blocks=layout.total_blocks`. The three
transpose kernels are **not modified**: block ids are already global and
`B = 1`.

Add `layout` to `CSCIndex` (default `None`) so consumers can reach the block
ranges without a second argument everywhere.

- [ ] **Step 4: Add a packed `csc_to_mask`**

When `index.layout` is set, return `[1, H_q, T, max_blocks_in_any_sequence]`
with **sequence-local** columns, matching the oracle's convention from Task 1
Step 8. Document the conversion — `global = cu_blocks[seq] + local` — in one
place.

- [ ] **Step 5: Run**

Run: `uv run pytest tests/test_index.py tests/test_index_fused.py tests/test_varlen.py -q`

- [ ] **Step 6: Commit**

```bash
git add src/moba/kernels/index.py src/moba/kernels/index_fused.py tests/test_varlen.py
git commit -m "feat: CSC index over a packed batch, unchanged transpose"
```

---

## Task 5: Forward attention over a packed batch

**Files:**
- Modify: `src/moba/kernels/moba_fwd.py` — `_moba_attn_fwd`, `_validate`,
  `attend_partials`, `moba_forward`
- Test: `tests/test_varlen.py`

**Interfaces:**
- Produces: `moba_forward(q, k, v, cfg, index=None, causal=True,
  partials_dtype=None, cu_seqlens=None)`.
- **Not produced, deliberately:** any change to `_moba_combine` (D-VL-8).

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("gate", [(4, 3), (8, 4)])
def test_forward_matches_running_each_sequence_alone(device, gate):
    """Bitwise per sequence: the packed run must reproduce the sliced runs.

    Not `allclose`. Slicing gives the same inputs in the same order, and the
    kernel's per-(query, slot) addressing is order-independent, so anything but
    equality is a boundary bug.
    """
    from moba.kernels.moba_fwd import moba_forward

    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    cu = torch.tensor(CU, dtype=torch.int32, device=device)
    q, k, v = _packed(device, torch.bfloat16)

    packed = moba_forward(q, k, v, cfg, cu_seqlens=cu)
    lo = 0
    for n in LENS:
        piece = moba_forward(q[:, :, lo:lo + n], k[:, :, lo:lo + n],
                             v[:, :, lo:lo + n], cfg)
        assert torch.equal(packed[:, :, lo:lo + n], piece), (
            f"sequence starting at {lo} differs"
        )
        lo += n


def test_packed_forward_matches_the_oracle(device):
    """Routing held fixed, so this measures arithmetic and not gate flips."""
    from moba.kernels.index_fused import build_csc_fused
    from moba.kernels.moba_fwd import moba_forward
    from moba.ref.naive import moba_attention_naive
    from moba.tolerance import (
        TOLERANCE_FACTOR, TOLERANCE_FLOOR, max_abs_error)
    from moba.kernels.index import csc_to_mask

    cfg = MoBAConfig(block_size=4, top_k=3, dtype=torch.bfloat16)
    cu = torch.tensor(CU, dtype=torch.int32, device=device)
    lay = SeqLayout.from_cu_seqlens(cu, cfg.block_size, tile=32)
    q64, k64, v64 = _packed(device, torch.float64)
    qb, kb, vb = (t.to(torch.bfloat16) for t in (q64, k64, v64))

    index = build_csc_fused(qb, kb, cfg, layout=lay)
    selected = csc_to_mask(index, lay.total_tokens)

    truth = moba_attention_naive(q64, k64, v64, cfg, selected=selected,
                                 cu_seqlens=cu)
    ref = moba_attention_naive(qb, kb, vb, cfg, selected=selected,
                               cu_seqlens=cu)
    ours = moba_forward(qb, kb, vb, cfg, index=index)

    cap = max(TOLERANCE_FACTOR * max_abs_error(ref, truth), TOLERANCE_FLOOR)
    assert max_abs_error(ours, truth) <= cap


def test_packed_input_with_batch_above_one_is_rejected(device):
    from moba.kernels.moba_fwd import moba_forward
    cfg = MoBAConfig(block_size=4, top_k=2, dtype=torch.bfloat16)
    cu = torch.tensor([0, 8, 16], dtype=torch.int32, device=device)
    q = torch.randn(2, 2, 16, 16, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="batch 1"):
        moba_forward(q, q, q, cfg, cu_seqlens=cu)
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Give the attention kernel a block range**

Add `BlockStart, BlockEnd` and `VARLEN: tl.constexpr`, and replace the key
range. This is the **only** change to `_moba_attn_fwd`:

```python
    if VARLEN:
        # A block's extent is looked up, not computed, because blocks are cut
        # per sequence and the last of each is short. Two scalar loads.
        lo_k = tl.load(BlockStart + blk)
        hi_k = tl.load(BlockEnd + blk)
    else:
        lo_k = blk * BLOCK_SIZE
        hi_k = S
    key_pos = lo_k + tl.arange(0, BLOCK_SIZE)
    key_ok = key_pos < hi_k
```

`kv_base`, the gathered Q load, `q_row`, the additive masks, `tl.dot`, the
online softmax and every store are **untouched**. In particular the causal mask
stays

```python
        qk += tl.where(key_pos[None, :] <= rows[:, None], 0.0, float("-inf"))
```

with a comment pointing at D-VL-7 and at
`test_no_selected_block_crosses_a_sequence_boundary`.

- [ ] **Step 4: Validation and plumbing**

`_validate` gains `cu_seqlens`; when set it requires `q.shape[0] == 1` and
`cu_seqlens[-1] == q.shape[2]`. `attend_partials` and `moba_forward` accept
`cu_seqlens` and build the layout once (or take `index.layout` when an index is
passed). Extend `_GROUP_CACHE`'s key with `bool(varlen)`.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: 379 existing pass unchanged, plus the new ones. **A pre-existing
failure means the padded path moved — stop.**

- [ ] **Step 6: Commit**

```bash
git add src/moba/kernels/moba_fwd.py tests/test_varlen.py
git commit -m "feat: forward attention over a packed batch"
```

---

## Task 6: Backward over a packed batch

Mirrors Task 5. Same two loads, same untouched everything-else.

**Files:**
- Modify: `src/moba/kernels/moba_bwd.py`, `src/moba/kernels/autograd.py`
- Test: `tests/test_varlen.py`

**Interfaces:**
- Produces: `moba_attention(q, k, v, cfg, index=None, causal=True,
  cu_seqlens=None)` differentiable end to end over a packed batch.
- **Not produced, deliberately:** changes to `_moba_dq_reduce`,
  `_moba_bwd_preprocess`, `compute_delta`, `reduce_dq` (D-VL-8).

- [ ] **Step 1: Write the failing test**

```python
def _grads(fn, q, k, v, g):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    (fn(q, k, v) * g).sum().backward()
    return q.grad, k.grad, v.grad


def test_gradients_match_running_each_sequence_alone(device):
    """Gradient must not leak across a boundary.

    dK and dV are reductions down a column. If a column ever held a query from
    another sequence, that query's gradient would be added to this document's
    keys -- finite, plausible, and wrong.
    """
    from moba.kernels.autograd import moba_attention

    cfg = MoBAConfig(block_size=4, top_k=3, dtype=torch.bfloat16)
    cu = torch.tensor(CU, dtype=torch.int32, device=device)
    q, k, v = _packed(device, torch.bfloat16)
    gen = torch.Generator(device=device).manual_seed(5)
    g = torch.randn_like(q) * 0.1

    pq, pk, pv = _grads(
        lambda a, x, y: moba_attention(a, x, y, cfg, cu_seqlens=cu), q, k, v, g)

    lo = 0
    for n in LENS:
        sq, sk, sv = _grads(
            lambda a, x, y: moba_attention(a, x, y, cfg),
            q[:, :, lo:lo + n], k[:, :, lo:lo + n], v[:, :, lo:lo + n],
            g[:, :, lo:lo + n])
        for name, a, b in [("dq", pq, sq), ("dk", pk, sk), ("dv", pv, sv)]:
            assert torch.allclose(a[:, :, lo:lo + n].float(), b.float(),
                                  rtol=2e-2, atol=1e-3), (
                f"{name} differs for the sequence starting at {lo}")
        lo += n
```

- [ ] **Step 2–4: Run, patch `_moba_bwd`, plumb**

Identical treatment to Task 5 Step 3: `BlockStart`/`BlockEnd` plus `VARLEN`,
replacing only

```python
    key_pos = blk * BLOCK_SIZE + sub * KEY_TILE + tl.arange(0, KEY_TILE)
    key_ok = key_pos < S
```

with the looked-up range, keeping the sub-tile offset:

```python
    lo_k = tl.load(BlockStart + blk) if VARLEN else blk * BLOCK_SIZE
    hi_k = tl.load(BlockEnd + blk) if VARLEN else S
    key_pos = lo_k + sub * KEY_TILE + tl.arange(0, KEY_TILE)
    key_ok = key_pos < hi_k
```

`moba_backward` takes the layout from `index.layout`. `autograd.py` plumbs
`cu_seqlens` into `forward` and stores the index as it already does.

- [ ] **Step 5: Run the full suite**

- [ ] **Step 6: Commit**

```bash
git add src/moba/kernels/moba_bwd.py src/moba/kernels/autograd.py tests/test_varlen.py
git commit -m "feat: backward over a packed batch"
```

---

## Task 7: Baseline, measurement, documentation

**Files:**
- Modify: `src/moba/backends/sdpa_dense.py`, `bench/suite.py`
- Create: `bench/varlen_sweep.py`, `docs/varlen-measurements.md`
- Modify: `CLAUDE.md` status line

- [ ] **Step 1: Give the dense baseline varlen**

`torch` supports it without kernel work, verified on this box: a
`torch.nested.nested_tensor(..., layout=torch.jagged)` passed to
`scaled_dot_product_attention` runs on the `FLASH_ATTENTION` backend, and
`MATH` *refuses* with "no viable backend" — proving it dispatches to the fused
varlen kernel rather than silently padding.

This matters for honesty: the current padded benchmark compares us against a
baseline that is also wasting work on padding. A packed comparison is the fair
one.

- [ ] **Step 2: Benchmark, with allocation excluded**

Reuse the lesson from `docs/gqa-measurements.md`: preallocate partials with
`empty_partials` and pass `partials=`, and take medians. Timing
`moba_forward` directly measures the WDDM allocator, not the kernel.

Measure, at `H_q=8, H_kv=2, block_size=128, top_k=8`:
- a realistic ragged batch (e.g. lengths sampled 256–2048) packed vs padded to
  max
- tokens saved, `part_acc` bytes saved
- grid size, packed vs padded — this should *rise*, which per
  `docs/gqa-measurements.md` is worth more on 20 SMs than the byte saving
- against nested-tensor dense SDPA on the same ragged batch

- [ ] **Step 3: Write `docs/varlen-measurements.md`**

Structured like `docs/gqa-measurements.md`. State the prediction **before** the
numbers: padding waste is `1 - mean(len)/max(len)`, so the gain is entirely a
function of how ragged the batch is and should be ~0% on a uniform batch. Say
so first, then measure.

- [ ] **Step 4: Update the status line, then commit**

```bash
git add src/moba/backends/sdpa_dense.py bench/varlen_sweep.py bench/suite.py docs/varlen-measurements.md CLAUDE.md
git commit -m "bench: packed batches against a packed dense baseline"
```

---

## Risks

**A block or a tile straddling a boundary does not crash.** It produces finite,
plausible numbers of the right shape and the model quietly attends across
documents. Every task therefore compares against *per-sequence slices* rather
than checking shapes, and Tasks 1 and 3 assert the boundary property directly.

**`N_BLOCKS` now means two things in the gate** — the stride of `Kbar` (total
blocks) and, formerly, the candidate range (now `n_seq_blocks`). Conflating them
is the most likely bug in Task 3 and would show as queries ranking against
another sequence's representatives.

**`_offsets_and_order` is O(N²) in a single program.** It does a
`static_range(N_BLOCKS)` of full-vector reductions, one program per
`(batch, kv_head)`. Packing raises `N_BLOCKS` from "blocks in one sequence" to
"blocks in the whole batch", so a batch of 32 sequences multiplies it by ~32.
This is a **pre-existing** scaling limit that packing makes reachable sooner.
Measure it in Task 7 before assuming it is fine; if it dominates, the fix is a
multi-block scan, which is its own piece of work and not in this plan.

**Memory.** Packing removes padding from `part_acc`, the largest buffer, so
packed should use *less* memory than padded at equal token count. If it does
not, the layout is wrong somewhere.
