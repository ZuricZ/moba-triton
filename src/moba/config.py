"""Configuration for MoBA attention.

The three semantics flags below are the reason this module exists as more than
a tuple of ints. MoBA's paper does not fully pin down what happens at the
edges: how ties in the gate ranking break, whether a query's own block is
mandatory, and how a partially-masked block is pooled. Different published
implementations answer these differently.

Encoding each ambiguous choice as an explicit field turns "why does our kernel
disagree with FlashMoBA?" from an archaeology expedition into flipping one
field and re-running. See the design spec, section "MoBA semantics".
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MoBAConfig:
    """Parameters defining one MoBA attention configuration.

    Frozen because configs identify benchmark rows: mutating one after results
    were recorded against it would silently invalidate them.

    Attributes:
        block_size: Number of key positions per block.
        top_k: Number of past blocks selected per query, excluding the query's
            own block.
        dtype: Compute dtype for the attention itself. The gate and the oracle
            may run at higher precision.
        always_select_current: If True, the block containing the query is
            always attended (causally masked within) and does not consume a
            top_k slot.
        tie_break_low_index: If True, equal gate scores resolve toward the
            lower block index. Determinism matters here: an unstable tie-break
            makes a kernel non-reproducible run to run, which is
            indistinguishable from a race condition when debugging.
        pool_masked_only: If True, a block's representative key averages only
            causally-visible positions. If False, it averages the whole block.
    """

    block_size: int
    top_k: int
    dtype: torch.dtype = torch.bfloat16
    always_select_current: bool = True
    tie_break_low_index: bool = True
    pool_masked_only: bool = False

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")

    def num_blocks(self, seq_len: int) -> int:
        """Returns the number of blocks covering `seq_len` positions.

        Rounds up: a sequence that does not divide evenly ends in a short
        block, which is a real case the kernel must handle rather than an
        error to reject.

        Args:
            seq_len: Sequence length in positions.

        Returns:
            Number of blocks, at least 1.
        """
        return math.ceil(seq_len / self.block_size)
