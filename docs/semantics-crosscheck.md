# MoBA semantics cross-check

**Date:** 2026-08-15
**Our spec:** `docs/superpowers/specs/2026-08-15-moba-foundations-design.md`,
section "MoBA semantics" (nine numbered rules).

Purpose: for every rule the MoBA paper leaves underdetermined, record what each
reference implementation actually does. When a future kernel disagrees with a
reference, this table is what turns "something is wrong" into "we knowingly
differ on rule 5".

## Availability at time of writing

| Reference | Read? | Runs? | Note |
|---|---|---|---|
| `MoonshotAI/MoBA` (naive) | yes | **yes** | `moba_naive.py` imports only torch + math |
| `MoonshotAI/MoBA` (efficient) | no | no | needs `flash-attn==2.6.3` |
| `ReyJerry/Triton-MoBA` | partly | no | imports `flash_attn` at module scope |
| `mit-han-lab/flash-moba` | pending (Task 10) | no | `setup.py` targets archs 80;90;100;120; this GPU is sm_86 |

## Rule-by-rule

Source for Moonshot is `references/MoBA/moba/moba_naive.py`.

| # | Rule | Ours | Moonshot | Status |
|---|---|---|---|---|
| 1 | Blocking | `ceil(S / block_size)`, short final block | same (`moba_naive.py:45`) | agree |
| 2 | Block representative | mean of block's keys | same (`:49`) | agree |
| 3 | Gate score | unscaled `q · k̄` | same (`:55`), but computed in **fp32** | agree on math |
| 4 | Causal candidacy | strictly past blocks only | same (`:60`) | agree |
| 5 | Current block | always selected | always selected, via `+inf` (`:61`) | agree |
| 6 | **top-k budget** | **was: excludes own block** | **includes own block** (`:63`) | **DIVERGED — resolved** |
| 7 | Tie-breaking | stable sort, lowest index | `torch.topk`, unspecified (`:63-73`) | we are stricter |
| 8 | Partial-block pooling | whole block | same (`:48-49`) | agree |
| 9 | Attention | scale `1/sqrt(D)`, causal | same (`:87-92`), forced **fp32** | agree on math |

## Divergence 6: does `top_k` count the query's own block?

**What we found.** Moonshot assigns the query's own block a gate score of
`+inf` (`moba_naive.py:61`) and then runs `torch.topk(gate, k=moba_topk)` over
everything. The own block therefore wins a slot rather than being exempt from
the budget. Our original rule 5 made it free, so we attended to `top_k + 1`
blocks where they attend to `top_k`.

**Evidence.** With `their moba_topk == our top_k + 1`, outputs matched to
**0.000e+00** — bitwise, in fp32, at `top_k` 2, 3 and 4. A bitwise-zero match
across several values rules out coincidence and simultaneously proves no other
rule diverges, since any second difference would break exact equality.

**Resolution: adopt their convention as our default.** Not because they are
authoritative on the algorithm, but for a concrete practical reason: this
project targets FlashMoBA's published numbers. If our `top_k=2` attends 3
blocks and theirs attends 2, every benchmark comparison would be made at a
different sparsity, and any speedup we reported would be meaningless.

Both readings remain reachable via `MoBAConfig.top_k_includes_current`
(default `True`). After the change our oracle matches Moonshot **bitwise at
every `top_k` tested**.

## Divergence 7: tie-breaking

Moonshot uses `torch.topk`, whose tie order is unspecified, then intersects the
result with a threshold mask (`gate >= kth_value`) — their comment at `:68`
acknowledges the corner case. We use a stable descending sort, so equal scores
resolve toward the lower block index deterministically.

**Resolution: keep ours.** This is strictly stronger, and it costs nothing in
an oracle. Non-determinism in a GPU kernel is nearly indistinguishable from a
race condition when debugging, so a fixed rule is worth more than matching an
unspecified one. Ties require exactly equal gate scores and are vanishingly
rare on real data; they show up mainly in synthetic tests.

## Observation: Moonshot forces fp32 internally

`moba_naive.py:53-54` computes the gate in fp32 with the comment *"use fp32 to
avoid precision issue in bf16"*, and `:84-86` casts q, k and v to fp32 for the
attention itself.

Two consequences worth recording:

1. **They independently hit the routing-instability problem we found.** Their
   fp32 gate is the same mitigation our tolerance policy arrived at from the
   other direction — see the spec's revised D5. Independent discovery is good
   evidence the effect is real and not an artifact of our implementation.
2. **Passing them fp64 silently downcasts to fp32.** That is why their output
   sits ~6.7e-7 from dense causal attention in an fp64 comparison, rather than
   at fp64 noise (~1e-15). Not a bug, but it means they cannot serve as a
   high-precision oracle — which is one more reason D1's spec-first oracle was
   the right call.

## Still open

- **`ReyJerry/Triton-MoBA`** — gate not yet compared; module needs flash-attn
  to import. Revisit on a machine where that builds, or by reading rather than
  running.
- **`mit-han-lab/flash-moba`** — Task 10 reads `csrc/` for the digest and will
  record its gate semantics here at the same time.
