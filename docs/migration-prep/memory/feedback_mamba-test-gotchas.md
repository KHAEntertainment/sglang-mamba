---
name: Mamba test implementation gotchas
description: Surprising pitfalls found while writing tests for MambaRadixCache, Mamba2Metadata, and MambaPool
type: feedback
---

When writing tests for the sglang-mamba fork, watch for these non-obvious issues discovered during Phase 0–5 implementation:

**MambaRadixCache tombstone tests:**
- Tombstones (KV kept, Mamba state evicted) are only created on INTERNAL nodes (nodes with children). Evicting a LEAF node removes it entirely — no tombstone, no `mamba_branching_seqlen`.
- To create a tombstone: insert base sequence, then insert an extension of it (making base an internal node), then evict Mamba from the base.
- `mamba_branching_seqlen` is computed as `(matched_tokens // chunk_size) * chunk_size`. Chunk size is `max(FLA_CHUNK_SIZE, page_size) = 64`. Sequences must be ≥ 64 tokens or the result rounds to zero and the field stays `None`.

**Why:** Original test used 8-token sequences and evicted a leaf — both wrong. Caused `AssertionError: mamba_branching_seqlen should be set` that looked like an implementation bug.

**Mamba2Metadata `query_start_loc`:**
- For `prepare_mixed` with N sequences of length L each, use `torch.tensor([0, L, 2L, ..., N*L])` (N+1 elements), NOT `torch.arange(N+1)`.
- `repeat_interleave` uses the differences between consecutive entries to determine repetition counts. Using `arange` gives diffs of 1 (one token per seq), not L.
- Same fix applies to `test_has_initial_states_flag`: 5-token sequences → `[0, 5, 10, 15, 20]`.

**Why:** The template in the phase document had `arange(N+1)` which would cause a `repeat_interleave` size mismatch crash.

**MambaPool dtype attribute:**
- Wrong: `cache_params.dtype.ssm_state_dtype` (does not exist)
- Right: `cache_params.dtype.temporal`

**MambaRadixCache match result field:**
- Wrong: `match_result.value`
- Right: `match_result.device_indices`

**MambaPool req pool exhaustion:**
- `HybridReqToTokenPool` has `max_num_reqs` slots. In tests, call `req_to_token_pool.free(req)` after inserting into the radix cache — the tree node owns the mamba slot, but the req pool slot can be reused.

**How to apply:** Reference these whenever writing new tests for MambaRadixCache or Mamba2Metadata to avoid false-negative failures that look like implementation bugs.
