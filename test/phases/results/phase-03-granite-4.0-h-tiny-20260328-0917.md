# Phase 3 — MambaRadixCache Component Tests
**Model**: granite-4.0-h-tiny  *(model not loaded; unit tests only)*
**Date**: 2026-03-28
**Result**: PASS

## Regression Guard (10 tests)
test_mamba_radix_cache_1: PASS
test_mamba_radix_cache_comprehensive (9): 9/9 PASS

Two tests in comprehensive were fixed this session:
- `test_full_cache_eviction`: Added explicit `cache.evict(mamba_num=1)` before `_make_dummy_req()`
  (insert() does not auto-evict; mamba pool assertion fired otherwise). Fixed `.value` → `.device_indices`
  and replaced `assertIsNone/assertIsNotNone` with `len()` checks on device_indices.
- `test_mamba_branching_seqlen`: Replaced "insert extension first" approach with A(65)/B(75)/C(85)
  sequential insert → evict 2 → match B. Uses fake zero KV tensors to stay within 128-slot KV pool.
  Internal-node eviction does not free KV, so fake tensors are safe.

## Gauntlet Tests (test_mamba_radix_cache_gauntlet.py)
| Test | Result | sanity_check |
|------|--------|--------------|
| test_interleaved_insert_evict_match | PASS | PASS |
| test_tombstone_does_not_match_mamba | PASS | PASS |
| test_branching_seqlen_triggered | PASS | PASS |
| test_cow_state_independence | PASS | PASS |
| test_inc_dec_lock_ref_symmetry | PASS | PASS |
| test_full_evictable_and_protected_size_accounting | PASS | PASS |

## sanity_check violations in tearDown
NONE

## Failures & Tracebacks
None

## Notes on tombstone / device_indices behavior (discovered during implementation)
- `_match_post_processor` line 1023: `value = value[:best_value_len]` — device_indices is
  truncated to nodes with live mamba state. A path ending on a tombstone gives empty device_indices.
- Matching through a tombstone to a live mamba node returns full KV for the entire path.
- `mamba_branching_seqlen` = `(total_matched_tokens // chunk_size) * chunk_size` when any tombstone
  is encountered on the path (len(value) > best_value_len after truncation check).
