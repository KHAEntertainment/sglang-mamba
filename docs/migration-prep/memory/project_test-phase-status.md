---
name: sglang-mamba test phase status
description: Current pass/fail status of all 9 test phases; hardware blocker and new VM setup details
type: project
---

## Phase completion status (as of 2026-03-28)

| Phase | Description | Result | Notes |
|-------|-------------|--------|-------|
| Phase 0 | Environment verification | **PASS** | |
| Phase 1 | Stateless inference baseline | **INCOMPLETE** | Needs sm75+ GPU |
| Phase 2 | MambaPool unit tests | **PASS** (5/5) | `test_mamba_pool_extended.py` |
| Phase 3 | MambaRadixCache gauntlet | **PASS** (16/16) | `test_mamba_radix_cache_comprehensive.py` (9 tests, 2 fixed) + `test_mamba_radix_cache_gauntlet.py` (6 new) |
| Phase 4 | Live server integration (no_buffer) | **INCOMPLETE** | Needs sm75+ GPU |
| Phase 5 | Mamba2Metadata integrity | **PASS** (5/5) | `test_mamba_metadata.py` |
| Phase 6 | extra_buffer strategy | **INCOMPLETE** | Needs sm75+ GPU |
| Phase 7 | Snapshot system e2e | **INCOMPLETE** | Needs sm75+ GPU; validates gap fixes from PRs #4 #6 |
| Phase 8 | Gauntlet stress tests | **INCOMPLETE** | Needs sm75+ GPU |

## Hardware blocker

V100 = sm70. FLA Mamba2 CUDA kernels + FlashInfer both require sm75+. All hybrid Mamba inference is blocked on V100. Unit-test phases (0/2/3/5) run fine. Server phases (1/4/6/7/8) require sm75+ (A100, T4, A10G, RTX 30xx+).

**New VM setup guide**: `test/phases/NEW_VM_SETUP.md` — clone, install, model via hf-mount, config.sh update, resume order.

**Why:** V100 hardware wall hit mid-run; VM may need to be replaced.
**How to apply:** Check NEW_VM_SETUP.md first; run server phases in order 1→4→7→6→8; stop at first failure.

## Phase 3 test fixes (2026-03-28)

`test_mamba_radix_cache_comprehensive.py` two tests fixed:
- `test_full_cache_eviction`: `insert()` does not auto-evict; added explicit `cache.evict(mamba_num=1)` before `_make_dummy_req()`. Fixed `.value` → `.device_indices` (correct MatchResult field). Fixed `assertIsNone/IsNotNone` → `len()` checks.
- `test_mamba_branching_seqlen`: A(65)/B(75)/C(85) sequential inserts → evict 2 LRU (both internal → tombstone). Uses fake zero KV tensors (safe: internal-node eviction never frees KV). Asserts `mamba_branching_seqlen == 64`.

Key insight: `_match_post_processor` truncates `device_indices` to `value[:best_value_len]` — path ending on tombstone gives empty device_indices; path through tombstone to live mamba node returns full KV for all matched nodes.
