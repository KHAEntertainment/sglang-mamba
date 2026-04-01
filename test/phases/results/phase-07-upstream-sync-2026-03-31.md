# Phase 07 Results — Re-apply Snapshot Features (Upstream Sync)

**Date:** 2026-03-31
**Branch:** `upstream-sync-2026-Q1`
**Base tag:** `phase-06b-pass`
**Result tag:** `phase-07-pass`
**Result:** PASS

---

## Objective

Re-port and validate snapshot-specific product features on the updated upstream substrate after the bulk merge of 1,387 upstream commits (Phase 06B).

---

## Commits Integrated

| Commit | Description |
|--------|-------------|
| `970298b9b` | phase-07: fix Dict import and device mismatch from upstream merge |

One commit on top of `phase-06b-pass`. The snapshot system survived the bulk merge largely intact; only two targeted fixes were required.

---

## Conflicts Resolved

### 1. Missing `Dict` typing import in scheduler.py

**File:** `python/sglang/srt/managers/scheduler.py:27`
**Symptom:** `NameError: name 'Dict' is not defined` at class definition time (line 2277, `get_init_info` return annotation).
**Root cause:** The upstream merge rewrote the typing import line, dropping `Dict` from `from typing import Any, Deque, List, Optional, Tuple, Union`. Our `get_init_info` method (added for snapshot system handshake) uses `Dict[str, Any]`.
**Fix:** Added `Dict` back to the import list.

### 2. CPU/CUDA device mismatch in gauntlet test

**File:** `test/registered/radix_cache/test_mamba_radix_cache_gauntlet.py:228-230`
**Symptom:** `RuntimeError: Expected all tensors to be on the same device, but got tensors is on cpu, different from other tensors on cuda:0` in `test_branching_seqlen_triggered`.
**Root cause:** The upstream merge changed the allocator (`allocator.py:163`) to use `torch.cat` for freeing pages, which requires device consistency. The test created KV tensors with `torch.zeros(N, dtype=torch.int64)` (defaults to CPU) while the allocator's `free_pages` tensor lives on CUDA.
**Fix:** Added `device=self.device` to the three KV tensor constructors in the test.

---

## Design Decisions

### 1. No re-porting needed — snapshot code survived merge

The Phase 07 plan anticipated needing to manually re-apply the entire `snapshot/` package, scheduler handlers, tokenizer manager queues, HTTP routes, and CLI flags. In practice, the Phase 06B bulk merge preserved all of this code. The decision was to **verify and fix** rather than **re-port**.

### 2. Preserve existing hook attachment points

The `init_snapshot_system()` call in `Scheduler.__init__` (line 406, after `init_cache_with_memory_pool`) remains correctly positioned relative to upstream's new initialization sequence. No changes needed to hook placement.

### 3. MambaPool extract/inject compatibility confirmed

Upstream did not change `MambaPool.State` or `SpeculativeState` dataclass fields. The `extract_state_from_pool` and `inject_state_to_pool` methods in `mamba_snapshot.py` remain compatible with the pool's tensor layout (`conv`: list of tensors per layer, `temporal`: single tensor with `[num_layers, ...]` shape).

### 4. SessionController coexistence

The `SessionController` (upstream addition) manages session lifecycle independently of our snapshot system. The snapshot handlers in `scheduler.py` do not need to interact with `SessionController` directly — they operate on `Req` objects and `MambaPool` indices, which remain stable across the merge.

---

## Test Results

### Unit Tests: 30 passed, 1 skipped

```text
test/sglang/snapshot/test_mamba_snapshot.py                      11 passed, 1 skipped
test/registered/unit/mem_cache/test_mamba_unittest.py             4 passed
test/registered/radix_cache/test_mamba_radix_cache_comprehensive.py  9 passed
test/registered/radix_cache/test_mamba_radix_cache_gauntlet.py    6 passed
```

**Skipped:** `TestSnapshotStateExtraction::test_extract_and_inject_roundtrip` — requires GPU tensors in a specific MambaPool configuration (deferred to Phase 08 live server validation).

### Breakdown by category

| Category | Tests | Result |
|----------|-------|--------|
| Snapshot metadata (create, serialize, roundtrip) | 3 | PASS |
| Snapshot manager (save, load, list, delete, branch) | 7 | PASS |
| Snapshot state extraction | 1 | SKIP |
| MambaPool (alloc, free, state) | 1 | PASS |
| HybridLinearKVPool | 1 | PASS |
| Radix cache insert with prev_prefix_len | 1 | PASS |
| MambaRadixCache basic operations | 1 | PASS |
| MambaRadixCache comprehensive (COW, eviction, LRU, tombstone) | 9 | PASS |
| MambaRadixCache gauntlet (interleave, branching, accounting) | 6 | PASS |

### Import Verification: 14/14 modules clean

All snapshot-adjacent modules import without errors:

```text
sglang.srt.snapshot                    OK
sglang.srt.snapshot.mamba_snapshot     OK
sglang.srt.snapshot.tier_manager       OK
sglang.srt.snapshot.conversation_tracker OK
sglang.srt.snapshot.snapshot_policy    OK
sglang.srt.snapshot.snapshot_hooks     OK
sglang.srt.snapshot.mamba_host_pool    OK
sglang.srt.managers.scheduler          OK
sglang.srt.managers.tokenizer_manager  OK
sglang.srt.managers.io_struct          OK
sglang.srt.entrypoints.http_server     OK
sglang.srt.server_args                 OK
sglang.srt.mem_cache.mamba_radix_cache OK
sglang.srt.mem_cache.memory_pool       OK
```

### pip install: SUCCESS

`pip install -e "python/"` completed without errors.

---

## Snapshot Infrastructure Verification

All components confirmed present and functional:

| Component | Files/Items | Status |
|-----------|-------------|--------|
| snapshot/ package | 7 files (mamba_snapshot, tier_manager, conversation_tracker, snapshot_policy, snapshot_hooks, mamba_host_pool, `__init__`) | PRESENT |
| I/O structs | 10 (Save/Restore/List/GetInfo/Delete Request+Response) | PRESENT |
| GenerateReqInput.conversation_id | 1 field | PRESENT |
| Scheduler handlers | 6 (save, restore, list, get_info, delete, hook_trigger) | PRESENT |
| Tokenizer manager | 5 async forwarders + 5 result queues | PRESENT |
| HTTP REST endpoints | 5 (POST /save_snapshot, /list_snapshots, /get_snapshot_info, /restore_snapshot, /delete_snapshot) | PRESENT |
| CLI flags | 8 (--enable-snapshot-persistence, --snapshot-dir, --snapshot-retention-count, --snapshot-trigger-policy, --snapshot-every-n-turns, --snapshot-min-interval-seconds, --snapshot-keep-named-branches, --snapshot-auto-restore) | PRESENT |

---

## Issues Discovered

### Fixed in this phase

1. **Missing `Dict` import** — Broke scheduler module load entirely. Fixed by adding `Dict` to typing imports.
2. **Device mismatch in test** — `test_branching_seqlen_triggered` created CPU tensors passed to a CUDA allocator. Fixed by specifying `device=self.device`.

### Pre-existing (not introduced by this phase)

- **FastAPIDeprecationWarning** on `ORJSONResponse` import in `http_server.py` — cosmetic, does not affect functionality.
- **SwigPyPacked/SwigPyObject DeprecationWarning** — from system-level SWIG bindings, unrelated to snapshot code.

---

## Files Modified

| File | Change |
|------|--------|
| `python/sglang/srt/managers/scheduler.py` | Added `Dict` to typing imports (line 27) |
| `test/registered/radix_cache/test_mamba_radix_cache_gauntlet.py` | Added `device=self.device` to 3 tensor constructors (lines 228-230) |

---

## Next Steps

- **Phase 08:** Full validation and stress test — run phases 0-8 with live server on granite-4.0-h-tiny to confirm snapshot save/restore cycle and stateful inference work end-to-end.
