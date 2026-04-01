# Phase 3 Audit Archive

> This page consolidates Phase 3 audit materials originally in `phase3/` in the repo.

## Performance Analysis Summary

**Date:** 2026-02-16
**File Analyzed:** `python/sglang/srt/mem_cache/mamba_radix_cache.py` (1,233 lines)

### Key Findings

| Optimization | Impact | Complexity | Priority |
|-------------|---------|-----------|----------|
| Remove setattr/getattr overhead | 20-30% LRU ops | Medium | CRITICAL |
| Optimize tensor cloning | 10-15% memory | High | HIGH |
| Cache tree depth for locks | 5-10% lock ops | Medium | MEDIUM |
| Optimize LRU traversal | 10-20% eviction | Medium | HIGH |
| Batch lock operations | 5-10% overall | High | MEDIUM |

### Architecture Overview

#### TreeNode
```python
class TreeNode:
    children: defaultdict[TreeNode]     # Child nodes
    parent: TreeNode                     # Parent reference
    value: torch.Tensor                  # KV cache indices
    mamba_value: torch.Tensor           # Mamba state indices
    full_lock_ref: int                   # Lock count for full cache
    mamba_lock_ref: int                  # Lock count for Mamba cache
    prev/next: TreeNode                  # Full LRU double-linked list
    mamba_prev/mamba_next: TreeNode     # Mamba LRU double-linked list
```

**Analysis:**
- ✅ Good: Dual LRU lists avoid conflicts between full and Mamba eviction
- ✅ Good: Separate lock references maintain clear invariants
- ⚠️ Issue: 8 pointer fields per node = 64 bytes overhead
- ✅ Good: `defaultdict` usage is appropriate for sparse trees

#### LRUList
```python
class LRUList:
    head: TreeNode          # Dummy head (MRU side)
    tail: TreeNode          # Dummy tail (LRU side)
    cache: dict[int, TreeNode]  # ID -> node mapping
```

**Analysis:**
- ✅ Good: Dummy head/tail simplifies edge cases
- ✅ Good: Maintains `cache` dict for O(1) membership checks
- ❌ CRITICAL ISSUE: Uses `setattr`/`getattr` for dynamic attributes

## Gap Analysis Summary (as of 2026-03-28)

### Complete (built and working)

| Feature | Code Location | Notes |
|---------|--------------|-------|
| Snapshot save/list/get/delete | `python/sglang/srt/snapshot/` + scheduler + HTTP endpoints | 7 source files, fully wired |
| Snapshot restore (in-place + create_new_request) | `scheduler.py:1294+` | PR #4, merged 2026-03-28 |
| Startup snapshot warm restore (Gap 3) | `tier_manager.py:restore_latest_snapshots_to_warm_tier()` | PR #6, merged to `main` |
| MambaRadixCache | `python/sglang/srt/mem_cache/mamba_radix_cache.py` (1239 lines) | Dual LRU, tombstones, COW |
| Agent Framework | `python/sglang/srt/agents/` (10 files) | 4 built-in tools, REST + WebSocket |
| 3-Tier Memory | `python/sglang/srt/snapshot/tier_manager.py` (468 lines) | VRAM/RAM/Disk with LRU |
| SnapshotManager API | `python/sglang/snapshot.py` (303 lines) | Public API exported from `__init__.py` |
| ConversationTracker | `python/sglang/srt/snapshot/conversation_tracker.py` | Tier state tracking |
| Snapshot hooks + policy | `snapshot_hooks.py`, `snapshot_policy.py` | Branching, retention, trigger policies |
| Fork-specific unit tests | `test/sglang/snapshot/` (4 files), `test/sglang/agents/` (3 files) | 46 pass, 1 skip (snapshot); 37 pass (agents) |
| Test phase infrastructure | `test/phases/` (9 plans, config.sh, codemap, results/) | Phases 0/2/3/5 PASS |

### In Progress / Partially Done

| Item | Status | Notes |
|------|--------|-------|
| Server-phase testing (1/4/6/7/8) | Blocked | All need sm75+ GPU; V100 is sm70 |
| Phase 3.4 Final Audit (KHA-6) | Pending | Scope needs updating to reflect test phases |
| Model documentation (KHA-12, KHA-13) | Not started | References `phase3/docs/pending_updates.json` |

### Planned but Not Built

| Item | Plan Source | Status |
|------|-------------|--------|
| Phase test files (1,4,6,7,8) | `test/phases/` plans | Plans exist, no `.py` test files created |
| Automatic snapshot triggers | `MAMBA_SNAPSHOT_RESTORATION_PLAN.md` Phase 3 Task 3 | Deferred |
| Bulk snapshot operations | `MAMBA_SNAPSHOT_RESTORATION_PLAN.md` Phase 3 Task 5 | Not started |
| Performance optimizations | KHA-7 through KHA-11 | Identified via static analysis, not profiled on running system |

## Related Documentation

- [Stateful Mamba Guide](./Stateful-Mamba-Guide.md)
- [Upstream Sync](./Upstream-Sync-Q1-2026.md)
