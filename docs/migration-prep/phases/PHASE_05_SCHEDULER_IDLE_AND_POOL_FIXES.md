# PHASE 05 — Scheduler Idle and Pool Fixes

## Worktree Safety Check (MANDATORY)

Before making any changes, run:
```bash
if [[ "$(git branch --show-current)" != "upstream-sync-2026-Q1" ]]; then
    echo "ERROR: Not on upstream-sync-2026-Q1 branch. Aborting."
    exit 1
fi
if [[ "$(git rev-parse --show-toplevel)" != *"worktrees/upstream-sync-2026-Q1" ]]; then
    echo "ERROR: Not inside the designated worktree. Aborting."
    exit 1
fi
echo "Worktree safety check passed."
```
If this fails, **stop immediately** and switch to the correct worktree.

## Objective

Merge scheduler idle-detection and Mamba slot-release fixes from upstream that touch the same lifecycle paths as our background snapshot/tier behavior.

## Upstream Commits to Integrate

| Commit | Risk | Topic |
|--------|------|-------|
| `b1246c50f` | MEDIUM | Fixes streaming-session KV cache leaks in `memory_pool.py` |
| `e2fccb2ee` | MEDIUM | Reverts Mamba slot release behavior to fix flaky tests |
| `8541b1118` | MEDIUM | Passes max Mamba cache size through disaggregation decode path |
| `4d3976b6c` | MEDIUM | Changes `is_fully_idle()` to account for async HiCache ops |
| `5270a0648` | MEDIUM | Fixes disaggregation false positives in idle detection |
| `079a1fd35` | MEDIUM | Fixes write-through events when idle |
| `50953aea8` | MEDIUM | Unifies idle checks into `is_fully_idle()` |
| `26f709e97` | LOW-MEDIUM | Makes prefill-delayer work with multiple mem-pool types; nearby pool abstraction churn |

## Files Touched

- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/mem_cache/memory_pool.py`
- Disaggregation decode path (if `8541b1118` touches files outside the core 6)

## Decision Points

### 1. Adopt all upstream fixes directly
**Decision:** Accept all of these commits. They improve correctness and fix known flaky behaviors.

### 2. Coordinate `TierManager` background tasks with `is_fully_idle()`
**Concern:** Upstream's unified `is_fully_idle()` now includes async HiCache operations. Our `TierManager` runs a background cleanup thread that promotes/demotes Mamba states and may interact with scheduler idle assumptions.

**Decision:** After merge, verify that `TierManager`'s background operations do not prevent the scheduler from reaching idle. If they do, gate the tier background thread so it pauses when the scheduler has active batches.

### 3. Mamba slot release revert (`e2fccb2ee`)
**Concern:** This commit changes when Mamba slots are released. Our restore/reinsert lifecycle depends on slot release timing to know when restored state remains valid.

**Decision:** Adopt upstream's revert, but run Phase 7 and Phase 8 immediately to confirm restore semantics still hold.

## Execution Steps

1. Cherry-pick / merge the commits above.
2. Resolve any trivial conflicts in `scheduler.py` or `memory_pool.py`.
3. Review the new `is_fully_idle()` implementation and note any new conditions.
4. Check `TierManager` background thread logic — ensure it does not hold locks or keep structures "busy" that the scheduler uses for idle detection.
5. Run Phases 2, 3, 4, 5, 6, 7, and 8.

## Validation Criteria

- Phases 2, 3, 4, 5, 6, 7, and 8 all **PASS**.
- No `mamba_lock_ref` violations or pool exhaustion errors in logs.

## Rollback Plan

- Reset worktree to `phase-04-pass` tag.

## Estimated Complexity

**MEDIUM** — 4 to 8 hours. Mostly correctness verification rather than structural redesign.

## Dependencies

- `PHASE_04_MAMBA_CACHE_ARCHITECTURE_RECONCILE` complete and tagged `phase-04-pass`.

## Team Structure

**Solo agent** (or 2-agent if Research Agent is needed to verify disaggregation decode path details).

## bd Workflow

```bash
bd ready --json                    # Confirm Phase 05 is unblocked
bd update <phase-05-id> --claim    # Claim before starting
# ... do the work ...
bd close <phase-05-id> --reason "Phase 05 PASS. Tagged phase-05-pass."
```