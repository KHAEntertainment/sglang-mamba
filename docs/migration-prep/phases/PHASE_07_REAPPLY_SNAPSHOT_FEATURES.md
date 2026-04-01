# PHASE 07 — Re-apply Snapshot Features

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

Re-port and validate our snapshot-specific product features on the updated upstream substrate. This is the phase where our product differentiation is explicitly preserved.

## Custom Code to Re-apply

- `python/sglang/srt/snapshot/` package (entire directory)
  - `mamba_snapshot.py`
  - `tier_manager.py`
  - `mamba_host_pool.py`
  - `conversation_tracker.py`
  - `snapshot_policy.py`
  - `snapshot_hooks.py`
  - `__init__.py`
- `python/sglang/srt/managers/io_struct.py` — snapshot I/O structs
- `python/sglang/srt/managers/scheduler.py` — snapshot handlers and `init_snapshot_system`
- `python/sglang/srt/managers/tokenizer_manager.py` — snapshot response queues and forwarding
- `python/sglang/srt/entrypoints/http_server.py` — snapshot REST routes
- `python/sglang/srt/server_args.py` — snapshot CLI flags
- `GenerateReqInput.conversation_id` field
- `fill_ids` restore sync logic
- Startup warm restore (`restore_snapshots_on_startup`)

## Files Touched

All of the above, plus any new upstream files that our hooks must now interface with (e.g., `session_controller.py`).

## Decision Points

### 1. Preserve the user-facing API
**Decision:** The external REST API and CLI surface must remain stable.
- `RestoreSnapshotReqInput` keeps `continuation_ids` and `max_new_tokens`.
- `RestoreSnapshotReqOutput` keeps `output_text` and `output_ids`.
- CLI flags keep their current names.

### 2. Adapt internal hook points to upstream's new structure
**Decision:** Internal implementation details may change, but semantics must not.

- `init_snapshot_system()` — attach to `Scheduler.__init__()` after `SessionController` initialization.
- `maybe_save_mamba_snapshots()` — attach post-forward in the batch result path, using the new request/session traversal pattern.
- `handle_restore_snapshot()` — if using `create_new_request=True`, route generated tokens back through the snapshot result channel (the Phase 8 fix).

### 3. MambaPool extract/inject compatibility
**Decision:** Confirm that `MambaSnapshotManager.extract_state_from_pool` and `inject_state_to_pool` work with upstream's `MambaPool` tensor layout.

If upstream changed `MambaPool.State` or `SpeculativeState` fields, update the extract/inject code accordingly while keeping the serialization format backward-compatible.

### 4. `fill_ids` restore synchronization
**Decision:** Preserve the `fill_ids` sync fix from PR #6. After restoring Mamba state into a new request, ensure `req.fill_ids` is synchronized with the restored token sequence so that decode bookkeeping is consistent.

## Execution Steps

1. **Research Agent** uses DeepWiki to verify:
   - Current `MambaPool.State` / `SpeculativeState` dataclass fields in upstream
   - `Scheduler` batch result path where post-forward hooks run
   - `SessionController` interaction points for session-aware restore

2. **Implementation Agent** copies the `snapshot/` package into the worktree and adapts any broken imports or pool references.

3. Re-apply snapshot I/O structs to the updated `io_struct.py`.

4. Re-apply snapshot handlers to the updated `scheduler.py`, adapting to `SessionController` presence.

5. Re-apply `TokenizerManager` response queues and forwarding.

6. Re-apply REST routes to the updated `http_server.py` and CLI flags to `server_args.py`.

7. Validation Agent runs **ALL** phases 0–8.

## Validation Criteria

- **ALL** phases 0–8 must **PASS**.
- Snapshot save/restore cycle must complete without errors.
- True stateful inference (Phase 8) must recall semantic context across turns.

## Rollback Plan

- Reset worktree to `phase-06b-pass` tag.

## Estimated Complexity

**HIGH** — 8 to 14 hours. This is the "product preservation" phase; every custom feature must be verified.

## Dependencies

- `PHASE_06B_BULK_UPSTREAM_MERGE` complete and tagged `phase-06b-pass`.

## Team Structure

**4-agent team required:**
1. **Orchestrator (Opus)** — Sequencing, API stability enforcement, hook placement decisions.
2. **Research Agent (Sonnet)** — Verifies upstream struct layouts and scheduler paths before code changes.
3. **Implementation Agent (Sonnet)** — Ports snapshot package and handlers.
4. **Validation Agent (Sonnet)** — Runs full phase suite 0–8 and documents regressions.

## bd Workflow

Phase 07 has 3 sub-tasks in bd with dependencies:

1. **Research** (unblocked after Phase 06B) → `bd update <id> --claim` → `bd close <id>`
2. **Implementation** (unblocked after Research) → `bd update <id> --claim` → `bd close <id>`
3. **Validation** (unblocked after implementation) → `bd close <id> --reason "ALL phases 0-8 PASS"`

Close the parent Phase 07 task after all sub-tasks pass. Tag `phase-07-pass`.
