# PHASE 03 — Session Controller Port

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

Port our scheduler/session customizations onto upstream's new `SessionController` abstraction so that snapshot restore, save hooks, and session-aware request creation continue to work.

## Upstream Commits to Integrate

| Commit | Risk | Topic |
|--------|------|-------|
| `5acb45cf3` | HIGH | Extracts `SessionController` from `Scheduler`; replaces `Scheduler.sessions` |
| `c6cb0c964` | HIGH | Adds streaming sessions and `SessionAwareCache` fast path |
| `e08ef0675` | HIGH | Gates streaming sessions behind `--enable-streaming-session` and adds Spec V2 guards |
| `e1ee68d0f` | MEDIUM | Releases multimodal features on session close and expands rerun-ut behavior; session-close path may interact with restore/save lifecycle |

## Files Touched

- `python/sglang/srt/managers/scheduler.py` (major structural changes)
- `python/sglang/srt/managers/session_controller.py` (new upstream file)
- `python/sglang/srt/managers/io_struct.py` (session request models)
- `python/sglang/srt/managers/tokenizer_manager.py` (session routing)
- `python/sglang/srt/mem_cache/memory_pool.py` (indirect, via `SessionAwareCache`)
- `python/sglang/srt/server_args.py` (session gating flags)

## Decision Points

### 1. Adopt `SessionController` fully
**Decision:** Adopt upstream's `SessionController` as the single source of truth for session existence and lifecycle.

Our fork currently keeps session logic inside `Scheduler` (e.g., `Scheduler.sessions`, `create_new_request` restore flow). This must move to coordinated hooks or thin wrappers around `SessionController`.

### 2. Snapshot hook placement
**Decision:** Keep our snapshot hooks but move their attachment points.

- `init_snapshot_system()` — still called from `Scheduler.__init__()` after `SessionController` is initialized.
- `maybe_save_mamba_snapshots()` — still called post-forward in the batch result path, but must interact with `SessionController` or `Req` objects rather than a raw `sessions` dict.
- `handle_save_snapshot`, `handle_restore_snapshot`, etc. — still dispatched by the scheduler event loop, but may delegate session lookups to `self.scheduler.session_controller`.

### 3. `create_new_request` restore flow — DESIGN REVIEW REQUIRED
**Conflict:** Upstream's `Session.create_req()` now handles session-aware request creation (trimming BOS, managing `input_ids` based on `replace` / `drop_previous_output`). Our restore flow currently creates a `Req` directly and then injects restored Mamba state into it.

**Options to evaluate:**
- **Option A:** Inject restored Mamba state *before* `Session.create_req()` returns, by overriding or extending `Session.create_req()`.
- **Option B:** Add a post-`create_req()` snapshot restore hook in the scheduler's request routing path (after the session request is created but before it enters the batch).

**Orchestrator must choose one before Implementation Agent proceeds.** Research Agent should read upstream `session_controller.py` to determine which seam is least invasive.

### 4. Server args coexistence
**Decision:** Adopt `--enable-streaming-session` and its gating plumbing. Re-apply our `--enable-snapshot-persistence`, `--snapshot-dir`, and related flags alongside it. The two features are orthogonal (streaming sessions are runtime KV sharing; snapshot persistence is disk durability).

## Execution Steps

1. Research Agent reads upstream `session_controller.py` and reports on:
   - `SessionController` class API (`open`, `close`, `get`, `maybe_reap`)
   - `Session.create_req()` signature and return type
   - Where `Scheduler` delegates to `SessionController` for request creation

2. Implementation Agent merges the three upstream commits into the worktree, resolving conflicts in `scheduler.py` where our snapshot handlers currently live.

3. At the `create_new_request` restore flow, pause for Orchestrator decision (Option A vs Option B).

4. Port the restore flow according to the chosen option.

5. Re-apply snapshot server args and ensure `TokenizerManager` snapshot forwarding methods still route correctly.

6. Validation Agent runs:
   - Phase 1 (stateless inference baseline)
   - Phase 4 (live server no_buffer)
   - Phase 7 (snapshot E2E)
   - Phase 8 (true stateful inference)

## Validation Criteria

- Phase 1 (stateless inference) **PASS**
- Phase 4 (live server no_buffer) **PASS**
- Phase 7 (snapshot E2E) **PASS** (6/6)
- Phase 8 (true stateful inference) **PASS**

## Rollback Plan

- Reset worktree to `phase-02-pass` tag.

## Estimated Complexity

**HIGH** — 8 to 16 hours. The structural change to session ownership is the deepest refactor in the entire sync.

## Dependencies

- `PHASE_02_OBSERVABILITY_IMPORT_REBASE` complete and tagged `phase-02-pass`.

## Team Structure

**4-agent team required:**
1. **Orchestrator (Opus)** — Sequencing, design decision at `create_new_request` conflict point, coordination.
2. **Research Agent (Sonnet)** — Queries DeepWiki for `SessionController` API contracts and `Session.create_req()` internals.
3. **Implementation Agent (Sonnet)** — Executes merge and ports snapshot hooks.
4. **Validation Agent (Sonnet)** — Runs test phases after each checkpoint, documents results, flags regressions.

## bd Workflow

Phase 03 has 4 sub-tasks in bd with dependencies:

1. **Research** (unblocked after Phase 02) → `bd update <id> --claim` → `bd close <id>`
2. **DESIGN REVIEW** (unblocked after Research) → Orchestrator closes with `--reason "Option A/B chosen because..."`
3. **Implementation** (unblocked after design review) → `bd update <id> --claim` → `bd close <id>`
4. **Validation** (unblocked after implementation) → `bd close <id> --reason "Phase 1/4/7/8 PASS"`

Close the parent Phase 03 task after all sub-tasks pass. Tag `phase-03-pass`.
