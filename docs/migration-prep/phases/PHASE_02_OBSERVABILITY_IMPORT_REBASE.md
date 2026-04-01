# PHASE 02 — Observability Import Rebase

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

Absorb the observability refactor (`3b8930227`) that moves metrics/tracing modules into `srt/observability/` so that our custom code is re-ported onto the new import paths rather than the old ones.

## Upstream Commit to Integrate

| Commit | Topic |
|--------|-------|
| `3b8930227` | Observability code cleanup: moves `req_time_stats`, `scheduler_metrics_mixin`, `metrics_collector`, and `trace` into `srt/observability/` |

## Files Touched

- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/entrypoints/http_server.py`
- `python/sglang/srt/managers/io_struct.py`
- New directory: `python/sglang/srt/observability/`

## Decision Points

1. **Adopt upstream module layout fully.** Do not preserve old import paths.
2. **Update our fork's custom snapshot imports** in `scheduler.py`, `http_server.py`, and `io_struct.py` to reference `sglang.srt.observability.*`.
3. No functional behavior changes — this is purely import/path alignment.

## Execution Steps

1. Cherry-pick / apply the observability refactor commit.
2. Fix any merge conflicts in `scheduler.py`, `http_server.py`, and `io_struct.py` where our custom code references the old module paths.
3. Run a quick import sanity check:
   ```bash
   python -c "import sglang.srt.managers.scheduler"
   python -c "import sglang.srt.entrypoints.http_server"
   python -c "import sglang.srt.managers.io_struct"
   ```
4. Start the server and run Phase 1 (stateless inference baseline) to confirm no runtime import failures.

## Validation Criteria

- `python -c "import sglang.srt.managers.scheduler"` succeeds.
- `python -c "import sglang.srt.entrypoints.http_server"` succeeds.
- Phase 1 server start succeeds without `ModuleNotFoundError`.

## Rollback Plan

- `git checkout --` the affected files and re-apply the import fixes from scratch.

## Estimated Complexity

**LOW** — 1 to 2 hours. Pure mechanical import updates.

## Dependencies

- `PHASE_01_DEPENDENCY_BASELINE` must be complete and stable.

## Team Structure

**Solo agent**.

## bd Workflow

```bash
bd ready --json                    # Confirm Phase 02 is unblocked
bd update <phase-02-id> --claim    # Claim before starting
# ... do the work ...
bd close <phase-02-id> --reason "Phase 02 PASS. Tagged phase-02-pass."
```
