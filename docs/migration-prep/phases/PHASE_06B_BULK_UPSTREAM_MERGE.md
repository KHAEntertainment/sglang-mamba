# PHASE 06B — Bulk Upstream Merge

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

Merge the remaining ~1,290 non-conflicting upstream commits that are not addressed by Phases 01–06. By this point, all architecturally sensitive files (scheduler, session controller, cache hierarchy, memory pools, server args, HTTP server, io_struct) have been reconciled. This phase sweeps in everything else — other model backends, CI, tooling, new features, bugfixes — so that Phase 07 re-applies snapshot features on the **complete** upstream codebase, not a partial one.

## Why Here (Between 06 and 07)

- Phases 01–06 resolved every HIGH and MEDIUM risk commit from the sync report's Direct Conflict Table.
- Phase 07 re-ports our snapshot product features. It must land on the full upstream substrate to avoid a second merge pass later.
- Doing the bulk merge *after* Phase 07 would risk undoing snapshot hook placements if any swept-in commit touches nearby code.

## Execution Steps

1. **Add upstream remote** (if not already present):
   ```bash
   git remote add upstream https://github.com/sgl-project/sglang.git || true
   git fetch upstream main
   ```

2. **Merge upstream/main into the worktree branch:**
   ```bash
   git merge upstream/main --no-edit
   ```

3. **Resolve conflicts.** At this point, conflicts should be limited to:
   - Files already reconciled in Phases 01–06 where upstream has additional commits beyond what we cherry-picked. These should be straightforward since we've already adopted upstream's structure.
   - Possible `pyproject.toml` or CI file collisions.

   **Conflict resolution rule:** For any file already reconciled in a prior phase, preserve our phase-reconciled version and manually integrate only genuinely new upstream additions (if any). Do NOT let the merge overwrite phase work.

4. **Verify no snapshot code was clobbered:**
   ```bash
   # Quick check that our snapshot package is intact
   ls python/sglang/srt/snapshot/
   grep -r "enable-snapshot-persistence" python/sglang/srt/server_args.py
   grep -r "save_snapshot" python/sglang/srt/entrypoints/http_server.py
   ```
   If any of these are missing, the merge overwrote phase work — reset and resolve manually.

5. **Run baseline validation:**
   - Phase 0 (environment verification)
   - Phase 1 (stateless inference baseline)
   - Phase 2 (MambaPool unit tests)

   These are fast and catch import breakage or dependency issues introduced by the bulk merge.

6. **If Phase 0/1/2 pass**, run the broader suite:
   - Phase 4 (live server no_buffer)
   - Phase 7 (snapshot E2E)

## Validation Criteria

- `pip install -e "python/"` succeeds.
- Phase 0 (environment) **PASS**
- Phase 1 (stateless inference) **PASS**
- Phase 2 (MambaPool unit tests) **PASS** (5/5)
- Phase 4 (live server no_buffer) **PASS**
- Phase 7 (snapshot E2E) **PASS** (6/6)
- Snapshot REST endpoints respond to health probes.

## Rollback Plan

- Reset worktree to `phase-06-pass` tag.
- If the merge produces an unmanageable number of conflicts, consider an alternative approach: `git merge upstream/main --strategy=ours` for non-critical paths, then selectively merge specific directories.

## Estimated Complexity

**MEDIUM** — 4 to 8 hours. The merge itself may be fast (most conflicts already resolved), but validation takes time on A100.

## Dependencies

- `PHASE_06_SERVER_AND_HTTP_DRIFT` complete and tagged `phase-06-pass`.

## Team Structure

**Solo agent** — this is primarily mechanical merge resolution and validation. Escalate to 2-agent if unexpected conflicts arise in Mamba-adjacent code.

## bd Workflow

```bash
bd ready --json                     # Confirm Phase 06B is unblocked
bd update <phase-06b-id> --claim    # Claim before starting
# ... do the work ...
bd close <phase-06b-id> --reason "Phase 06B PASS. Bulk merge complete. Tagged phase-06b-pass."
```

## Notes

- This phase produces a `phase-06b-pass` tag.
- After this phase, the worktree contains the **full upstream codebase** plus all Phase 01–06 reconciliation work.
- Phase 07 (Re-apply Snapshot Features) builds on this complete substrate.
