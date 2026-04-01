# Upstream SGLang Sync — Master Execution Plan

## Context

- **Fork:** `KHAEntertainment/sglang-mamba`
- **Upstream:** `sgl-project/sglang`
- **Delta:** 1,327 commits behind upstream, 97 ahead
- **Merge risk:** CRITICAL
- **Validation baseline:** Phases 0–8 must pass after the sync (snapshot E2E validated on granite-4.0-h-tiny)

Upstream has introduced overlapping abstractions (`SessionController`, `HiCache`, `HybridCacheController`, `HiMambaRadixCache`, and now owns `mamba_radix_cache.py`), but **does not** implement our product differentiators: disk snapshots, restore APIs, startup warm restore, or server-side Mamba state persistence.

## Strategy

Treat upstream's new Mamba/HiCache/session work as the new substrate, then re-port our snapshot persistence feature on top of it after explicit design reviews. Blind branch merging would damage restore semantics or upstream cache invariants.

## Phase Sequence

| Phase | Document | Complexity | Team |
|-------|----------|------------|------|
| 1 | `PHASE_01_DEPENDENCY_BASELINE.md` | LOW-MEDIUM | Solo |
| 2 | `PHASE_02_OBSERVABILITY_IMPORT_REBASE.md` | LOW | Solo |
| 3 | `PHASE_03_SESSION_CONTROLLER_PORT.md` | HIGH | 4-agent |
| 4 | `PHASE_04_MAMBA_CACHE_ARCHITECTURE_RECONCILE.md` | HIGH | 4-agent |
| 5 | `PHASE_05_SCHEDULER_IDLE_AND_POOL_FIXES.md` | MEDIUM | Solo/2-agent |
| 6 | `PHASE_06_SERVER_AND_HTTP_DRIFT.md` | MEDIUM | Solo |
| 6B | `PHASE_06B_BULK_UPSTREAM_MERGE.md` | MEDIUM | Solo |
| 7 | `PHASE_07_REAPPLY_SNAPSHOT_FEATURES.md` | HIGH | 4-agent |
| 8 | `PHASE_08_FULL_VALIDATION_AND_STRESS_TEST.md` | MEDIUM | 4-agent |

## Critical Path

```
PHASE_01 → PHASE_02 → PHASE_03 → PHASE_04 → PHASE_05 → PHASE_06 → PHASE_06B → PHASE_07 → PHASE_08
```

No parallel phases — each builds on substrate stability. Within HIGH phases, Research and Validation agents can run sub-tasks in parallel.

**GPU provisioning note:** Phases 01 and 02 (dependency resolution and import fixups) can run without a GPU. Phase 03 onward requires an A100 for validation steps. Plan RunPod provisioning accordingly to avoid burning hours on the mechanical early phases.

## Task Tracking with Beads (bd)

This sync uses [Beads](https://github.com/steveyegge/beads) (`bd`) for inter-session task state and dependency tracking. Phase docs define *what* to do; bd tracks *where we are*.

### One-time setup (before Phase 01)

```bash
cd <sglang-mamba repo root>
bash docs/migration-prep/phases/bd_setup.sh
```

This initializes bd in **stealth mode** (no `.beads/` committed to git) and creates the full phase dependency graph with sub-tasks for HIGH-complexity phases.

### Per-session workflow

Every Claude Code session should begin with:

```bash
bd ready --json        # What's unblocked right now?
bd show <phase-id>     # Details + sub-task status for the current phase
```

During work:

```bash
bd update <id> --claim           # Claim a task before starting
bd close <id> --reason "Done"    # Close when validation passes
```

When discovering new issues mid-phase:

```bash
bd create "Found: <issue>" -t bug -p 1 --deps discovered-from:<parent-id>
```

**Do NOT use `bd edit`** — it opens an interactive editor that hangs agents. Use `bd update <id> --description "new text"` instead.

### Design review checkpoints

DESIGN REVIEW tasks are P0 and block implementation sub-tasks. The Orchestrator (you or Opus) must close them with `--reason` documenting the decision before implementation proceeds.

## Isolation Requirements

1. **Worktree:** All merge work happens in `worktrees/upstream-sync-2026-Q1` (at repo root) branched from `main`. **Phase 01 creates this worktree** — see its Execution Step 0. All subsequent phases assume it already exists.
2. **Worktree Safety Check (run at the start of EVERY phase):**
   ```bash
   # Abort if not in the correct worktree branch
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
3. **Commit discipline:** Each phase ends as a discrete commit (or small series) tagged `phase-N-pass`.
4. **Rollback:** If a phase fails validation, reset worktree to the previous `phase-(N-1)-pass` tag and resume.
5. **No main contamination:** Do not push or merge to `main` until `PHASE_08` passes fully.

## Design Review Checkpoints

Two explicit checkpoints require Orchestrator sign-off before Implementation Agent proceeds:

1. **PHASE_03 — `create_new_request` restore flow:** Decide whether to inject restored Mamba state before `Session.create_req()` or add a post-creation hook.
2. **PHASE_04 — TierManager vs HybridCacheController:** Confirm that our `TierManager` (disk persistence / cross-session / startup warm) layers on top of upstream's `HybridCacheController` (GPU↔host offload) without replacement.

## Reference Documents

- `docs/migration-prep/UPSTREAM_SYNC_REPORT.md` — Commit audit, direct conflict table, Mamba/SSM intelligence
- `test/phases/config.sh` — Model paths, ports, snapshot directory
- `test/phases/phase-0*.md` — Individual validation procedures
- `CLAUDE.md` — Key source files, correct server flags, test resumption order

## Success Criteria

After `PHASE_08`, the following must all be true:

- `pip install -e "python/"` succeeds on A100
- Phase 0 (environment) **PASS**
- Phase 1 (stateless inference) **PASS**
- Phase 2 (MambaPool unit tests) **PASS** (5/5)
- Phase 3 (MambaRadixCache gauntlet) **PASS** (16/16)
- Phase 4 (live server no_buffer) **PASS**
- Phase 5 (Mamba2Metadata integrity) **PASS** (5/5)
- Phase 6 (extra_buffer strategy) **PASS**
- Phase 7 (snapshot system E2E) **PASS** (6/6)
- Phase 8 (true stateful inference) **PASS**
- Phase 9 (gauntlet stress tests) **PASS**
- Phase 10e (context window scaling 2K–128K) **PASS** (5/5 tiers)
- Phase 10f (resilience/crash testing) **PASS** (4/5 minimum; SIGTERM hang is known Bug #16)

**Model coverage:** granite-4.0-h-tiny is the gate model for the sync. Expanded model testing (Nemotron-Cascade, Codestral Mamba) is deferred to the multi-GPU cluster validation milestone.
