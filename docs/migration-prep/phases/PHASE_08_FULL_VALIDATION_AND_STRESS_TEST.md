# PHASE 08 — Full Validation and Stress Test

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

Run the complete validation suite and Phase 9 gauntlet to prove the merged system is production-ready. No upstream code is merged in this phase; it is pure validation, regression diagnosis, and fix-forward.

## Upstream Commits to Integrate

None. This phase is validation-only.

## Files Potentially Touched

Any file where a regression is discovered during testing.

## Decision Points

### 1. Fix-forward vs reset
**Decision rule:**
- If a regression is small and localized (≤ 2 files, ≤ 50 lines), fix it within this phase and continue.
- If a regression indicates a fundamental architectural mismatch with an earlier phase, **reset worktree to the prior phase tag** and re-execute that phase with the new knowledge.

### 2. Stress test scope
**Decision:** Run the full Phase 9 gauntlet as documented in `test/phases/phase-09-gauntlet-stress-tests.md`. This includes:
- Concurrency floods
- Rapid eviction cycles
- Repeated identical requests
- Alternating long/short contexts
- Multi-turn conversation floods

### 3. Production-readiness gate (Phase 10e/10f)
**Decision:** After Phase 9 passes, run the context-scaling and resilience tests that were validated pre-sync:
- **Phase 10e (Context Window Scaling):** 2K / 8K / 32K / 64K / 128K token tiers on granite-4.0-h-tiny. Confirms constant snapshot size and ~2ms restore across context lengths.
- **Phase 10f (Resilience / Crash Testing):** Client disconnect, SIGKILL mid-inference, SIGKILL during snapshot write, graceful SIGTERM, abort + snapshot save. The pre-sync baseline was 4/5 PASS (SIGTERM hang is Bug #16 / KHA-174, may resolve during sync).

These tests represent the production-readiness bar established before the sync and must re-pass.

## Execution Steps

1. Run Phase 0 (environment verification).
2. Run Phase 1 (stateless inference baseline).
3. Run Phase 2 (MambaPool unit tests).
4. Run Phase 3 (MambaRadixCache gauntlet).
5. Run Phase 4 (live server no_buffer).
6. Run Phase 5 (Mamba2Metadata integrity).
7. Run Phase 6 (extra_buffer strategy).
8. Run Phase 7 (snapshot system E2E).
9. Run Phase 8 (true stateful inference).
10. Run Phase 9 (gauntlet stress tests).
11. Run Phase 10e (context window scaling: 2K/8K/32K/64K/128K).
12. Run Phase 10f (resilience/crash testing: 5 scenarios).

For each phase:
- Document pass/fail in `test/phases/results/phase-XX-<model>-<date>.md`.
- Capture full server logs.
- If a phase fails, stop and diagnose before proceeding.

## Validation Criteria

| Phase | Description | Criterion |
|-------|-------------|-----------|
| 0 | Environment verification | **PASS** |
| 1 | Stateless inference baseline | **PASS** |
| 2 | MambaPool unit tests | **PASS** (5/5) |
| 3 | MambaRadixCache gauntlet | **PASS** (16/16) |
| 4 | Live server no_buffer | **PASS** |
| 5 | Mamba2Metadata integrity | **PASS** (5/5) |
| 6 | Extra buffer strategy | **PASS** |
| 7 | Snapshot system E2E | **PASS** (6/6) |
| 8 | True stateful inference | **PASS** |
| 9 | Gauntlet stress tests | **PASS** |
| 10e | Context window scaling (2K–128K) | **PASS** (5/5 tiers) |
| 10f | Resilience / crash testing | **PASS** (4/5 minimum; SIGTERM hang is known Bug #16) |

## Rollback Plan

- If a phase fails, diagnose root cause.
- If fix is trivial → patch forward.
- If fix requires revisiting an earlier phase → reset to `phase-N-pass` tag for the appropriate N and re-execute from there.

## Estimated Complexity

**MEDIUM** — 6 to 12 hours. Time is dominated by test runtime on A100 and regression debugging.

## Dependencies

- `PHASE_07_REAPPLY_SNAPSHOT_FEATURES` complete and tagged `phase-07-pass`.

## Team Structure

**4-agent team recommended:**
1. **Orchestrator (Opus)** — Go/no-go decisions at failure points, reset-or-fix-forward rulings.
2. **Research Agent (Sonnet)** — Available for DeepWiki lookups if regressions touch unfamiliar upstream internals.
3. **Implementation Agent (Sonnet)** — Applies any fix-forward patches.
4. **Validation Agent (Sonnet)** — Primary role: runs tests, collects logs, reports results after each phase.

## bd Workflow

Phase 08 has 3 sub-tasks in bd:

1. **Core validation (Phases 0-9)** → `bd update <id> --claim` → `bd close <id> --reason "Phases 0-9 PASS"`
2. **Phase 10e context scaling** (unblocked after core) → `bd close <id> --reason "5/5 tiers PASS"`
3. **Phase 10f resilience** (unblocked after core) → `bd close <id> --reason "4/5 PASS (SIGTERM known)"`

Note: 10e and 10f can run in parallel after core validation passes.

Close the parent Phase 08 task after all sub-tasks pass. Tag `phase-08-pass`.

## Completion Criteria

This phase is complete only when **all 12 test phases pass** (Phases 0–9 + 10e + 10f) and the worktree is tagged `phase-08-pass`. At that point, the upstream sync is validated and ready for merge into `main`.