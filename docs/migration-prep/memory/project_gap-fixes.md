---
name: sglang-mamba gap fixes merged
description: Three snapshot/restore gaps fixed and merged; testing context for resumption
type: project
---

Gap-fix work complete and merged into the PR branch. Test on `main` branch.

**Gap 1:** Restored requests now sync `fill_ids` / `origin_input_ids` with restored Mamba state.
**Gap 2:** `create_new_request` restore correctly creates fresh request backed by restored snapshot state and preserves conversation namespace.
**Gap 3:** Startup restore now actually preloads latest snapshots into WARM tier instead of only logging them.

Additional correctness fixes in scheduler/tier_manager/io_struct. Targeted startup-restore tests pass.

**Existing phase results:**
- `test/phases/results/phase-00-granite-4.0-h-tiny-20260324.md`
- `test/phases/results/phase-02-granite-4.0-h-tiny-20260324-0503.md`
- `test/phases/results/phase-03-granite-4.0-h-tiny-20260324-0507.md`
- `test/phases/results/phase-05-granite-4.0-h-tiny-20260324-0509.md`

**Pending phases (in order):** Phase 3 (re-run) → Phase 4 → Phase 7 → (if pass) Phase 6 → Phase 8.

**Why:** Phase 7 especially validates all 3 gap fixes. Stop and diagnose at first failure.
**How to apply:** Start from Phase 3 re-run; use `test/phases/config.sh`; write results to `test/phases/results/`.
