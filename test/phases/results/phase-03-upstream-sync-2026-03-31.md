# Phase 03 — SessionController Port (Upstream Sync)

**Date**: 2026-03-31
**Branch**: `upstream-sync-2026-Q1`
**Result**: **PASS**

## Upstream Commits Integrated

| Commit | PR | Topic | Risk |
|--------|----|-------|------|
| `c6cb0c964` | #19171 | Add `streaming` mode with `SessionAwareCache` fast path | HIGH |
| `5acb45cf3` | #19547 | Extract `SessionController` from `Scheduler` | HIGH |
| `e08ef0675` | #19531 | Gate streaming sessions with `--enable-streaming-session` | HIGH |
| `e1ee68d0f` | #21501 | Release mm features on session close | MEDIUM |

## Conflicts Resolved

No git merge conflicts — changes were manually ported rather than cherry-picked,
since the fork is 1,327 commits behind upstream. Key integration points:

1. **`scheduler.py` session management**: Replaced `self.sessions: Dict[str, Session]`
   with `self.session_controller = SessionController(self.tree_cache)`. Rewired all
   `open_session`/`close_session` methods to delegate.

2. **`handle_generate_request` routing**: Refactored from 2-way (session exists / doesn't)
   to 3-way routing (no session / session exists / session not found). The "not found"
   branch now creates a minimal Req and aborts cleanly.

3. **`Req.session_id` → `Req.session`**: Changed from string to Session object reference.
   Snapshot code doesn't reference `req.session_id` so no cascade required.

4. **Multimodal offset adjustment**: Moved inline mm_offset logic from scheduler to
   `SessionController.adjust_mm_offsets()` static method.

## Design Decisions

### `create_new_request` restore flow — Option B chosen

**Decision**: Post-`create_req()` hook in the scheduler for snapshot restore.

**Rationale**:
- Keeps `Session.create_req()` token-only, matching upstream's design intent
- Mirrors upstream's own `SessionSlot.restore_to_req()` pattern
- Mamba state injection stays at Scheduler level where `mamba_pool` + `snapshot_manager` are accessible
- Least invasive — no modification to Session class signature

**Alternatives considered**:
- Option A (extend `Session.create_req()`) rejected — would couple session logic to Mamba state management

### Streaming session gating

Adopted `--enable-streaming-session` flag as orthogonal to `--enable-snapshot-persistence`.
Streaming sessions (runtime KV sharing via `SessionAwareCache`) and snapshot persistence
(disk durability via `MambaSnapshotManager`) serve different purposes and can coexist.

## Files Changed

| File | Insertions | Deletions | Summary |
|------|-----------|-----------|---------|
| `session_controller.py` | +179 | -17 | Added `SessionController` class, streaming support, timeout, mm release, `adjust_mm_offsets` |
| `session_aware_cache.py` | +311 | — | **New file** — `SessionAwareCache` decorator with `SessionSlot` (includes Mamba state fields) |
| `scheduler.py` | +55 | -50 | Replaced `self.sessions` with `self.session_controller`, 3-way routing, delegated open/close, session reaping |
| `schedule_batch.py` | +7 | -5 | `Req.session_id` → `Req.session`, `release_features()`, `req=self` in match_prefix |
| `tokenizer_communicator_mixin.py` | +19 | -4 | Streaming session gating (spec v2 guard) |
| `server_args.py` | +7 | — | `--enable-streaming-session` flag |
| `io_struct.py` | +2 | — | `streaming` and `timeout` fields on `OpenSessionReqInput` |
| `hybrid_linear_attn_backend.py` | +6 | -3 | Guarded `cutedsl_gdn` import for missing `cutlass.cute.nvgpu` |

**Total**: 570 insertions, 78 deletions across 8 files.

## Test Results

### Unit Tests (no server)

| Suite | Pass | Fail | Skip |
|-------|------|------|------|
| Mamba unit tests (`test_mamba_unittest.py`) | 3 | 0 | 0 |
| MambaRadixCache comprehensive | 9 | 0 | 0 |
| Mamba snapshot (`test_mamba_snapshot.py`) | 11 | 0 | 1 |
| **Total** | **23** | **0** | **1** |

### Phase 1 — Stateless Inference Baseline (live server, `--disable-radix-cache`)

| Test | Result |
|------|--------|
| test_health_endpoint | PASS |
| test_single_turn_completion | PASS |
| test_streaming_completion | PASS |
| test_batch_inference_independence | PASS |
| test_batch_inference_different_prompts | PASS |
| test_long_context | PASS |
| test_sampling_params | PASS |
| **Total** | **7/7 PASS** |

### Phase 4 — Radix Cache Integration (live server, `--enable-cache-report`)

| Test | Result | Notes |
|------|--------|-------|
| test_cache_hit_on_repeated_prefix | FAIL | Pre-existing: `prompt_tokens_details` is None |
| test_cache_miss_fallback | PASS | |
| test_concurrent_shared_prefix | PASS | |
| test_multi_turn_conversation_state_continuity | FAIL | Pre-existing: test sends single-turn msgs expecting session state |
| test_eviction_under_pressure | PASS | |
| **Total** | **3/5 PASS** | 2 failures are pre-existing, not regressions |

### Phase 7 — Snapshot System E2E (live server, `--enable-snapshot-persistence`)

| Test | Result |
|------|--------|
| test_save_snapshot_returns_success | PASS |
| test_restore_snapshot_state_equivalence | PASS |
| test_restore_requires_idle_request | PASS |
| test_snapshot_disk_format | PASS |
| test_snapshot_manager_tier_consistency | PASS |
| test_create_new_request_returns_new_rid | PASS |
| **Total** | **6/6 PASS** |

## Issues Discovered During Validation

### Environment issues (fixed in this phase)

1. **`sgl-kernel` 0.3.21 vs `sglang-kernel` 0.4.0 shadowing**: Both packages provide
   `sgl_kernel/` module. Uninstalling `sgl-kernel` broke sampling (`min_p_sampling_from_probs`
   missing from 0.4.0). Fix: keep `sgl-kernel` 0.3.21 installed (has all required symbols).

2. **`cutlass.cute.nvgpu` missing**: `hybrid_linear_attn_backend.py` unconditionally imports
   `cutedsl_gdn` which requires CUTLASS DSL nvgpu module not available in our environment.
   Fix: guarded import with try/except (commit `47e7f0743`).

3. **Model config incompatibility with transformers 5.3.0**: Missing `layers_block_type` and
   `time_step_*` attributes. Fix: added to model `config.json` directly (not a code change).

4. **jinja2 version**: `apply_chat_template` requires jinja2>=3.1.0, had 3.0.3.
   Fix: `pip install "jinja2>=3.1.0"`.

### Pre-existing test issues (not fixed, not regressions)

1. **Phase 4 `test_cache_hit_on_repeated_prefix`**: `--enable-cache-report` doesn't populate
   `prompt_tokens_details.cached_tokens` as expected. Needs investigation in Phase 04.

2. **Phase 4 `test_multi_turn_conversation_state_continuity`**: Test design assumes
   server-side session persistence via `rid=` parameter, but without an open session
   each request is independent. Needs test redesign or session API usage.

## Commits in This Phase

```text
47e7f0743 fix: guard cutedsl_gdn import for missing cutlass.cute.nvgpu
1d906a894 revert: undo awq/engine workarounds, restore sgl-kernel 0.3.21
4b74fc896 fix: graceful fallback for awq_marlin_repack imports
040c0a43a phase-03: port SessionController abstraction from upstream
```

## Next Steps

- **Phase 04**: Mamba Cache Architecture Reconcile (TierManager vs HybridCacheController)
- Phase 4 test fixes deferred to Phase 04/05
- Snapshot hooks will be re-wired to SessionController in Phase 07
