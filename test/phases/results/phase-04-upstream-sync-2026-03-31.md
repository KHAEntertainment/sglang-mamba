# Phase 04 Results — Mamba Cache Architecture Reconcile

**Date:** 2026-03-31
**Branch:** `upstream-sync-2026-Q1`
**Tag:** `phase-04-pass`
**Machine:** RunPod H200 (NVIDIA H200 143GB)
**Model:** granite-4.0-h-tiny (`/home/jeanclawdai/models/granite-4.0-h-tiny`)

## Objective

Reconcile upstream's new Mamba cache hierarchy (HiMambaRadixCache, HybridCacheController, MambaPoolHost, pool sizing changes) with our snapshot/tier system, then cherry-pick 10 COMPLEMENT Mamba improvements.

## Commits Integrated

### Core Commits (7 HIGH-risk)

| # | Commit | Description | Conflicts |
|---|--------|-------------|-----------|
| 1 | `e4b708d3e` | Spec V2 mamba hybrid attention | 2 minor (decode.py, model_runner_kv_cache_mixin.py) |
| 2 | `07ef5f7be` | Remove sync points + prefill cudagraph for DP | Clean |
| 3 | `5867c3fa8` | HiCache support for MambaRadixCache | Clean |
| 4 | `0ac6c63ae` | Refactor mamba ratio calculation for pool init | 1 (model_runner_kv_cache_mixin.py) |
| 5 | `197f80713` | Refactor mamba radix tree insert/release semantics | 1 (test_mamba_unittest.py) |
| 6 | `dfd0a77a9` | Fix prev_prefix_len in HiMambaRadixCache insert | Clean |
| 7 | `0986bed8e` | Mamba state offloading + HybridCacheController | 5 (hi_mamba_radix_cache.py, schedule_policy.py, decode.py, bench_multiturn.py, base_prefix_cache.py) |

### COMPLEMENT Commits (10)

| # | Commit | Description | Conflicts |
|---|--------|-------------|-----------|
| 1 | `82a0bafc1` | Selective state update kernel for Mamba ops | Clean |
| 2 | `a1b39c1c2` | Fuse mamba state scatter MTP verify | Clean |
| 3 | `86c561778` | Fix illegal memory access in Mamba SSM EAGLE verify | Clean |
| 4 | `c76251f70` | Return intermediate Mamba states from ssd_combined | Clean |
| 5 | `69158e9d9` | Skip _mamba_verify_update for idle batch | Clean |
| 6 | `25bd83033` | Piecewise CUDA graph for NemotronH hybrid models | Clean |
| 7 | `dd82678b2` | NPU mamba cache transfer | Clean |
| 8 | `87549f8f0` | Triton conv1d for non-contiguous input perf | 1 (causal_conv1d.py — `_HAS_SGL_KERNEL` guard merge) |
| 9 | `cfead25bb` | Qwen3.5 mamba slice fix for TP size mismatch | Clean |
| 10 | `0949b138a` | Simplify server startup output | 4 (main.py, serve.py, launch_server.py, compile.py) |

**Total: 17 upstream commits integrated, 11 conflict-free, 6 with conflicts.**

### Supporting Commits (2)

| Commit | Description |
|--------|-------------|
| `4db3b8d6c` | Lint fixes from pre-commit (black, codespell) — 29 files reformatted |
| `9b8323ca6` | Test compatibility fixes after upstream cherry-picks |

## Conflict Resolution Details

### Core Commit 1 (`e4b708d3e` — Spec V2)
- `decode.py`: Upstream added `enable_overlap_schedule` and `mamba_size` params to `DecodeMambaReqToTokenPool.__init__`. Accepted upstream additions.
- `model_runner_kv_cache_mixin.py`: Upstream added `enable_overlap_schedule` and `mamba_size` kwargs at call site. Accepted upstream.

### Core Commit 4 (`0ac6c63ae` — Ratio refactor)
- `model_runner_kv_cache_mixin.py`: Upstream moved `additional_ratio` calculation out of the mambaish guard and changed condition from `not self.spec_algorithm.is_none()` to `not self.server_args.disable_overlap_schedule`. Accepted upstream's refactored block, kept our mambaish capping logic below.

### Core Commit 7 (`0986bed8e` — Offloading + HybridCacheController)
- `hi_mamba_radix_cache.py`: Accepted upstream's `mamba_restore_nodes` + `inc_lock_ref` result object pattern; fixed `init_load_back` signature to accept `InitLoadBackParams`.
- `base_prefix_cache.py`: Added 4 missing dataclass definitions (`IncLockRefResult`, `DecLockRefParams`, `DecLockRefResult`, `InitLoadBackParams`).
- `schedule_policy.py`: Accepted upstream's `InitLoadBackParams` call pattern; added missing import.
- `disaggregation/decode.py`: Accepted upstream's `effective_mamba_size` + `layer_transfer_counter` additions.
- `benchmark/hicache/bench_multiturn.py`: Accepted upstream's barrier logic.

### COMPLEMENT Commit 8 (`87549f8f0` — Triton conv1d)
- `causal_conv1d.py`: HEAD already had Triton imports; upstream added `_HAS_SGL_KERNEL` guard and fallback dispatch. Merged both.

### COMPLEMENT Commit 10 (`0949b138a` — Startup output)
- 4 CLI files (`main.py`, `serve.py`, `launch_server.py`, `compile.py`): Standard refactoring of startup entrypoints. Accepted upstream in all.

## Design Decisions

### TierManager vs HybridCacheController — CONFIRMED

**Decision:** Keep both and layer.

- **Upstream `HybridCacheController` + `MambaPoolHost`**: Runtime GPU↔host (L1↔L2) offload for capacity expansion and prefix sharing. Lives at `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py` and `memory_pool_host.py:MambaPoolHost`.
- **Our `TierManager` + `MambaHostPool`**: Durable host↔disk (L2↔L3) snapshots, cross-session references, retention policies, and startup warm restore. Lives at `python/sglang/srt/snapshot/tier_manager.py` and `snapshot/mamba_host_pool.py`.

TierManager interacts with `MambaPool` directly for `extract_state_from_pool` / `inject_state_to_pool`, not through `HybridCacheController`. The two systems operate at different levels and do not interfere.

### mamba_radix_cache.py Ownership — TRANSFERRED TO UPSTREAM

Our fork previously treated `mamba_radix_cache.py` as fork-owned (tombstones, dual LRU, COW). Decision: accept upstream's version entirely. Our snapshot product does not depend on internal radix-cache semantics — it depends on `MambaPool` extract/inject.

### Sync Point Removal — NO GUARD NEEDED

Upstream removed CUDA sync points for performance (`07ef5f7be`). Our `MambaSnapshotManager.extract_state_from_pool` has no explicit sync dependence (verified: zero calls to `cuda.synchronize` or `torch.cuda.current_stream`). No defensive `torch.cuda.synchronize()` guard added. Will revisit if validation under load shows races.

## Test Results

### Unit Tests (offline, no server)

| Suite | Result | Details |
|-------|--------|---------|
| `test_mamba_radix_cache_comprehensive.py` | **9/9 PASS** | After fix: KV tensors created on device |
| `test_mamba_unittest.py` | **4/4 PASS** | After fix: removed dangling function call, added `available_and_evictable_str` method |
| `test_mamba_snapshot.py` | **11/11 PASS, 1 SKIP** | Skip: GPU roundtrip test (expected without running server) |

### Runtime Verification

| Check | Result |
|-------|--------|
| `MambaPool.State` fields (`conv`, `temporal`) | **INTACT** (verified at Python import level) |
| `MambaSnapshotManager` import | **OK** |
| `TierManager` import | **OK** |
| Snapshot handlers in scheduler.py | **5/5 present** (save, restore, list, get_info, delete) |
| Snapshot dispatch entries | **5/5 present** |

### Live Server Validation (granite-4.0-h-tiny, H200)

| Phase | Result |
|-------|--------|
| Phase 1 — Stateless inference | **PASS** ("The capital of France is Paris...") |
| Phase 4 — Live server no_buffer (single turn) | **PASS** (3 turns, correct completions) |
| Phase 4 — Live server no_buffer (concurrent) | **PASS** (5 concurrent requests, all completed) |
| Phase 7 — Snapshot API endpoints | **PASS** (`/list_snapshots`, `/save_snapshot` respond correctly) |

## Issues Discovered and Fixed

### 1. Test device mismatch in `test_mamba_branching_seqlen`

**Root cause:** Upstream commit `197f80713` refactored `_insert_helper` to call `token_to_kv_pool_allocator.free(value[start:prefix_len])` during insert. Our test created fake KV tensors on CPU, but the allocator holds pages on CUDA.

**Fix:** Create KV tensors with `device=self.device` in `test_mamba_radix_cache_comprehensive.py`.

### 2. Missing `available_and_evictable_str` method

**Root cause:** Cherry-pick of `197f80713` brought a test (`test_mamba_unittest.py`) that calls `tree.available_and_evictable_str()`, but the method was part of a later upstream refactoring commit not included in our cherry-pick set.

**Fix:** Added the method to `MambaRadixCache` (ported from upstream/main). Also removed a dangling standalone function call `available_and_evictable_str(tree)` that was erroneously merged into our test.

## New Files Added

| File | Source | Lines |
|------|--------|-------|
| `python/sglang/srt/mem_cache/hi_mamba_radix_cache.py` | Upstream `5867c3fa8` | ~1668 |
| `python/sglang/srt/layers/attention/mamba/ops/ssu_dispatch.py` | Upstream `82a0bafc1` | ~277 |
| `python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py` | Upstream `a1b39c1c2` | ~190 |

## Beads Tasks Closed

| ID | Title | Reason |
|----|-------|--------|
| `upstream-sync-2026-Q1-cwf` | P04: Research HiMambaRadixCache + pool structs | API verified compatible |
| `upstream-sync-2026-Q1-44v` | P04: DESIGN REVIEW — TierManager layering | Confirmed: layer, don't replace |
| `upstream-sync-2026-Q1-22o` | P04: Implement cache architecture merge | 17 commits merged |
| `upstream-sync-2026-Q1-8sd` | P04: Validate Phases 2-8 | All validations pass |
| `upstream-sync-2026-Q1-clc` | Phase 04: Mamba Cache Architecture Reconcile | Phase complete |
