# PHASE 04 — Mamba Cache Architecture Reconcile

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

Reconcile upstream's new Mamba cache hierarchy (`HiMambaRadixCache`, `HybridCacheController`, host pools) with our snapshot/tier system so that both can coexist without breaking cache invariants or restore semantics.

## Upstream Commits to Integrate

| Commit | Risk | Topic |
|--------|------|-------|
| `5867c3fa8` | HIGH | Adds `HiMambaRadixCache` and HiCache support for MambaRadixCache |
| `0986bed8e` | HIGH | Mamba state offloading and `HybridCacheController`; introduces host-pool offload path |
| `197f80713` | HIGH | Refactors upstream `mamba_radix_cache.py` insert/release semantics |
| `dfd0a77a9` | HIGH | Fixes prefix insertion semantics in `hi_mamba_radix_cache.py` |
| `07ef5f7be` | HIGH | Removes sync points and changes Mamba cache / prefill cudagraph plumbing |
| `e4b708d3e` | HIGH | Spec V2 support for Mamba hybrid attention; changes pool sizing/init |
| `0ac6c63ae` | MEDIUM | Refactors Mamba ratio calculation during pool initialization; direct overlap with hybrid memory sizing logic |

## Files Touched

- `python/sglang/srt/mem_cache/mamba_radix_cache.py`
- `python/sglang/srt/mem_cache/hi_mamba_radix_cache.py`
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/mem_cache/memory_pool_host.py`
- `python/sglang/srt/mem_cache/hi_cache_controller.py` (new upstream file)
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/snapshot/mamba_snapshot.py`
- `python/sglang/srt/snapshot/tier_manager.py`

## Decision Points

### 1. Adopt upstream's `mamba_radix_cache.py` and `hi_mamba_radix_cache.py`
**Decision:** Replace our fork's `mamba_radix_cache.py` entirely with upstream's version.

Our fork previously treated this file as fork-owned (tombstones, dual LRU, COW). However, our **snapshot product does not depend** on those internal radix-cache semantics. It depends on `MambaPool` extract/inject and the snapshot package. Upstream's `MambaRadixCache` + `HiMambaRadixCache` are now their actively maintained cache-tree implementations.

### 2. TierManager vs HybridCacheController — DESIGN REVIEW REQUIRED
**Conflict:** Both systems manage Mamba state across memory tiers.

- **Upstream `HybridCacheController` + `MambaPoolHost`**: Runtime GPU↔host (L1↔L2) offload for capacity expansion and prefix sharing.
- **Our `TierManager` + `MambaHostPool`**: Durable host↔disk (L2↔L3) snapshots, cross-session references, retention policies, and startup warm restore.

**Decision:** **Keep both and layer.**
- Upstream's system owns GPU↔host movement.
- Our `TierManager` operates at a higher level, treating upstream's host cache as just another source/sink of Mamba state tensors, but primarily managing disk persistence and conversation lifecycle.
- `TierManager` should interact with `MambaPool` directly for extract/inject, not try to replace `HybridCacheController`.

**Research Agent must verify:** That `MambaSnapshotManager.extract_state_from_pool` and `inject_state_to_pool` can still reach the correct `MambaPool` slots when `HybridLinearKVPool` and `HybridReqToTokenPool` are initialized by upstream's new constructors.

### 3. Sync point removal (`07ef5f7be`)
**Decision:** Adopt upstream's sync point removal for performance, but verify that `maybe_save_mamba_snapshots()` (which runs post-forward) still sees fully synchronized Mamba state before extraction.

If upstream removes a sync that our snapshot hook implicitly relied on, we may need to add an explicit `torch.cuda.synchronize()` guard around the extract path.

### 4. Pool sizing/init changes (`e4b708d3e`, `0ac6c63ae`)
**Decision:** Adopt upstream's sizing logic. Validate that `MambaPool` allocation still matches what `MambaSnapshotManager` expects (tensor shapes for `conv` and `temporal` states).

## Execution Steps

1. **Research Agent** uses DeepWiki to confirm:
   - `HiMambaRadixCache` class hierarchy and initialization signature
   - `HybridCacheController` API (write, load, prefetch)
   - `MambaPoolHost` buffer layout vs our `MambaHostPool`
   - `HybridLinearKVPool` and `HybridReqToTokenPool` constructor changes in Spec V2

2. **Implementation Agent** merges the upstream commits, letting upstream files win for `mamba_radix_cache.py` and `hi_mamba_radix_cache.py`.

3. Resolve any `memory_pool.py` merge conflicts where our `MambaPool` / `HybridReqToTokenPool` customizations overlap with upstream's host-pool and Spec V2 changes.

4. **Design Review Checkpoint:** Orchestrator confirms the TierManager layering strategy before Implementation Agent adapts `tier_manager.py` to the new pool structures.

5. Add any required `torch.cuda.synchronize()` in `MambaSnapshotManager.extract_state_from_pool` if upstream sync removal creates a race.

6. **COMPLEMENT commit sweep:** After the cache architecture is stable, cherry-pick the low-risk Mamba COMPLEMENT commits from the sync report that don't have an explicit phase home. These are performance and correctness improvements that don't conflict with our snapshot features:
   - `87549f8f0` — Mamba convolution perf (avoids unnecessary `.contiguous()` copy)
   - `c76251f70` — Returns intermediate Mamba states from `ssd_combined.py`
   - `a1b39c1c2` — Fuses Mamba state scatter in MTP verify
   - `82a0bafc1` — Selective state update kernel call in Mamba ops
   - `25bd83033` — Piecewise CUDA graph for NemotronH hybrid models
   - `cfead25bb` — Qwen3.5 Mamba slicing fix for differing prefill/decode TP sizes
   - `dd82678b2` — Mamba cache transfer support for NPU
   - `69158e9d9` — Skips `_mamba_verify_update` for idle batches
   - `86c561778` — Fixes illegal memory access in Mamba SSM tracking during EAGLE verification
   - `0949b138a` — SSU dispatch startup logging cleanup

   Apply each and run a quick Phase 2 + Phase 3 sanity check after the batch.

7. Validation Agent runs the full Mamba cache test suite:
   - Phase 2 (MambaPool unit tests)
   - Phase 3 (MambaRadixCache gauntlet)
   - Phase 4 (live server no_buffer)
   - Phase 5 (metadata integrity)
   - Phase 6 (extra_buffer strategy)
   - Phase 7 (snapshot E2E)
   - Phase 8 (true stateful inference)

## Validation Criteria

- Phase 2 (MambaPool unit tests) **PASS** (5/5)
- Phase 3 (MambaRadixCache gauntlet) **PASS** (16/16)
- Phase 4 (live server no_buffer) **PASS**
- Phase 5 (Mamba2Metadata integrity) **PASS** (5/5)
- Phase 6 (extra_buffer strategy) **PASS**
- Phase 7 (snapshot E2E) **PASS** (6/6)
- Phase 8 (true stateful inference) **PASS**

## Rollback Plan

- Reset worktree to `phase-03-pass` tag.

## Estimated Complexity

**HIGH** — 10 to 20 hours. This is the widest architectural surface area.

## Dependencies

- `PHASE_03_SESSION_CONTROLLER_PORT` complete and tagged `phase-03-pass`.

## Team Structure

**4-agent team required:**
1. **Orchestrator (Opus)** — Design decisions at cache overlap points, coordinates agents.
2. **Research Agent (Sonnet)** — DeepWiki verification of `HiMambaRadixCache`, `HybridCacheController`, and pool struct layouts.
3. **Implementation Agent (Sonnet)** — Executes merge and adapts `tier_manager.py` / `mamba_snapshot.py`.
4. **Validation Agent (Sonnet)** — Runs Phases 2–8 and documents any regressions.

## bd Workflow

Phase 04 has 4 sub-tasks in bd with dependencies:

1. **Research** (unblocked after Phase 03 validation) → `bd update <id> --claim` → `bd close <id>`
2. **DESIGN REVIEW** (unblocked after Research) → Orchestrator closes with `--reason "TierManager layering confirmed..."`
3. **Implementation + COMPLEMENT sweep** (unblocked after design review) → `bd update <id> --claim` → `bd close <id>`
4. **Validation** (unblocked after implementation) → `bd close <id> --reason "Phases 2-8 PASS"`

Close the parent Phase 04 task after all sub-tasks pass. Tag `phase-04-pass`.
