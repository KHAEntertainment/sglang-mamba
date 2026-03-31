# Upstream SGLang Sync Report

Date: 2026-03-28
Branch: `upstream-sync-audit`
Fork checked against GitHub: local `main` is in sync with `origin/main` (`0/0` divergence)
Canonical upstream: `https://github.com/sgl-project/sglang.git`
Merge base: `91230dcca890252018e04f838db7f19995a62aa1`

## 1. Executive Summary

- Commits behind upstream: `1327`
- Commits ahead on our fork: `97`
- Upstream commits touching our six fork-modified files: `176`
- Mamba/SSM-related upstream commits reviewed: `25`
- Per-file churn in the direct-conflict surface:
  - `scheduler.py`: 61 commits
  - `io_struct.py`: 12 commits
  - `memory_pool.py`: 17 commits
  - `server_args.py`: 105 commits
  - `http_server.py`: 18 commits
  - `mamba2_metadata.py`: 0 commits
- Overall merge risk: `CRITICAL`

Upstream has moved materially toward our hybrid-Mamba cache surface, but not toward our actual product feature. They now own `mamba_radix_cache.py`, `hi_mamba_radix_cache.py`, `memory_pool_host.py`, and `session_aware_cache.py`, and they have added Mamba offloading, HiCache integration, session-aware cache paths, and multiple scheduler/session refactors. However, upstream still does not implement disk snapshots, request-level restore APIs, startup snapshot warm restore, or server-side Mamba state persistence. The merge risk is therefore not "feature duplication" so much as "architectural overlap on the same cache/scheduler seams."

## 2. Direct Conflict Table

This table focuses on the actionable non-low-risk commits. The remaining direct-touch commits are mostly additive `server_args.py` and `http_server.py` churn and should be treated as low-risk unless they collide with nearby local edits during the merge.

| Commit | File(s) | Type | Risk | Notes |
|--------|---------|------|------|-------|
| `0986bed8e` | `scheduler.py`, `memory_pool.py` | feature | HIGH | Adds Mamba state offloading and `HybridCacheController`; overlaps our Mamba pool, tiering, and scheduler cache lifecycle assumptions. |
| `5867c3fa8` | `scheduler.py` | feature/refactor | HIGH | Introduces HiCache support for `MambaRadixCache`; strong conceptual overlap with our VRAM/RAM/Disk tier design. |
| `197f80713` | upstream `mamba_radix_cache.py` | refactor | HIGH | Upstream now owns a `mamba_radix_cache.py` and refactors insert/release semantics; this invalidates the old assumption that the file is fork-owned only. |
| `dfd0a77a9` | upstream `hi_mamba_radix_cache.py` | bugfix | HIGH | Fixes prefix insertion semantics in upstream Mamba radix cache hierarchy; relevant because our cache semantics and tombstone behavior must be reconciled intentionally. |
| `07ef5f7be` | `memory_pool.py` | perf/refactor | HIGH | Removes sync points and changes Mamba cache / prefill cudagraph plumbing; conflicts with our `MambaPool` and hybrid pool assumptions. |
| `e4b708d3e` | `memory_pool.py` | feature | HIGH | Adds Spec V2 support for Mamba hybrid attention and changes Mamba pool sizing/init behavior. |
| `c6cb0c964` | `scheduler.py`, `io_struct.py`, `memory_pool.py` | feature | HIGH | Adds streaming sessions and `SessionAwareCache` fast path; overlaps our `session_params` and restore/session lifecycle surface. |
| `e08ef0675` | `scheduler.py`, `server_args.py` | feature | HIGH | Gates streaming sessions behind a server arg and adds Spec V2 guards; touches the same scheduler/server-arg coordination layer we extended. |
| `5acb45cf3` | `scheduler.py` | refactor | HIGH | Extracts `SessionController` out of `Scheduler`; high risk for our restore hooks, `create_new_request`, and session bookkeeping edits. |
| `8a4cdcd53` | `http_server.py`, `io_struct.py`, `scheduler.py` | refactor/bugfix | MEDIUM | Reworks flush-cache API behavior across request structs and scheduler dispatch; same files as our snapshot endpoints and request IO models. |
| `2e65c27b2` | `http_server.py`, `io_struct.py`, `scheduler.py` | feature | MEDIUM | Adds flush-cache timeout and grows scheduler API plumbing; medium risk because it touches the same request-routing surface, but not snapshot-specific symbols. |
| `3b8930227` | `http_server.py`, `io_struct.py`, `scheduler.py` | refactor | HIGH | Large observability cleanup that moves imports and reshapes request timing/tracing fields in all three files we extended. |
| `198381d9c` | `http_server.py`, `server_args.py` | feature | MEDIUM | Adds TLS/grpc TLS flags and HTTP startup plumbing; route/bootstrap overlap with our snapshot REST endpoint additions. |
| `cc451671b` | `http_server.py` | feature | MEDIUM | Adds Anthropic-compatible API endpoints; same router file, different route area. |
| `581bf53e0` | `http_server.py`, `server_args.py` | feature | MEDIUM | Adds Whisper transcription endpoint and flags; same server bootstrap and route registry surface as our snapshot endpoints. |
| `ced69c9f8` | `http_server.py`, `server_args.py` | feature | LOW-MEDIUM | Adds Whisper CUDA graph and timestamp options; nearby route/arg churn, unlikely to overlap our symbols directly. |
| `e1ee68d0f` | `scheduler.py` | feature | MEDIUM | Releases multimodal features on session close and expands rerun-ut behavior; session-close path may interact with our restore/save lifecycle. |
| `b1246c50f` | `memory_pool.py` | bugfix | MEDIUM | Fixes streaming-session KV leaks; same pool layer we extended for hybrid Mamba state, though not snapshot-specific. |
| `e2fccb2ee` | `scheduler.py` | bugfix | MEDIUM | Reverts Mamba slot release behavior; relevant to our restore/reinsert lifecycle assumptions. |
| `8541b1118` | upstream Mamba decode path | feature/bugfix | MEDIUM | Passes max Mamba cache size through disaggregation decode; indicates upstream pool sizing is actively moving. |
| `4d3976b6c` | `scheduler.py` | bugfix | MEDIUM | Changes `is_fully_idle()` to account for async HiCache ops; relevant because our snapshot tier movement also depends on scheduler idle assumptions. |
| `5270a0648` | `scheduler.py` | bugfix | MEDIUM | Fixes disaggregation false positives in idle detection; same idle-state path as our background snapshot/tier behavior. |
| `079a1fd35` | `scheduler.py` | bugfix | MEDIUM | Fixes write-through events when idle; relevant because upstream now has async cache plumbing in the same scheduler lifecycle slots we use. |
| `50953aea8` | `scheduler.py` | refactor/bugfix | MEDIUM | Unifies idle checks into `is_fully_idle()`; medium conflict risk with our save/restore/background hook assumptions. |
| `26f709e97` | `scheduler.py` | compatibility | LOW-MEDIUM | Makes prefill-delayer work with multiple mem-pool types; nearby pool abstraction churn. |
| `f0458e0b4` | `scheduler.py`, `server_args.py` | refactor | MEDIUM | Moves socket/network utilities and touches startup plumbing in both files we extended. |
| `745840743` | `scheduler.py`, `server_args.py` | bugfix | LOW-MEDIUM | Non-CUDA/VLM fix; same files, different feature area. |
| `135af6dc9` | `io_struct.py`, `server_args.py` | feature | MEDIUM | Adds video/audio input fields; expands request schemas in `io_struct.py`, which we also extended with restore/snapshot request models. |
| `d5307ce02` | `io_struct.py` | feature | MEDIUM | Adds metadata to `UpdateWeightFromDiskReqInput`; indicates continued schema growth in the request model file we extended. |
| `fc7f9c1de` | `server_args.py` | rename | LOW-MEDIUM | CLI flag rename in a file where we added multiple custom flags; low direct overlap, but merge noise is guaranteed. |

## 3. Mamba/SSM Intelligence

### Critical Answer

Upstream is not building disk-backed state persistence, snapshot APIs, or conversation restore for Mamba models. There are no upstream matches for `save_snapshot`, `restore_snapshot`, `SnapshotManager`, `--enable-snapshot-persistence`, or snapshot-specific server args in `upstream/main`. What upstream is building instead is a richer in-memory and host-memory hierarchy for hybrid Mamba cache management: HiCache, host-pool offload, session-aware cache paths, and Mamba-specific pool/radix maintenance.

### OVERLAP

`0986bed8e` adds Mamba state offloading and a `HybridCacheController`. This overlaps conceptually with our tiered VRAM/RAM/Disk architecture, but upstream stops at in-memory/host offload rather than durable disk snapshots.

`5867c3fa8` adds HiCache support for `MambaRadixCache` and introduces upstream `hi_mamba_radix_cache.py`. This is the strongest overlap signal because upstream is now formalizing hierarchical Mamba cache residency and scheduler/cache-controller coordination on the same seams we extended.

`197f80713` refactors upstream `mamba_radix_cache.py` and changes insert-time release behavior. This matters because our fork previously treated `mamba_radix_cache.py` as fork-owned; upstream now has a competing implementation on the same path and same abstraction name.

`dfd0a77a9` fixes `HiMambaRadixCache` prefix insertion with `prev_prefix_len`. This suggests upstream is still refining correctness in the exact cache tree logic area where our tombstone/COW semantics live.

`07ef5f7be` removes sync points and changes Mamba cache/prefill cudagraph plumbing. It does not implement persistence, but it does change performance-sensitive assumptions around when Mamba state is synchronized and released.

`e4b708d3e` adds Spec V2 support for Mamba hybrid attention. This overlaps with our hybrid Mamba support surface through memory pool sizing and decode/prefill bookkeeping, even though it is not about snapshots.

`0ac6c63ae` refactors how the additional Mamba ratio is calculated during pool initialization. This is direct overlap with the hybrid memory sizing logic we depend on.

`8541b1118` passes max Mamba cache size through the disaggregation decode path. This is another signal that upstream is still moving the control plane for Mamba memory sizing.

`e2fccb2ee` reverts Mamba slot release behavior to fix flaky tests. That overlaps with our restore/reinsert lifecycle because slot release timing is central to whether restored state remains valid.

### COMPLEMENT

`87549f8f0` improves Mamba convolution performance by avoiding an unnecessary `.contiguous()` copy. This complements our work and should be low-risk to adopt once the cache/scheduler surface is reconciled.

`c76251f70` returns intermediate Mamba states from `ssd_combined.py`. This complements our product direction because it exposes more internal state, but it still does not provide durable snapshotting or session continuity.

`a1b39c1c2` fuses Mamba state scatter in MTP verify. This is a pure performance complement.

`82a0bafc1` adds a selective state update kernel call in the Mamba ops layer. This is a low-conflict Mamba kernel enhancement.

`25bd83033` enables piecewise CUDA graph for NemotronH hybrid models. This broadens hybrid-Mamba support and should help model coverage rather than conflict with snapshots.

`cfead25bb` fixes Qwen3.5 Mamba slicing for differing prefill/decode TP sizes. This complements our hybrid support because it improves correctness for another Mamba-family model.

`dd82678b2` adds Mamba cache transfer support for NPU. This broadens device/backend support around the same cache abstractions.

`69158e9d9` skips `_mamba_verify_update` for idle batches. This is a correctness/perf complement inside speculative decoding.

`86c561778` fixes illegal memory access in Mamba SSM tracking during EAGLE verification. This is a pure bugfix complement.

`d1e95af28` upgrades `transformers` to `5.3.0`, including Granite-related model evolution. This complements model support, though it increases merge complexity through dependency drift.

### CONFLICT

`c6cb0c964` adds streaming sessions with `SessionAwareCache`. This is not a duplicate of our persistence feature, but it changes the exact session/cache lifecycle where our restore/save hooks and `create_new_request` logic operate.

`e08ef0675` gates streaming sessions with `--enable-streaming-session` and applies Spec V2 guards. This deepens the scheduler/server-arg split around sessions and raises merge risk for our snapshot flags.

`5acb45cf3` extracts `SessionController` from `Scheduler`. This is a structural conflict because our scheduler changes are concentrated in request routing, restore handling, and session-aware request creation.

`b1246c50f` fixes streaming-session KV cache leaks in `memory_pool.py`. This is not conceptually conflicting, but it touches the same hybrid pool infrastructure where we added Mamba pool/tier behavior.

### NEUTRAL

`7142a594f` adds CI coverage for `HiMambaRadixTree` and Qwen3.5. Useful evidence, but not product overlap.

`a4528a573` restores CI for Mamba PD disaggregation. This is CI-only.

`472607386` fixes a Mamba2 mixer CI test. This is test-only.

`1b79934d` fixes an AMD CI test involving a tool-choice model. The Mamba mention is incidental.

`0949b138a` touches `ssu_dispatch.py` during startup logging cleanup. The Mamba path is incidental and not strategically important.

## 4. Dangerous Refactors

- `5acb45cf3` `[Session] Extract SessionController and clean up session logic in Scheduler`
  - `Scheduler.sessions` is replaced by `SessionController`.
  - Our `create_new_request` restore flow and snapshot/session hook logic will need manual porting rather than a blind merge.

- `3b8930227` `Refactor: observability code cleanup`
  - Moves metrics/tracing modules into `srt/observability/`.
  - Touches `http_server.py`, `io_struct.py`, and `scheduler.py` simultaneously.
  - High merge-noise risk because it rewires imports and request timing fields in the same files we extended.

- `0986bed8e` `[HiCache][HybridModel]: Support mamba state offloading & HybridCacheController`
  - Introduces a new upstream cache-controller abstraction and host offload path.
  - This is an architectural refactor on the same problem boundary as our tier manager and host pool.

- `5867c3fa8` `Support HiCache for MambaRadixCache`
  - Adds `hi_mamba_radix_cache.py` and extends `mamba_radix_cache.py`.
  - Upstream now owns a large portion of the cache-tree surface that our fork originally invented privately.

- `f0458e0b4` `[Utils] Move network/socket utilities from common.py to network.py`
  - Moderate import churn in `scheduler.py` and `server_args.py`.
  - Not Mamba-specific, but likely to create merge conflicts adjacent to our custom route/arg changes.

- `654fc02cf` `[gRPC] Extract gRPC servicer into standalone package`
  - Removes multiple direct grpc-related deps from the main package and replaces them with `smg-grpc-servicer`.
  - Build and packaging behavior changed materially.

## 5. Dependency Changes

| Area | Fork / merge-base state | Upstream current | Evidence |
|------|-------------------------|------------------|----------|
| `flashinfer_python` / `flashinfer_cubin` | `0.6.3` | `0.6.6` | `682294151`, `93afe15b4` |
| `sgl-kernel` | `0.3.21` | `0.4.0` | `15097c5c3` |
| `transformers` | `4.57.1` | `5.3.0` | `d1e95af28` |
| `xgrammar` | `0.1.27` | `0.1.32` | `f289d173a` |
| `torch` / `torchaudio` | `2.9.1` | `2.9.1` | no effective bump in current upstream |
| `torchcodec` | `0.8.0` | `0.9.1` | `4a757990a` |
| video decode deps | unconditional `decord2` | ARM Linux uses `decord2` + `av`, non-ARM Linux/macOS use `torchcodec 0.9.1` | `4a757990a` |
| gRPC packaging | direct `smg-grpc-proto`, `grpcio`, reflection, health-checking deps | `smg-grpc-servicer>=0.5.0` | `654fc02cf` |
| diffusion optional dep | `cache-dit==1.2.0` | `cache-dit==1.3.0` | `025691cd9` |

## 6. Recommended Merge Strategy

Recommendation: `REVIEW-THEN-MERGE`

Rationale:

1. A straight rebase is too risky. The number of direct-touch commits is high, and upstream now owns overlapping Mamba cache abstractions on paths and concepts that our fork had treated as custom.
2. A pure cherry-pick strategy is also the wrong default. Too many important upstream changes are structural refactors or cumulative dependency moves rather than isolated bugfixes.
3. The real decision point is architectural: whether our snapshot/tier system should be rebased onto upstream's newer HiCache + session-controller world, or whether some of our current implementation should be replaced with upstream primitives before merging.

Suggested order:

1. Reconcile the dependency baseline first in an isolated branch.
   - `flashinfer 0.6.6`
   - `sgl-kernel 0.4.0`
   - `transformers 5.3.0`
   - grpc packaging change
2. Port our scheduler/session customizations onto upstream's session refactors.
   - `c6cb0c964`
   - `e08ef0675`
   - `5acb45cf3`
   - `3b8930227`
3. Reconcile the Mamba cache architecture before merging our snapshot code.
   - `5867c3fa8`
   - `0986bed8e`
   - `197f80713`
   - `07ef5f7be`
   - `e4b708d3e`
4. Merge the lower-risk `server_args.py` and `http_server.py` drift after the scheduler/cache layer is stable.
5. Re-apply and validate our snapshot-specific product feature last.
   - snapshot REST routes
   - `RestoreSnapshotReqInput` / `RestoreSnapshotReqOutput`
   - `handle_save_snapshot` / `handle_restore_snapshot`
   - startup warm restore
   - `fill_ids` restore sync

## Bottom Line

Upstream is moving toward us on cache hierarchy and hybrid-Mamba runtime support, but not on durable session continuity. The safest path is to treat upstream's new Mamba/HiCache/session work as the new substrate, then re-port our snapshot persistence feature on top of it after an explicit design review. Blindly merging the branch history would almost certainly damage either our restore semantics or upstream's newer cache/session invariants.
