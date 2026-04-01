# PHASE 06 — Server and HTTP Drift

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

Merge lower-risk `server_args.py` and `http_server.py` drift from upstream, then re-apply our snapshot REST endpoints and CLI flags on top of the updated substrate.

## Upstream Commits to Integrate

| Commit | Risk | Topic |
|--------|------|-------|
| `198381d9c` | MEDIUM | TLS / grpc TLS flags and HTTP startup plumbing |
| `cc451671b` | MEDIUM | Anthropic-compatible API endpoints |
| `581bf53e0` | MEDIUM | Whisper transcription endpoint and flags |
| `ced69c9f8` | LOW-MEDIUM | Whisper CUDA graph and timestamp options |
| `f0458e0b4` | MEDIUM | Moves network/socket utilities from `common.py` to `network.py` |
| `135af6dc9` | MEDIUM | Video/audio input fields in `io_struct.py` and `server_args.py` |
| `d5307ce02` | MEDIUM | Adds metadata to `UpdateWeightFromDiskReqInput` |
| `fc7f9c1de` | LOW-MEDIUM | CLI flag rename in `server_args.py` |
| `2e65c27b2` | MEDIUM | Flush-cache timeout and scheduler API plumbing |
| `8a4cdcd53` | MEDIUM | Reworks flush-cache API behavior across request structs and scheduler dispatch |

## Files Touched

- `python/sglang/srt/server_args.py`
- `python/sglang/srt/entrypoints/http_server.py`
- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/managers/scheduler.py` (flush-cache dispatch)
- `python/sglang/srt/common.py` → `python/sglang/srt/utils/network.py` (import churn)

## Decision Points

### 1. Adopt upstream additions directly
**Decision:** Accept TLS, Whisper, Anthropic routes, video/audio fields, and flush-cache timeout. These do not overlap our snapshot symbols directly.

### 2. Re-apply snapshot REST routes after upstream merge
**Decision:** Our snapshot endpoints must be re-added to the updated `http_server.py`:
- `POST /save_snapshot`
- `POST /list_snapshots`
- `POST /get_snapshot_info`
- `POST /restore_snapshot`
- `POST /delete_snapshot`

Place them in a logically grouped router section, distinct from the new Anthropic/Whisper routes.

### 3. Re-apply snapshot CLI flags after upstream merge
**Decision:** Re-add our flags to the updated `server_args.py`:
- `--enable-snapshot-persistence`
- `--snapshot-dir`
- `--snapshot-retention-count`
- `--snapshot-trigger-policy`
- `--enable-memory-tiers`
- `--snapshot-auto-restore`
- (and any others from `server_args.py` diff)

### 4. Handle network-utils move (`f0458e0b4`)
**Decision:** If our custom server bootstrap code references utilities that moved from `common.py` to `network.py`, update the imports. This is mechanical import churn.

### 5. `io_struct.py` coexistence
**Decision:** Upstream has expanded `io_struct.py` with video/audio fields and timing structs. Our snapshot I/O structs (`SaveSnapshotReqInput`, `RestoreSnapshotReqInput`, etc.) should be appended or placed in a dedicated section without interfering with upstream models.

## Execution Steps

1. Merge the upstream commits into the worktree. Expect merge noise in `server_args.py` and `http_server.py` due to high commit count.
2. Resolve conflicts by keeping upstream additions and then re-adding our snapshot routes/flags.
3. Update any imports affected by `f0458e0b4`.
4. Ensure `TokenizerManager` snapshot forwarding methods still route correctly.
5. Start the server and run:
   - Phase 1 (stateless inference)
   - Phase 4 (live server no_buffer)
   - Phase 7 (snapshot E2E)
   - Phase 8 (true stateful inference)

## Validation Criteria

- Phase 1 (stateless inference) **PASS**
- Phase 4 (live server no_buffer) **PASS**
- Phase 7 (snapshot E2E) **PASS** (6/6)
- Phase 8 (true stateful inference) **PASS**
- Snapshot REST endpoints respond correctly to health probes.

## Rollback Plan

- Reset worktree to `phase-05-pass` tag.

## Estimated Complexity

**MEDIUM** — 4 to 6 hours. Mostly merge-noise resolution and mechanical re-application.

## Dependencies

- `PHASE_05_SCHEDULER_IDLE_AND_POOL_FIXES` complete and tagged `phase-05-pass`.

## Team Structure

**Solo agent**.

## bd Workflow

```bash
bd ready --json                    # Confirm Phase 06 is unblocked
bd update <phase-06-id> --claim    # Claim before starting
# ... do the work ...
bd close <phase-06-id> --reason "Phase 06 PASS. Tagged phase-06-pass."
```
