# Phase 10f Resilience Test Results

**Date:** 2026-03-30
**Machine:** RunPod A100-SXM4-80GB (80 GB VRAM)
**Model:** `granite-4.0-h-tiny` (4B, BF16)
**Phase:** 10f — Crash recovery, atomic writes, startup preload

> **Note:** Reconstructed from memory system. Original file was on the RunPod A100 instance and was not committed before instance termination. The script used to run these tests (`phase-10-resilience.py`) is also not recoverable — see notes.

---

## Test Results

| Test | Result | Notes |
|------|--------|-------|
| Startup preload after SIGKILL | **PASS** | COLD snapshots auto-loaded to WARM; restore latency = 5ms |
| Atomic writes under SIGKILL | **PASS** | No `.tmp` files or partial safetensors left on disk |
| Client disconnect handling | **PASS** | Server recovers cleanly; no VRAM leak; pre-free hook skips abandoned requests |
| Abort + snapshot save | **PASS** | Snapshot saved from WARM tier before abort cleanup; state preserved |
| SIGTERM graceful shutdown | **FAIL** | Hangs >60s; worker processes survive SIGTERM |

**4/5 PASS, 1 FAIL**

---

## Key Findings

### 1. Startup Preload Confirmed (Gap 3 Fix Validated)

After a `SIGKILL` + server restart, the tier manager automatically loaded COLD snapshots back to the WARM tier during startup. The restore latency measured 5ms — within the WARM tier range (2–5ms) — confirming that the preload completed before the first request was processed. This validates the Gap 3 fix from PR #6.

### 2. Atomic Writes Verified

After issuing `SIGKILL` mid-write during a snapshot save operation, no orphaned `.tmp` files or partial `.safetensors` files remained on disk. The write-to-temp-then-rename pattern is functioning correctly.

### 3. Client Disconnect Recovery

When a client disconnected mid-generation, the server recovered without VRAM leak. The pre-free hook correctly identified and skipped the abandoned request during cleanup. Subsequent requests processed normally.

### 4. Abort + Save Interaction

Saving a snapshot for a request that was subsequently aborted succeeded — the snapshot captured state from the WARM tier before the abort cleanup ran. This means an aborted conversation can be restored to the point before the abort occurred.

### 5. SIGTERM Hangs (Bug #16)

Graceful shutdown via `SIGTERM` did not complete within 60s. Worker processes (tokenizer, scheduler, model workers) survived the signal. The snapshot data on disk was unaffected. Workaround for production: send `SIGTERM` then `SIGKILL` after a timeout.

This is a scheduler drain issue (the worker pool drain loop doesn't terminate when no new requests arrive), not related to the snapshot infrastructure.

---

## Bug Filed

**Bug #16:** SIGTERM graceful shutdown hangs >60s.
- Root cause: Scheduler drain loop does not exit when queue is empty
- Snapshot impact: None — data is safe on disk
- Workaround: `SIGKILL` after timeout
- Priority: Low (snapshots survive, only affects clean shutdown UX)

---

## Script Note

The test script (`test/phases/scripts/phase-10-resilience.py`) was not committed from the A100 instance and cannot be recovered. The test methodology was: (1) start server, (2) establish conversation with snapshots, (3) inject fault condition, (4) measure recovery. Manual reproduction is possible using the curl commands in the phase 8 docs.
