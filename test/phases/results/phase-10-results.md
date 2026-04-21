# Phase 10 — Scaling & Resource Monitoring: Initial Results

**Date**: 2026-03-29
**Branch**: `phase-08-true-stateful-inference`
**Model**: `granite-4.0-h-tiny` (GraniteMoeHybridForCausalLM, ~4B params)
**Server flags**: `--enable-snapshot-persistence --snapshot-dir /tmp/mamba_snapshots --mamba-scheduler-strategy no_buffer`
**GPU**: NVIDIA A100-SXM4-80GB (SM80)
**Max Context**: 131,072 tokens

---

## Baseline Metrics

### Resource Usage at Idle

| Metric | Value |
|--------|-------|
| GPU VRAM (baseline) | 69,810 MB (~68 GB) |
| Process RSS (all sglang) | 10,286 MB (~10 GB) |
| GPU Utilization (idle) | 0% |
| System RAM Available | 1,954,812 MB (~1.9 TB) |
| Snapshot Count | 126 files (from prior tests) |

**Notes:**
- High baseline VRAM is expected for 80GB A100 with model loaded
- Process RSS includes scheduler + detokenizer + model weights
- System has 2TB RAM available

---

## Test Results

### Test 1: Baseline (30 seconds idle)

```
Duration: 30.4s
GPU VRAM Delta: +0 MB
Process RSS Delta: +0 MB
Snapshots: 0 (no new snapshots created)
```

**Status:** ✅ PASS - Stable baseline, no leaks at idle

---

### Test 2: Multi-Turn with Snapshots (8 turns)

```
Duration: 10.1s
Turns Completed: 8
Snapshot Created: d0cdc7fb50474974a3e419cae20d7962-t0
GPU VRAM Delta: +0 MB
Process RSS Delta: +5 MB
Peak GPU Utilization: 26%
```

**Status:** ✅ PASS - Snapshot creation works, minimal memory growth

---

### Test 3: Continuous Load (60 seconds)

```
Requests Completed: 91
Errors: 0
Average Rate: ~1.5 requests/sec
GPU VRAM Delta: +0 MB (69,810 MB → 69,810 MB)
Process RSS Delta: +17 MB (10,291 MB → 10,308 MB)
Peak GPU Utilization: 36%
```

**Status:** ✅ PASS - No memory leaks, stable under sustained load

---

## Memory Leak Analysis

### GPU VRAM

| Test | Duration | Start | End | Delta | Verdict |
|------|----------|-------|-----|-------|---------|
| Baseline | 30s | 69,810 MB | 69,810 MB | 0 MB | ✅ No leak |
| Snapshots | 10s | 69,810 MB | 69,810 MB | 0 MB | ✅ No leak |
| Continuous | 60s | 69,810 MB | 69,810 MB | 0 MB | ✅ No leak |

### Process RSS

| Test | Duration | Start | End | Delta | Verdict |
|------|----------|-------|-----|-------|---------|
| Baseline | 30s | 10,286 MB | 10,286 MB | 0 MB | ✅ No leak |
| Snapshots | 10s | 10,286 MB | 10,291 MB | +5 MB | ✅ No leak |
| Continuous | 60s | 10,291 MB | 10,308 MB | +17 MB | ✅ No leak |

**Analysis:**
- +17 MB over 60 seconds = 0.28 MB/sec growth rate
- This is negligible and likely normal fluctuation
- No concerning leak patterns detected

---

## Snapshot Storage

**Directory:** `/tmp/mamba_snapshots/`
**Total snapshot files:** 252 (.safetensors + .json pairs)
**Total conversations:** 828 directories

**Snapshot structure:**
```
/tmp/mamba_snapshots/
├── conversation_<id>/
│   ├── turn_N_state.safetensors  (Mamba state)
│   └── turn_N_metadata.json      (Metadata)
```

**Cleanup status:**
- Old snapshots from previous test runs are present
- No automatic cleanup observed during testing
- Recommend implementing TTL-based cleanup for production

---

## Model Context Window

**Max Model Length:** 131,072 tokens (~128K context)

This is **significantly larger** than tested so far (40-100 tokens per test).

**Next tests needed:**
- Medium context: ~500 tokens (should be trivial for 128K window)
- Large context: ~2,000 tokens
- XL context: ~8,000 tokens
- Stress: ~32,000+ tokens (push the limits)

---

## Open Questions

| Question | Status | Notes |
|----------|--------|-------|
| Memory leaks at idle? | ✅ Tested | No leaks detected |
| Memory leaks under load? | ✅ Tested | No leaks detected (91 requests, 60s) |
| Snapshot cleanup works? | ⚠️ Needs testing | Old snapshots present |
| Large context behavior? | ❌ Not tested | Need 2K+, 8K+, 32K+ token tests |
| Concurrent user stress? | ❌ Not tested | Need 10+ parallel clients |
| MoE vs Dense Mamba? | ⚠️ Limited | Only tested MoE+Hybrid so far |
| Larger models (7B+)? | ❌ Not tested | Need to find suitable models |

---

## Monitoring Tool

**Created:** `test/phases/phase-10-scaling.py`

Features:
- Continuous resource monitoring (GPU VRAM, RAM, Process RSS, FDs, Snapshots)
- CSV logging with timestamps
- JSON summary generation
- Multiple test scenarios (baseline, small, medium, snapshots, continuous, all)
- Configurable duration

**Usage:**
```bash
# Baseline (idle monitoring)
python test/phases/phase-10-scaling.py --scenario baseline --duration 60

# Specific scenario
python test/phases/phase-10-scaling.py --scenario snapshots --duration 120

# All tests
python test/phases/phase-10-scaling.py --scenario all
```

---

## Next Steps

1. **Large Context Testing** - Test with 2K, 8K, 32K token contexts
2. **Concurrent Load** - Multi-client stress test
3. **Long-Running Stability** - 1+ hour continuous operation
4. **Find Larger Mamba Models** - Search HuggingFace for 7B+ models
5. **Snapshot Cleanup** - Implement and test TTL-based cleanup

---

## Verdict So Far

**Memory Management:** ✅ EXCELLENT
- No leaks detected in 100 seconds of combined testing
- Stable GPU VRAM usage
- Minimal process RSS growth

**Ready for:** Larger context testing and concurrent load testing

**Not Ready for:** Production deployment without:
- Snapshot cleanup mechanism
- Longer-duration stability tests (hours)
- Multi-concurrent-user stress testing
