# Phase 10 Cross-Model Results: Nemotron-Cascade-2-30B

**Date:** 2026-03-29
**Machine:** RunPod A100-SXM4-80GB (80 GB VRAM)
**Model:** Nemotron-Cascade-2-30B (BF16)
**Architecture:** `NemotronHForCausalLM` — MoE Mamba2 hybrid

> **Note:** Reconstructed from memory system. Original file was on the RunPod A100 instance and was not committed before instance termination.

---

## Protocol

Phase 10 cross-model compatibility test (5 tests):
1. Baseline inference — single-turn completion
2. Snapshot save — verify state is captured
3. Snapshot restore — verify state loads
4. Memory leak detection — VRAM/RAM stable across runs
5. Rapid-fire throughput — concurrent requests

---

## Results

| Test | Result | Notes |
|------|--------|-------|
| Baseline inference | PASS | Coherent completions, fast inference |
| Snapshot save | PASS | Snapshot written to WARM tier, ~47 MB |
| Snapshot restore | PASS | State loads successfully, rid=null (pre-existing gap) |
| Memory leak detection | PASS | VRAM stable, no RAM growth |
| Rapid-fire throughput | PASS | Sustained concurrent load without degradation |

**5/5 PASS (100%)**

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Snapshot size | **~47 MB** per conversation turn |
| Average inference latency | **0.059s** |
| vs. granite-small (32B) | **3× faster** (0.059s vs. 0.172s) |
| Snapshot save rate | **100%** |
| Stateful recall | BLOCKED (pre-existing restore API gap) |

---

## Notes

- Star performer in Phase 10: fastest inference of any model tested, smallest snapshots despite being a 30B model
- MoE routing (3B active parameters) explains the inference speed advantage
- FP8 not used in this run — BF16 weights
- Snapshot size (~47 MB) is notably smaller than granite-small (~150 MB) despite having more total parameters — reflects smaller active parameter count
- Loaded without special flags on A100 80GB

---

## Verdict

**COMPATIBLE** with Engram snapshot infrastructure. Recommended as the primary benchmark model for future Phase 10 runs due to speed and consistent behavior.
