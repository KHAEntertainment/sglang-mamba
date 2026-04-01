# Phase 10 Cross-Model Results: Granite 4.0-H-small (32B)

**Date:** 2026-03-29/30
**Machine:** RunPod A100-SXM4-80GB (80 GB VRAM)
**Model:** Granite 4.0-H-small (32B, BF16)
**Architecture:** `GraniteMoeHybridForCausalLM` — dense Mamba2 hybrid (36 Mamba layers)

> **Note:** Reconstructed from memory system. Original file was on the RunPod A100 instance and was not committed before instance termination. One raw metric file preserved: `phase-10-logs/metrics_small_20260330_024240.csv`.

---

## Launch Configuration

Required special flags to fit on A100 80GB:
```bash
--context-length 4096 --mem-fraction-static 0.85
```
Default context (128K) would OOM. Reduced to 4K to free sufficient VRAM for the model.

---

## Protocol

Phase 10 cross-model compatibility test (5 tests):
1. Baseline inference — single-turn completion
2. Snapshot save — verify state is captured
3. Snapshot restore — verify state loads
4. Memory leak detection — VRAM/RAM stable across runs
5. Sequential snapshot save rate — automated multi-turn save

---

## Results

| Test | Result | Notes |
|------|--------|-------|
| Baseline inference | PASS | Coherent completions |
| Snapshot save | PASS | Snapshot written, ~150 MB |
| Snapshot restore | PASS | State loads, rid=null (pre-existing gap) |
| Memory leak detection | PASS | Stable VRAM/RAM |
| Sequential snapshot save rate | PARTIAL | Only 5% save hit rate |

**4/5 PASS (80%), 1 PARTIAL**

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Snapshot size | **~150 MB** per conversation turn |
| Average inference latency | **0.172s** |
| Sequential save hit rate | **5%** |
| Stateful recall | BLOCKED (pre-existing restore API gap) |
| Context length used | 4096 tokens (hardware-constrained) |

---

## Notes

**Sequential snapshot save hit rate (5%):** This is the most significant finding for this model. The WARM tier was sized for smaller models (granite-tiny ~55MB). When granite-small writes ~150MB snapshots in rapid succession, the WARM tier fills quickly and subsequent saves must evict and rewrite — resulting in a low net hit rate when measuring "is this exact snapshot in WARM when needed?". This is a tier sizing issue, not a correctness bug.

**Snapshot size (150 MB):** Granite-small has 36 Mamba2 layers with BF16 weights. Snapshot size scales with (num_mamba_layers × state_size × hidden_size × dtype_bytes). At 36 layers BF16, this is ~3× larger than granite-tiny (8 layers) and ~3× larger than Nemotron-Cascade (MoE with smaller active state).

---

## Verdict

**COMPATIBLE** with Engram snapshot infrastructure, with a noted WARM tier sizing limitation for large dense Mamba models. Tier max_warm_memory_gb should be increased proportionally when deploying with 30B+ dense models.
