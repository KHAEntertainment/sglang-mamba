# Model Compatibility Protocol — Standardized Agent Instructions

## Purpose

This document is a self-contained agent prompt for validating a new model against the Engram snapshot infrastructure. It produces consistent, comparable results across all models tested.

**When to use:** Any time a new model is being tested for Engram compatibility. The agent picks up this document, follows the protocol, and writes a standardized result file.

**Relationship to full phase suite:** The 9-phase test plan (phase-00 through phase-08) is the full integration ladder for deep validation of a single model. This protocol is a focused compatibility check that covers the critical gates. Think of it as a structured subset — rigorous enough for the model matrix, fast enough to run in a single session.

---

## Phase Coverage Map

This protocol maps to the full phase suite as follows:

| Protocol Step | Phases Covered | What It Validates |
|---------------|----------------|-------------------|
| Step 0: Environment | Phase 0 | CUDA, imports, GPU memory, model checkpoint |
| Step 1: Server Boot | Phase 1 | Server startup, HybridReqToTokenPool, model loading |
| Step 2: Unit Tests | Phases 2, 3, 5 | MambaPool, RadixCache components, Mamba2Metadata |
| Step 3: Server Integration | Phase 4 | Live radix cache, prefix sharing, multi-turn |
| Step 4: Snapshot System | Phase 7 | Snapshot save/restore, tier management |
| Step 5: Stateful Recall | Phase 8 (partial) | Semantic correctness after restore |
| Step 6: Stress | Phase 8 (partial) | Concurrent load, eviction under pressure |

**Not covered:** Phase 6 (extra_buffer strategy). Run the full phase suite if extra_buffer behavior matters for a specific model.

---

## Before You Start

### Required information

The operator (or calling agent) must provide:

| Field | Example | Required |
|-------|---------|----------|
| `MODEL_ID` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` | Yes |
| `MODEL_LOCAL_PATH` | `/home/jeanclawdai/models/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` | Yes |
| `MODEL_SHORT_NAME` | `nemotron-3-super-120b-fp8` | Yes |
| `ARCHITECTURE` | `NemotronHForCausalLM` | Discover during Step 1 |
| `EXTRA_SERVER_FLAGS` | `--tp 2` or env vars like `SGLANG_ENABLE_JIT_DEEPGEMM=0` | If needed |
| `LINEAR_ISSUE` | `KHA-203` | Optional |

### Download the model first

```bash
# If using HuggingFace
huggingface-cli download $MODEL_ID --local-dir $MODEL_LOCAL_PATH

# Verify
ls $MODEL_LOCAL_PATH/config.json
```

---

## Step 0: Environment Verification

**Maps to:** Phase 0
**Time estimate:** 2 minutes
**Server required:** No

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# 1. Verify installation
pip install -e python/ --quiet
python -c "import sglang; print('sglang version:', sglang.__version__)"

# 2. CUDA health
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_mem / 1024**3, 1), 'GB')
print('Free:', round(torch.cuda.mem_get_info()[0] / 1024**3, 1), 'GB')
"

# 3. Model checkpoint
ls $MODEL_LOCAL_PATH/config.json
python -c "
import json
with open('$MODEL_LOCAL_PATH/config.json') as f:
    cfg = json.load(f)
print('Architecture:', cfg.get('architectures', ['unknown']))
print('Model type:', cfg.get('model_type', 'unknown'))
print('Num layers:', cfg.get('num_hidden_layers', 'unknown'))
"
```

**Record:**
- sglang version
- GPU name and VRAM
- Architecture class from config.json
- Number of layers

**Pass criteria:** All commands succeed, CUDA available, config.json readable.

---

## Step 1: Server Boot & Baseline Inference

**Maps to:** Phase 1
**Time estimate:** 5–15 minutes (depends on model size)
**Server required:** Yes

### Launch server

```bash
# Set any model-specific env vars here
# Example: export SGLANG_ENABLE_JIT_DEEPGEMM=0

python -m sglang.launch_server \
    --model-path $MODEL_LOCAL_PATH \
    --port 30000 \
    --trust-remote-code \
    --mamba-scheduler-strategy no_buffer \
    $EXTRA_SERVER_FLAGS \
    > /tmp/compat_server.log 2>&1 &

SERVER_PID=$!

# Wait for ready (up to 5 minutes for large models)
for i in $(seq 1 150); do
    curl -s http://localhost:30000/health > /dev/null 2>&1 && break
    sleep 2
done
```

### Verify server state

```bash
# 1. Model endpoint
curl -s http://localhost:30000/v1/models | python -m json.tool

# 2. Check for Mamba cache allocation in logs
grep -i "mamba\|ssm_state\|conv_state\|HybridReqToTokenPool" /tmp/compat_server.log | head -20

# 3. Completions endpoint
curl -s http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "prompt": "The capital of France is", "max_tokens": 32, "temperature": 0}' \
  | python -m json.tool

# 4. Chat completions endpoint
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}], "max_tokens": 32, "temperature": 0}' \
  | python -m json.tool
```

**Record:**
- Server startup time (weight load + CUDA graph)
- VRAM at steady state (from `nvidia-smi`)
- Mamba/SSM cache sizes from logs (conv_state, ssm_state, max_mamba_cache_size)
- KV cache token count and size
- Whether Mamba cache was allocated (critical — if not, snapshot infra won't engage)
- Completions output (coherent yes/no)
- Chat completions output (coherent yes/no)
- Any errors or warnings during startup
- Any model-specific env vars or flags required

**Pass criteria:** Server starts, both endpoints return coherent output, Mamba cache is allocated.

**STOP GATE:** If Mamba cache is NOT allocated, the model does not use recurrent state layers and is not a candidate for Engram. Record as "Not applicable — no recurrent state" and stop.

---

## Step 2: Unit Tests (No Server)

**Maps to:** Phases 2, 3, 5
**Time estimate:** 2 minutes
**Server required:** No (kill server from Step 1 first to free VRAM, or run in separate terminal)

```bash
cd "$REPO_ROOT"

# MambaPool unit tests (Phase 2)
python -m pytest test/registered/radix_cache/test_mamba_pool_extended.py -v 2>&1 | tee /tmp/compat_phase2.log

# RadixCache component tests (Phase 3)
python -m pytest test/registered/radix_cache/test_mamba_radix_cache_gauntlet.py -v 2>&1 | tee /tmp/compat_phase3.log

# Mamba2Metadata integrity (Phase 5)
python -m pytest test/registered/radix_cache/test_mamba_metadata.py -v 2>&1 | tee /tmp/compat_phase5.log

# Existing unit tests (Phase 0 baseline)
python -m pytest test/registered/radix_cache/test_mamba_unittest.py -v 2>&1 | tee /tmp/compat_phase0_unit.log
```

**Record:** Per-suite pass/fail/skip counts.

**Pass criteria:** All suites pass. These are model-independent (no server), so failures here indicate infrastructure issues, not model incompatibility.

---

## Step 3: Server Integration Tests

**Maps to:** Phase 4
**Time estimate:** 5–10 minutes
**Server required:** Yes (restart with radix cache active)

### Restart server with cache reporting

```bash
# Kill any running server
kill $SERVER_PID 2>/dev/null; sleep 3

python -m sglang.launch_server \
    --model-path $MODEL_LOCAL_PATH \
    --port 30000 \
    --trust-remote-code \
    --mamba-scheduler-strategy no_buffer \
    --enable-cache-report \
    $EXTRA_SERVER_FLAGS \
    > /tmp/compat_server_integration.log 2>&1 &

SERVER_PID=$!

# Wait for ready
for i in $(seq 1 150); do
    curl -s http://localhost:30000/health > /dev/null 2>&1 && break
    sleep 2
done
```

### Run integration tests

```bash
cd "$REPO_ROOT"

# RadixCache server integration (Phase 4)
python -m pytest test/registered/radix_cache/test_mamba_radix_cache_server_integration.py -v 2>&1 | tee /tmp/compat_phase4.log

# Baseline inference (Phase 1 tests)
python -m pytest test/registered/radix_cache/test_mamba_baseline_inference.py -v 2>&1 | tee /tmp/compat_phase1.log

# RadixCache comprehensive
python -m pytest test/registered/radix_cache/test_mamba_radix_cache_comprehensive.py -v 2>&1 | tee /tmp/compat_phase0_comp.log
```

**Record:** Per-suite pass/fail/skip counts. Note any model-specific failures with tracebacks.

**Pass criteria:** Core integration tests pass. Model-specific behavioral failures (e.g., reasoning model output format, temperature defaults) should be noted but don't block — they indicate test harness assumptions, not infrastructure bugs.

---

## Step 4: Snapshot System

**Maps to:** Phase 7
**Time estimate:** 5–10 minutes
**Server required:** Yes (same server from Step 3, or restart with snapshots enabled)

```bash
cd "$REPO_ROOT"

# Snapshot unit tests
python -m pytest test/registered/radix_cache/test_mamba_snapshot.py -v 2>&1 | tee /tmp/compat_phase7_unit.log

# Snapshot end-to-end
python -m pytest test/registered/radix_cache/test_mamba_snapshot_e2e.py -v 2>&1 | tee /tmp/compat_phase7_e2e.log
```

**Record:** Per-suite pass/fail/skip. Note specifically:
- Whether snapshot save succeeds
- Whether snapshot restore succeeds
- Whether restored state produces output (restore API gap — known pre-existing issue)
- Snapshot file size on disk

**Pass criteria:** Snapshot save/restore mechanics work. The restore-then-generate API gap is a known pre-existing issue across all models — don't count it as a model-specific failure.

---

## Step 5: Stateful Recall Verification

**Maps to:** Phase 8 (true stateful inference)
**Time estimate:** 5 minutes
**Server required:** Yes

This is the test the ad-hoc runs missed. It verifies that restoring a snapshot produces **semantically correct** continuation — not just that the API call succeeds.

```bash
# 1. Establish a fact in conversation
RESPONSE1=$(curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "My name is Zephyr and I am a lighthouse keeper on a remote island called Thornwick. Remember this."},
      {"role": "assistant", "content": "Got it — you are Zephyr, a lighthouse keeper on Thornwick island."},
      {"role": "user", "content": "What supplies should I order for the coming storm season?"}
    ],
    "max_tokens": 256,
    "temperature": 0
  }')

echo "=== Turn 1 response ==="
echo "$RESPONSE1" | python -m json.tool

# 2. Save snapshot
SAVE_RESULT=$(curl -s http://localhost:30000/save_snapshot \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "stateful-recall-test"}')

echo "=== Snapshot save ==="
echo "$SAVE_RESULT" | python -m json.tool

# 3. Restore snapshot and ask a recall question
RESTORE_RESULT=$(curl -s http://localhost:30000/restore_snapshot \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "stateful-recall-test"}')

echo "=== Snapshot restore ==="
echo "$RESTORE_RESULT" | python -m json.tool

# 4. Ask recall question (requires restored state to answer correctly)
RESPONSE2=$(curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "What is my name and where do I live?"}
    ],
    "max_tokens": 128,
    "temperature": 0
  }')

echo "=== Recall response ==="
echo "$RESPONSE2" | python -m json.tool
```

**Evaluation criteria:**
- **PASS**: Response mentions "Zephyr" AND "Thornwick" (or "lighthouse"/"island")
- **PARTIAL**: Response is coherent but doesn't recall the specific facts (state not fully restored)
- **FAIL**: Error, empty response, or nonsensical output
- **BLOCKED**: Restore API returns error (known pre-existing gap — record but don't count as model failure)

**Record:** The recall response verbatim, and your PASS/PARTIAL/FAIL/BLOCKED assessment.

---

## Step 6: Stress & Gauntlet

**Maps to:** Phase 8 (stress portion)
**Time estimate:** 5 minutes
**Server required:** Yes

```bash
cd "$REPO_ROOT"

# Gauntlet stress tests
python -m pytest test/registered/radix_cache/test_mamba_gauntlet_stress.py -v 2>&1 | tee /tmp/compat_phase8.log
```

**Record:** Pass/fail/skip counts. Note VRAM after stress run (`nvidia-smi`).

**Pass criteria:** Gauntlet passes, no CUDA OOM, no state contamination.

---

## After All Steps: Shut Down

```bash
kill $SERVER_PID 2>/dev/null
```

---

## Result File Format

Save results to `test/phases/results/compat-<model_short_name>-<date>.md` using this exact template:

```markdown
# Model Compatibility Report: <MODEL_SHORT_NAME>

**Date:** <YYYY-MM-DD>
**Machine:** <GPU name> (<VRAM> GB)
**Model:** <MODEL_ID>
**Architecture:** <architecture class from config.json>
**Format:** <BF16/FP8/GGUF/etc>
**Linear Issue:** <KHA-NNN or N/A>

## Environment
- sglang version: <version>
- GPU: <name>, <VRAM> GB total, <free> GB free at start
- CUDA: <version>
- Special flags: <any env vars or extra server flags, or "None">

## Server Boot
- Startup time: <weight load>s + <CUDA graph>s = <total>s
- VRAM at steady state: <used> GB / <total> GB
- Mamba cache allocated: Yes/No
  - conv_state: <size> GB
  - ssm_state: <size> GB
  - max_mamba_cache_size: <N>
- KV cache: <tokens> tokens, <size> GB

## Bugs Found
<List any bugs discovered and fixed, with commit hashes. Or "None">

## Test Results

| Step | Suite | Passed | Failed | Skipped | Notes |
|------|-------|--------|--------|---------|-------|
| 0 | Environment | — | — | — | <PASS/FAIL> |
| 1 | Server boot + baseline | 4/4 | 0 | 0 | |
| 2 | test_mamba_pool_extended | /5 | | | |
| 2 | test_mamba_metadata | /5 | | | |
| 2 | test_mamba_unittest | /4 | | | |
| 3 | test_mamba_radix_cache_server_integration | /5 | | | |
| 3 | test_mamba_baseline_inference | /7 | | | |
| 3 | test_mamba_radix_cache_comprehensive | /9 | | | |
| 3 | test_mamba_radix_cache_gauntlet | /6 | | | |
| 4 | test_mamba_snapshot | /21 | | | |
| 4 | test_mamba_snapshot_e2e | /6 | | | |
| 5 | Stateful recall | — | — | — | <PASS/PARTIAL/FAIL/BLOCKED> |
| 6 | test_mamba_gauntlet_stress | /6 | | | |

## Totals

| Category | Count |
|----------|-------|
| Total tests run | |
| PASS | |
| FAIL (model-specific) | |
| FAIL (infra/path) | |
| FAIL (pre-existing) | |
| SKIP | |

**Effective pass rate:** <X/Y> = <percent>%

## Stateful Recall Assessment

<Verbatim recall response and PASS/PARTIAL/FAIL/BLOCKED assessment>

## Model-Specific Notes

<Architecture observations, FP8 behavior, recurrent state type, any operational notes (e.g., DeepGEMM flag), comparison with other models>

## Verdict

<MODEL_SHORT_NAME> is [COMPATIBLE / PARTIALLY COMPATIBLE / INCOMPATIBLE / BLOCKED] with Engram snapshot infrastructure.
```

---

## Adding Results to the Model Matrix

After completing this protocol, add the model to `test/phases/MODEL_MATRIX.md` following the format documented there.
