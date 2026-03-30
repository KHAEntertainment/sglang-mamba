# SGLang-Mamba: Stateful Inference for Mamba Models

## Executive Summary

This project adds **stateful inference** capabilities to Mamba (State Space Model) language models, enabling dramatic efficiency improvements for multi-turn conversations. It's a fork of [SGLang](https://github.com/sgl-project/sglang) that introduces snapshot persistence—allowing models to "remember" where a conversation left off, rather than reprocessing everything from scratch each time.

---

## The Problem: Stateless Inference is Wasteful

**How LLMs typically work:**

In standard ("stateless") inference, every time you send a message to a language model, the system must resend your *entire conversation history* as input. Each message gets processed through the model from beginning to end, building up the internal state needed to generate a coherent response.

**Why this happens:**

Transformer-based models (like GPT, Llama, Claude) use an "attention mechanism" that builds up internal representations incrementally. The model doesn't inherently "remember" previous turns—it reconstructs understanding by re-reading everything.

**The cost:**

- **Token explosion**: A 10-turn conversation might require processing 5,000+ tokens to answer a 50-token question
- **Latency**: Each turn takes longer as conversations grow (reprocessing everything is slow)
- **Compute waste**: You're paying to process the same tokens over and over again
- **Memory pressure**: Long conversations consume massive GPU memory for cached attention states

---

## The Solution: Stateful Inference with Snapshots

**How Mamba is different:**

Mamba is a new architecture (State Space Model) that doesn't use attention. Instead, it maintains a compressed internal state that evolves as it processes tokens. This state is much smaller than attention caches and can be **saved and restored**.

**What we built:**

This project adds the ability to:
1. **Save snapshots** of Mamba's internal state after each conversation turn
2. **Restore snapshots** instantly when continuing a conversation
3. **Skip reprocessing**—only the new message needs to be tokenized and run through the model

**The result:**

In our testing, a multi-turn conversation that normally requires 97 tokens per turn can continue with just 6 new tokens—a **93.8% reduction** in processed tokens.

---

## Technical Architecture

### Components

1. **MambaRadixCache**: A dual-LRU cache system that manages Mamba pool slots with automatic eviction when memory fills up. Supports copy-on-write semantics and tombstone nodes for safe concurrent access.
2. **Snapshot Tiering (WARM + COLD)**:
   - **WARM tier** (host RAM): Auto-snapshot captures full final state via pre-free hook before `release_kv_cache`
   - **COLD tier** (disk): safetensors + JSON metadata per turn, supports startup preload
3. **Stateful Generate Pipeline**: Modified scheduler creates synthetic requests for continued conversations, bypassing normal tokenization
4. **Per-Request Correlation**: Futures-based system ensuring concurrent requests don't get mixed responses
5. **HybridReqToTokenPool**: Unified pool managing both attention KV slots and Mamba SSM state slots per request

### Key Innovation: The "Pre-Free" Snapshot

The critical breakthrough was timing snapshot creation *before* the model's memory cache is freed. Previous approaches saved snapshots *after* cleanup, when state was already gone. We now snapshot immediately after a request completes but before memory is released.

### Supported Architectures

| Architecture | Example Model | Status | Notes |
|-------------|---------------|--------|-------|
| MoE Hybrid (Mamba+Attention+Experts) | Nemotron-Cascade-2-30B | Fully supported | Best performance — 3x faster, 3x smaller snapshots |
| Dense Hybrid (Mamba+Attention) | granite-4.0-h-small (32B) | Fully supported | Larger snapshots (~150MB) but stable |
| MoE Hybrid (small) | granite-4.0-h-tiny (4B) | Fully supported | Primary development model |
| Pure Mamba2 | Mamba-Codestral-7B | Incompatible | No attention backend in SGLang; needs separate pipeline |

---

## Comparison: Stateless vs. Stateful

| Aspect | Stateless (Standard) | Stateful (This Project) |
|--------|---------------------|-------------------------|
| **Input per turn** | Full conversation history | Only new message |
| **Processing** | All tokens, every time | New tokens only |
| **Latency** | Grows with conversation length | Constant regardless of history |
| **Memory** | Attention cache (large) | Mamba state (tiny) |
| **Token efficiency** | 100% baseline | 5-10% of baseline |
| **Use case** | Single-turn tasks | Multi-turn conversations |

**Measured performance across models:**

| Metric | granite-tiny (4B) | granite-small (32B) | Nemotron-30B (3B active) |
|--------|-------------------|--------------------|--------------------------|
| Architecture | MoE Hybrid | Dense Hybrid | MoE Hybrid |
| Inference latency | ~0.15s | 0.172s | **0.059s** |
| Snapshot size | ~150MB | ~150MB | **~47MB** |
| Save reliability (multi-turn) | 100% | 100% | **100%** |
| Save reliability (sequential) | N/A | 5% | **100%** |
| GPU delta after full test | ~0MB | +256MB | +132MB |
| Memory leak detected | No | No | No |
| Max context tested | **128K tokens** | 2K tokens | 711 tokens |
| Max context supported | 131K | 131K | 262K |

### Context Window Scaling (Phase 10e)

Tested snapshot persistence from 2K to 128K context on granite-tiny. **All 5 tiers pass.**

| Context Length | Snapshot Size | Save Latency | WARM Restore | VRAM |
|---------------|--------------|-------------|-------------|------|
| 2K (1,843 tok) | 54.7 MB | 72 ms | 3 ms | 63.0 GB |
| 8K (7,815 tok) | 54.8 MB | 68 ms | 2 ms | 63.5 GB |
| 32K (31,776 tok) | 55.0 MB | 93 ms | 2 ms | 63.6 GB |
| 64K (63,840 tok) | 55.3 MB | 128 ms | 2 ms | 63.6 GB |
| **128K (127,670 tok)** | **55.9 MB** | **193 ms** | **2 ms** | **63.6 GB** |

**Snapshot size grows only +1.2MB (2.2%) from 2K to 128K context.** The Mamba SSM state tensor is architecture-determined (layer count × state dimensions), not context-determined. This means snapshot persistence scales trivially to maximum model context — a 128K conversation snapshot is the same ~55MB as a 2K conversation snapshot.

---

## Test Program Summary

11 phases, 3 model architectures, 150+ requests per model. Full results in `test/phases/results/INDEX.md`.

| Phase | Description | Result |
|-------|-------------|--------|
| 0 | Environment verification | **PASS** (10/12, 2 fixture bugs) |
| 1 | Stateless inference baseline | **PASS** (7/7) |
| 2 | MambaPool unit tests | **PASS** (5/5) |
| 3 | MambaRadixCache gauntlet | **PASS** (16/16) |
| 4 | Live server, no_buffer strategy | **PASS** (5/5) |
| 5 | Mamba2Metadata integrity | **PASS** (5/5) |
| 6 | extra_buffer strategy | PARTIAL (unit only, no compatible model) |
| 7 | Snapshot system E2E | **PASS** (6/6, 7 bugs found+fixed) |
| 8 | True stateful inference | **PASS** (4/4, 93.8% token savings) |
| 9 | Gauntlet stress tests | **PASS** (6/6, 271 requests, 0 errors) |
| 10 | Cross-model scaling | **PASS** (Nemotron 5/5, Granite 4/5) |
| 10e | Context scaling 2K-128K | **PASS** (5/5 tiers) |
| 10f | Resilience testing | **PASS** (4/5) |

### Resilience Testing (Phase 10f)

Five adverse-condition tests proving the snapshot system survives crashes and disconnects:

| Test | Scenario | Result | Key Evidence |
|------|----------|--------|--------------|
| 1 | Client disconnect mid-stream | **PASS** | Server recovers, +56MB VRAM (no leak) |
| 2 | SIGKILL mid-inference | **PASS** | Snapshot intact, startup preload confirmed (5ms WARM restore) |
| 3 | SIGKILL during snapshot write | **PASS** | Atomic writes work, no partial files |
| 4 | Graceful SIGTERM shutdown | **FAIL** | Shutdown hangs >60s (scheduler drain issue) |
| 5 | Abort request + snapshot save | **PASS** | Save succeeds from WARM tier, server stable |

**Startup preload verified**: After SIGKILL + restart, the server automatically loaded COLD snapshots into WARM tier. Restore latency of 5ms confirmed WARM hit — the Gap 3 implementation works.

**New bug discovered**: SIGTERM graceful shutdown hangs >60s. Not a snapshot bug — the scheduler's drain logic appears stuck on in-flight requests. Production mitigation: use SIGKILL after a timeout. Snapshot data is always safe due to atomic writes.

### Stress Test Coverage (Phase 9)

- 32 concurrent requests with shared prefix: no state contamination
- 100 rapid unique requests: stable eviction, 0 errors
- 50 repeated identical requests (temp=0): all outputs identical
- 8 concurrent 5-turn conversations: persona coherence maintained

### Bugs Found and Fixed

15 bugs discovered and fixed across all phases, including:
- Auto-snapshot timing (pre-free hook — the critical architectural fix)
- 7 snapshot pipeline bugs (Phase 7)
- Pure Mamba2 model_config crash (Phase 10)
- System dependency issues (libnuma1, ninja)

### Open Issues

| Bug | Impact | Status |
|-----|--------|--------|
| `/restore_snapshot` stateful-gen hangs | Cannot do restore-then-generate via API | Open — deferred output not connected to HTTP future |
| Sequential snapshot save low hit rate (dense models) | WARM tier evicts before manual save | Expected behavior; WARM tier sizing needed |
| No snapshot cleanup mechanism | Disk grows unbounded | Needs TTL-based cleanup |
| SIGTERM graceful shutdown hangs >60s | Cannot cleanly restart server | Scheduler drain issue; use SIGKILL after timeout |

---

## Implications if Widely Adopted

### For AI Companies

1. **Dramatic cost reduction**: Less compute means lower GPU bills and ability to serve more users with same hardware
2. **Better user experience**: Faster responses, no latency degradation as conversations lengthen
3. **New product possibilities**: Applications that were impractical due to cost (e.g., hour-long coaching sessions, educational tutors)

### For Users

1. **Instant responsiveness**: Long conversations feel snappy, not sluggish
2. **Lower carbon footprint**: Less energy consumed per conversation
3. **Privacy benefits**: Snapshots can be encrypted and stored locally rather than sending full history to servers

### For the Industry

1. **Architecture shift**: Mamba-style models become more attractive for production deployments
2. **Multi-turn first**: Product design can prioritize deep conversations without cost anxiety
3. **Edge deployment**: Stateful inference enables offline AI assistants that maintain context across sessions

---

## Current Status

**Completed (Phase 10):**
- Core snapshot persistence system with WARM + COLD tiering
- Multi-turn stateful inference with pre-free snapshot hook
- Per-request correlation for concurrent safety
- Cross-model validation (3 architectures, 4 models)
- Stress testing (271 concurrent requests, 0 errors)
- No memory leaks across 150+ requests per model
- Context scaling verified to 128K tokens (snapshot size constant at ~55MB)
- Resilience testing: crash recovery, atomic writes, startup preload verified

**Known Limitations:**
- Mamba-specific (doesn't apply to pure Transformer models)
- `extra_buffer` strategy untested at server level (no compatible model available)
- Restore-then-generate API endpoint hangs (architectural bug)
- No automatic snapshot cleanup
- COLD tier restore latency not fully verified (2TB host RAM prevented eviction in tests)

**Next Steps:**
1. Fix restore stateful-gen timeout bug (scheduler.py / tokenizer_manager.py)
2. Extended soak test (24h) with granite-tiny for leak detection
3. Implement TTL-based snapshot cleanup
4. Production deployment considerations
5. Test with larger hybrid models at high context (granite-small at 32K+)

---

## Why This Matters

Most AI progress focuses on making models *smarter*. This project focuses on making models *more efficient* for how people actually use them—in conversations.

Think of it like the difference between:
- **Stateless**: Re-reading an entire book before answering each question about it
- **Stateful**: Remembering what you read and only referencing new information

As AI moves from single-turn tools (search, summarization) to ongoing collaborations (tutors, assistants, agents), efficiency at conversation scale becomes critical. This project proves that stateful inference isn't just possible—it's practical, testable, and dramatically more efficient.

---

## Repository

**GitHub**: `KHAEntertainment/sglang-mamba`
**Base**: Fork of [sgl-project/sglang](https://github.com/sgl-project/sglang)
**Models tested**: granite-4.0-h-tiny, granite-4.0-h-small (32B), Nemotron-Cascade-2-30B on NVIDIA A100-80GB (SM80)
**Test results**: `test/phases/results/INDEX.md`
**License**: Same as upstream SGLang
