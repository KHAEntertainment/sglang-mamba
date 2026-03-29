# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **SGLang with stateful Mamba inference** — a fork of [upstream SGLang](https://github.com/sgl-project/sglang) that adds snapshot persistence capabilities for Mamba SSM (State Space Model) hidden states. The key innovation is saving/restoring Mamba's internal memory to enable fast multi-turn conversations (25x+ speedup on subsequent turns, measured on a Tesla V100-SXM2-16GB while serving `granite-4.0-h-tiny` in the multi-turn phase validation flow documented under `test/phases/results/`).

**Upstream**: https://github.com/sgl-project/sglang

## Project Tracking

**This project is managed in Linear.** All work, issues, and milestones are tracked at:

> **Workspace:** https://linear.app/khaentertainment
> **Project:** [SGLang - Mamba](https://linear.app/khaentertainment/project/sglang-mamba-e14f2152be8d)

**Use Linear for:**
- Checking current work status and priorities
- Creating new issues before starting work
- Updating issue status as work progresses
- Tracking milestones and DoD checkpoints

**Do NOT use markdown files for tracking TODOs or project state.** The `phase3/` directory contains historical implementation notes — reference them for context, but Linear is the source of truth for active work.

## Common Commands

### Installation
```bash
pip install -e "python[all]"       # Install with all dependencies
pip install -e "python[test]"     # Install with test dependencies
```

### Running the Server
```bash
python -m sglang.launch_server --model-path <model_name> [--port 30000]
```

Key server flags for this fork:
- `--enable-snapshot-persistence` — Enable Mamba state snapshot save/restore
- `--snapshot-dir <path>` — Directory for snapshot storage
- `--enable-memory-tiers` — Enable VRAM/RAM/Disk tier management
- `--enable-agent-tools` — Enable agent tool calling framework

### Running Tests
```bash
# Unit tests for snapshot and agent framework (this fork's additions)
cd test/sglang
pytest snapshot/ agents/ -v                    # Run all snapshot/agent tests
pytest snapshot/test_mamba_snapshot.py -v     # Run specific test file
pytest snapshot/ -k "test_metadata" -v        # Run tests matching pattern

# SRT (runtime) tests
cd test/srt
python test_srt_endpoint.py                   # Run SRT endpoint tests
python run_suite.py --suite per-commit        # Run CI test suite

# Manual tests
cd test/manual
python test_*.py                              # Various manual tests
```

### Linting
```bash
pre-commit run --all-files                     # Run all pre-commit hooks
pre-commit run ruff --files <files>            # Run ruff linter on specific files
```

### Building Documentation
```bash
# Docs are in docs/ directory - check specific doc for build requirements
```

## Architecture

### Key Directory Structure

```text
python/sglang/
├── __init__.py              # Public API exports (Runtime, function, gen, SnapshotManager)
├── snapshot.py              # High-level snapshot API (user-facing)
├── launch_server.py         # Server entry point
├── lang/                    # Frontend Language API
│   ├── api.py              # @function, @gen, Runtime, etc.
│   ├── interpreter.py      # Statement interpretation
│   └── ir.py               # Intermediate representation
├── srt/                     # SGLang Runtime (core inference engine)
│   ├── entrypoints/        # HTTP/gRPC server entrypoints
│   ├── server_args.py      # Server configuration
│   ├── snapshot/           # Mamba snapshot implementation
│   │   ├── mamba_snapshot.py       # Core snapshot save/restore logic
│   │   ├── mamba_host_pool.py      # Host memory pool management
│   │   ├── tier_manager.py        # 3-tier VRAM/RAM/Disk orchestration + startup restore
│   │   ├── conversation_tracker.py # Tier state tracking
│   │   ├── snapshot_hooks.py      # Hook manager (post_forward, pre_eviction, on_demand)
│   │   └── snapshot_policy.py     # Snapshot eviction policies
│   ├── agents/             # Agent framework
│   │   ├── agent_loop.py          # Agent execution loop
│   │   ├── tool_registry.py       # Tool registration
│   │   ├── builtin_tools.py       # Calculator, memory tools
│   │   └── api/                   # REST/WebSocket handlers
│   ├── models/            # Model implementations
│   │   └── registry.py    # Model registry
│   ├── layers/
│   │   └── attention/mamba/   # Mamba attention layer
│   └── mem_cache/
│       └── mamba_radix_cache.py # Mamba-specific radix cache
└── jit_kernel/            # CUDA kernel implementations
    └── hadamard.py        # Fast Hadamard transform for Mamba

test/
├── sglang/                 # Tests for this fork's additions
│   ├── snapshot/          # Snapshot save/restore tests
│   └── agents/           # Agent framework tests
├── srt/                   # SRT runtime tests
├── registered/            # CI registry tests
│   └── radix_cache/       # MambaRadixCache comprehensive + gauntlet tests
├── phases/                # 9-phase GPU test plan (phases 0-8)
│   ├── config.sh          # Single source of truth for paths/ports
│   ├── codemap.md         # Code navigation reference
│   └── results/           # Phase execution reports
└── manual/                # Manual integration tests

.beads/                     # Beads issue tracking (initialized)
skills/mamba-sglang/        # Gemini CLI skill + reference docs
docs/migration-prep/        # VM migration context (gitignored)
```

### Core Components

**Snapshot System** (`python/sglang/srt/snapshot/`):
- `MambaSnapshotManager` — Low-level serialization to safetensors + JSON
- `MambaHostPool` — Host memory pool for snapshot staging between GPU and disk
- `TierManager` — 3-tier memory management (VRAM/RAM/Disk) with automatic transitions
- `restore_latest_snapshots_to_warm_tier()` — Startup restore into WARM tier (Gap 3 — DONE)
- `snapshot_policy.py` — LRU eviction policies for tier management

**Agent Framework** (`python/sglang/srt/agents/`):
- `AgentLoop` — Tool-calling agent execution with max iterations
- `ToolRegistry` — Registry for built-in (calculator, memory_store/recall/search) and custom tools
- REST API at `/v1/agent/*` endpoints

**Mamba Integration**:
- `sglang/srt/layers/attention/mamba/mamba.py` — Mamba layer implementation
- `sglang/srt/mem_cache/mamba_radix_cache.py` — Radix cache for Mamba
- `sglang/srt/configs/mamba_utils.py` — Mamba-specific utilities

### Key Data Structures

- `MambaSnapshotMetadata` — Dataclass tracking conversation_id, turn_number, token_count, mamba_pool_idx, layer_config, fill_ids
- `MambaHostPool` — Manages host (RAM) staging area for snapshots
- `SnapshotPolicy` — Enum controlling when snapshots are created (eager/lazy/effective_size)

## This Fork's Key Features

1. **Snapshot Persistence**: Save Mamba SSM hidden states to disk, restore later (10-50ms save, 5-30ms restore)
2. **3-Tier Memory Management**: Active (VRAM) → Warm (RAM) → Cold (Disk) with automatic LRU eviction
3. **Agent Framework**: Tool calling with 4 built-in tools (calculator, memory store/recall/search)
4. **Fast Hadamard Transform**: Custom CUDA kernels for Mamba's structured state space operations

## Testing Notes

- This fork adds tests in `test/sglang/snapshot/` and `test/sglang/agents/`
- Tests use `pytest` with `asyncio_mode = auto` (see `test/pytest.ini`)
- SRT tests use `unittest` framework directly
- Integration tests in `test/registered/` use the CI registry system
- **GPU test phases** in `test/phases/` (phases 0-8): phases 0/2/3/5 PASS on V100, phases 1/4/6/7/8 blocked by sm75+ GPU requirement
- `test/phases/config.sh` is the single source of truth for model paths and server ports

## Development Phases

> Historical implementation context lives in `phase3/` — see for architecture details and validation reports. Active work tracking is in Linear (see above).

### Phase Status (Linear)
- Phase 3.1 ✅ Complete - Foundation
- Phase 3.2 ✅ Complete - Core Implementation
- Phase 3.3 ✅ Complete - Static Analysis (KHA-7 through KHA-11: performance optimizations identified)
- Phase 3.4 ⬜ Pending - [KHA-6: Final Audit], [KHA-14: cleanup phase3 docs]
- **All 3 snapshot gaps CLOSED**: Gap 1 (fill_ids sync) ✅, Gap 2 (create_new_request) ✅, Gap 3 (startup warm restore) ✅

### Active Issues (Linear)
| ID | Priority | Title | Status |
|----|----------|-------|--------|
| ~~KHA-5~~ | ~~High~~ | ~~Implement restore_snapshots_on_startup (Gap 3)~~ | **CLOSED** |
| [KHA-6](https://linear.app/khaentertainment/issue/KHA-6) | Medium | Phase 3.4 — Final Audit | Backlog |
| [KHA-7](https://linear.app/khaentertainment/issue/KHA-7) | Urgent | [PERF] Remove setattr/getattr overhead in LRUList | Backlog |
| [KHA-8](https://linear.app/khaentertainment/issue/KHA-8) | High | [PERF] Optimize tensor cloning | Backlog |
| [KHA-9](https://linear.app/khaentertainment/issue/KHA-9) | Medium | [PERF] Cache tree depth for lock optimization | Backlog |
| [KHA-10](https://linear.app/khaentertainment/issue/KHA-10) | High | [PERF] Optimize LRU traversal | Backlog |
| [KHA-11](https://linear.app/khaentertainment/issue/KHA-11) | Medium | [PERF] Batch lock operations | Backlog |
| [KHA-12](https://linear.app/khaentertainment/issue/KHA-12) | High | Document sglang.srt.models.mamba | Backlog |
| [KHA-13](https://linear.app/khaentertainment/issue/KHA-13) | High | Document sglang.srt.layers.mamba | Backlog |
| [KHA-14](https://linear.app/khaentertainment/issue/KHA-14) | Low | Clean up phase3 docs after Phase 3.4 completes | Backlog |

> **Note:** Performance issues (KHA-7 through KHA-11) are blocked by server-phase testing. Do not start until Phases 1, 4, and 7 pass on sm75+ hardware.

## GCloud Testing Instance

> **Credentials**: See the private runbook for access credentials and instance details.

**Instance:** `<INSTANCE_NAME>`
**Zone:** `<ZONE>`
**Project:** `<PROJECT_ID>`
**GPU:** `<GPU_TYPE>`

Connect via:
```bash
# Template — fill in values from runbook
gcloud compute ssh --zone "<ZONE>" "<INSTANCE_NAME>" --project "<PROJECT_ID>"
```

Clone at: `<CLONE_PATH>`
Cloudflare tunnel: `<TUNNEL_HOST>` → `localhost:30000`

### Available Models

| Model | Path | Type | Use Case |
|-------|------|------|----------|
| granite-4.0-h-tiny | `<MODEL_DIR>/granite-4.0-h-tiny` | FP16, 40-layer hybrid | Primary (try first) |
| Nemotron-4B | `<MODEL_DIR>/NVIDIA-Nemotron-3-Nano-4B-BF16` | FP16, 4B | Backup if granite OOMs |
| Granite-Q4 | `<MODEL_DIR>/granite-4.0-h-tiny-gguf/granite-4.0-h-tiny-Q4_K_M.gguf` | GGUF Q4 | Quantized comparison (args TBD) |

**Testing priority:** granite-4.0-h-tiny → Nemotron (if OOM) → Granite-Q4 (comparison pass)

## Known Issues

1. **Mamba model config bug**: `architectures: None` causes crash in model_config.py line 149 *(fixed)*
2. **Granite GGUF**: `granitehybrid` architecture not supported by transformers GGUF loader
3. **V100 memory**: Full Granite MoE models require >16GB GPU memory — use Nemotron fallback
4. **sm75+ GPU blocker**: FLA Mamba2 CUDA kernels + FlashInfer require sm75+. Server test phases (1/4/6/7/8) cannot run on V100 (sm70). Need A100, T4, A10G, or RTX 30xx+
5. **Current branch behind main**: `fix/snapshot-restore-state-sync` is 8 commits behind `main` (missing PR #6 startup restore). Merge `main` in before continuing work.

## Context & Project Management Access

**Do not use the Linear CLI or native Linear MCP directly.**

All Linear interaction goes through **core-memory MCP** via `execute_integration_action`. This builds accumulated project memory over time.

**Session start pattern:**
```
# 1. Search memory for prior context
memory_search("sglang-mamba")

# 2. Check Linear backlog via core-memory
execute_integration_action(
  accountId: "0b4764e3-a793-4537-89b7-b26eff7b7675",
  action: "linear_search_issues",
  params: { query: "sglang-mamba", first: 20 }
)
```

**Available Linear actions through core-memory:**
- `linear_search_issues` — search/filter by project, label, state, text
- `linear_create_issue` — create with projectId, priority, labels, parent
- `linear_update_issue` — update state, project, labels (requires internal UUID, not KHA-XX)
- `linear_create_project` / `linear_update_project` — manage projects
- `linear_create_label` — create team labels

**Linear accountId:** `0b4764e3-a793-4537-89b7-b26eff7b7675`
**Linear projectId:** `f7f1cb8c-c4cd-4b63-83f6-58b9ddba6ce8`
**Linear teamId:** `1ee12f51-86fc-4bce-8cad-845d8a67bfa9`

## Memory Context

For session persistence, important project context is stored in:
- Global memory: `~/.claude/projects/<PROJECT_PATH>/memory/`
- Core-memory MCP: accumulated context across all agent sessions
- VM migration context: `docs/migration-prep/` (gitignored, not for commit)

**Linear is the source of truth for project management** — see the Project Tracking section above.
