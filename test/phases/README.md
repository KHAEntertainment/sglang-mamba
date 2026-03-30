# SGLang-Mamba Test Suite

Self-contained test program for Mamba SSM snapshot persistence. Each phase document in `prompts/` is an agent prompt that a fresh Claude Code session can pick up and execute autonomously.

## Directory Layout

```
test/phases/
├── README.md           ← you are here
├── prompts/            Phase definition docs (agent prompts)
│   ├── phase-00-*.md     Phase 0: Environment Verification
│   ├── phase-01-*.md     Phase 1: Stateless Inference Baseline
│   ├── ...
│   └── phase-10-*.md     Phase 10: Scaling Plan
├── scripts/            Runnable test scripts and tools
│   ├── phase-10-scaling.py        Resource monitor + load test runner
│   ├── phase-10-h-small-test.py   Granite-specific test script
│   └── download-model.sh          Model download helper
├── infra/              Infrastructure and configuration
│   ├── config.sh       Single source of truth: MODEL_PATH, PORT, SNAPSHOT_DIR
│   ├── codemap.md      File/line/method reference for all source code
│   ├── BOOTSTRAP_FRESH_VM.md  Fresh VM bootstrap instructions
│   └── NEW_VM_SETUP.md        New VM setup guide
└── results/            Test results and reports
    ├── INDEX.md        Master index with summaries of every phase
    ├── phase-NN-*.md   Individual phase result reports
    └── phase-10-logs/  Raw JSON/CSV data from Phase 10
```

## Phase Dependency Graph

```text
Phase 0 (Environment Verification)
  └─► Phase 1 (Stateless Inference Baseline)
        └─► Phase 2 (MambaPool Unit Tests)      ◄─ no server needed
              └─► Phase 3 (RadixCache Component) ◄─ no server needed
                    ├─► Phase 4 (Live Server, no_buffer)
                    │     └─► Phase 5 (Mamba2Metadata Integrity) ◄─ no server
                    │           └─► Phase 6 (extra_buffer Strategy)
                    │                 └─► Phase 7 (Snapshot System)
                    │                       └─► Phase 8 (Stateful Inference)
                    │                             └─► Phase 9 (Gauntlet / Stress)
                    │                                   └─► Phase 10 (Scaling)
                    └─► (can also run Phase 2, 3, 5 in parallel after Phase 1)
```

## Quick Start

```bash
# 1. Configure (edit MODEL_PATH if needed)
source test/phases/infra/config.sh

# 2. Run a phase by feeding its prompt doc to an agent
#    e.g. phase-01:
cat test/phases/prompts/phase-01-stateless-inference-baseline.md

# 3. Check results
cat test/phases/results/INDEX.md
```

## Configuration

`infra/config.sh` — single source of truth for `MODEL_PATH`, `MODEL_NAME`, `SERVER_PORT`, `SNAPSHOT_DIR`, and `RESULTS_DIR`. All phase prompts source this file.

**To test a different model**: edit `MODEL_PATH` and `MODEL_NAME` in `config.sh` only. All phases pick it up automatically.

## Results

All results in `results/`. Start with `results/INDEX.md` for the master summary with per-phase pass/fail, key findings, and cross-phase bug tracker.

Each phase writes a detailed markdown report named `phase-NN-<model>-<date>.md`. Reports include: per-test pass/fail table, HITL transcripts, tracebacks, server log excerpts, and phase-specific observations.

## Common Conventions

- **CI registration**: All new test files use `register_cuda_ci(est_time=..., suite="stage-b-test-small-1-gpu")` at the top.
- **Test runner**: `python test/run_suite.py --hw cuda --suite stage-b-test-small-1-gpu`
- **Project root**: `$(git rev-parse --show-toplevel)` (repository root)
- **Registered test dir**: `test/registered/radix_cache/`
- **HITL interface**: Available for qualitative smoke checks where noted.

## Reporting Convention

At the end of each phase session, the executing agent should output:

```text
PHASE N RESULT: PASS | FAIL
Tests run: X  Passed: Y  Failed: Z
HITL: PASS | SKIP | N/A
Notes: <any failures, tracebacks, or observations>
```
