#!/usr/bin/env bash
# ============================================================
# Beads (bd) Setup for Upstream SGLang Sync
# Run this ONCE before starting Phase 01.
# Uses Dolt SERVER mode to avoid embedded mode issues.
# ============================================================
set -euo pipefail

# Workaround for Dolt embedded mode issues: use server mode
REPO_ROOT="$(git rev-parse --show-toplevel)"
REPO_NAME="$(basename "$REPO_ROOT")"
BEADS_REAL="$HOME/.beads-stores/$REPO_NAME"

# Export BEADS_DIR to space-free location for all bd commands
export BEADS_DIR="$BEADS_REAL"

# Remove any existing .beads symlink to avoid confusion
rm -f .beads 2>/dev/null || true

if [ -d "$BEADS_REAL" ] && bd list --json > /dev/null 2>&1; then
  echo "=== Existing beads database found — adding phase tasks ==="
else
  echo "=== Initializing Beads (Dolt server mode) at $BEADS_REAL ==="
  rm -rf "$BEADS_REAL" 2>/dev/null || true
  mkdir -p "$BEADS_REAL"

  # Initialize with server mode
  bd init --server --skip-hooks --skip-agents

  # Create symlink for convenience
  ln -sf "$BEADS_REAL" .beads
fi

# Verify beads is working
if ! bd list --json > /dev/null 2>&1; then
  echo "ERROR: beads database not working. Check BEADS_DIR=$BEADS_DIR"
  ls -la "$BEADS_REAL"
  exit 1
fi

# ── Phase tasks (top-level) ──────────────────────────────────
echo "=== Creating phase tasks ==="

P01=$(bd create "Phase 01: Dependency Baseline" \
  -t task -p 2 \
  --description "Reconcile build/packaging deps (flashinfer 0.6.6, sgl-kernel 0.4.0, transformers 5.3.0, gRPC refactor). Validate: pip install, Phase 0/1/2." \
  --json | jq -r '.id')
echo "Phase 01: $P01"

P02=$(bd create "Phase 02: Observability Import Rebase" \
  -t task -p 2 \
  --description "Absorb observability refactor (3b8930227). Update fork imports to srt/observability/. Validate: import checks, Phase 1." \
  --json | jq -r '.id')
echo "Phase 02: $P02"

P03=$(bd create "Phase 03: SessionController Port" \
  -t task -p 0 \
  --description "Port scheduler/session customizations onto upstream SessionController. DESIGN REVIEW at create_new_request restore flow. Validate: Phase 1/4/7/8." \
  --json | jq -r '.id')
echo "Phase 03: $P03"

P04=$(bd create "Phase 04: Mamba Cache Architecture Reconcile" \
  -t task -p 0 \
  --description "Reconcile HiMambaRadixCache, HybridCacheController, host pools with snapshot/tier system. DESIGN REVIEW at TierManager layering. Includes COMPLEMENT commit sweep. Validate: Phase 2-8." \
  --json | jq -r '.id')
echo "Phase 04: $P04"

P05=$(bd create "Phase 05: Scheduler Idle and Pool Fixes" \
  -t task -p 1 \
  --description "Merge scheduler idle-detection and Mamba slot-release fixes. Coordinate TierManager background tasks with is_fully_idle(). Validate: Phase 2-8." \
  --json | jq -r '.id')
echo "Phase 05: $P05"

P06=$(bd create "Phase 06: Server and HTTP Drift" \
  -t task -p 1 \
  --description "Merge lower-risk server_args.py and http_server.py drift. Re-apply snapshot REST endpoints and CLI flags. Validate: Phase 1/4/7/8." \
  --json | jq -r '.id')
echo "Phase 06: $P06"

P06B=$(bd create "Phase 06B: Bulk Upstream Merge" \
  -t task -p 1 \
  --description "git merge upstream/main to sweep in ~1,290 remaining non-conflicting commits. All architecturally sensitive files already reconciled. Validate: Phase 0/1/2/4/7." \
  --json | jq -r '.id')
echo "Phase 06B: $P06B"

P07=$(bd create "Phase 07: Re-apply Snapshot Features" \
  -t task -p 0 \
  --description "Re-port snapshot/ package, io_struct, scheduler handlers, tokenizer_manager, REST routes, CLI flags onto full upstream substrate. Validate: ALL phases 0-8." \
  --json | jq -r '.id')
echo "Phase 07: $P07"

P08=$(bd create "Phase 08: Full Validation and Stress Test" \
  -t task -p 0 \
  --description "Run complete validation suite: Phases 0-9 + 10e (context scaling 2K-128K) + 10f (resilience/crash). Fix-forward or reset as needed. Gate model: granite-4.0-h-tiny." \
  --json | jq -r '.id')
echo "Phase 08: $P08"

# ── Dependencies (linear chain) ──────────────────────────────
echo "=== Setting up dependency chain ==="
bd dep add "$P02" "$P01"
bd dep add "$P03" "$P02"
bd dep add "$P04" "$P03"
bd dep add "$P05" "$P04"
bd dep add "$P06" "$P05"
bd dep add "$P06B" "$P06"
bd dep add "$P07" "$P06B"
bd dep add "$P08" "$P07"

# ── Sub-tasks for HIGH phases ────────────────────────────────
echo "=== Creating sub-tasks for Phase 03 (HIGH) ==="

P03_R=$(bd create "P03: Research SessionController API contracts" \
  -t task -p 1 \
  --description "Use DeepWiki to read upstream session_controller.py. Report on: SessionController class API, Session.create_req() signature, where Scheduler delegates for request creation." \
  --json | jq -r '.id')
bd dep add "$P03_R" "$P02"

P03_DR=$(bd create "P03: DESIGN REVIEW — create_new_request restore flow" \
  -t task -p 0 \
  --description "Orchestrator decides Option A (inject before Session.create_req) vs Option B (post-create_req hook). Research Agent findings required first." \
  --json | jq -r '.id')
bd dep add "$P03_DR" "$P03_R"

P03_I=$(bd create "P03: Implement SessionController merge + snapshot hook port" \
  -t task -p 0 \
  --description "Merge 3 upstream commits (5acb45cf3, c6cb0c964, e08ef0675 + e1ee68d0f). Port snapshot handlers to SessionController. Apply design review decision." \
  --json | jq -r '.id')
bd dep add "$P03_I" "$P03_DR"

P03_V=$(bd create "P03: Validate Phase 1/4/7/8" \
  -t task -p 0 \
  --description "Run Phase 1 (stateless), Phase 4 (live server), Phase 7 (snapshot E2E 6/6), Phase 8 (stateful inference). All must PASS." \
  --json | jq -r '.id')
bd dep add "$P03_V" "$P03_I"

echo "=== Creating sub-tasks for Phase 04 (HIGH) ==="

P04_R=$(bd create "P04: Research HiMambaRadixCache + HybridCacheController + pool structs" \
  -t task -p 1 \
  --description "Use DeepWiki to confirm: HiMambaRadixCache init signature, HybridCacheController API, MambaPoolHost layout vs MambaHostPool, HybridLinearKVPool/HybridReqToTokenPool constructors." \
  --json | jq -r '.id')
bd dep add "$P04_R" "$P03_V"

P04_DR=$(bd create "P04: DESIGN REVIEW — TierManager vs HybridCacheController layering" \
  -t task -p 0 \
  --description "Confirm TierManager (disk persistence) layers on top of HybridCacheController (GPU<->host offload) without replacement. Verify extract/inject pool compatibility." \
  --json | jq -r '.id')
bd dep add "$P04_DR" "$P04_R"

P04_I=$(bd create "P04: Implement cache architecture merge + COMPLEMENT sweep" \
  -t task -p 0 \
  --description "Merge 7 upstream commits. Replace fork mamba_radix_cache.py with upstream. Adapt tier_manager.py. Cherry-pick 10 COMPLEMENT commits. Add cuda sync guard if needed." \
  --json | jq -r '.id')
bd dep add "$P04_I" "$P04_DR"

P04_V=$(bd create "P04: Validate Phases 2-8" \
  -t task -p 0 \
  --description "Run Phases 2 (MambaPool 5/5), 3 (gauntlet 16/16), 4, 5, 6, 7 (6/6), 8. All must PASS." \
  --json | jq -r '.id')
bd dep add "$P04_V" "$P04_I"

echo "=== Creating sub-tasks for Phase 07 (HIGH) ==="

P07_R=$(bd create "P07: Research upstream MambaPool.State + batch result path + SessionController hooks" \
  -t task -p 1 \
  --description "DeepWiki verification of current MambaPool.State/SpeculativeState fields, Scheduler batch result path for post-forward hooks, SessionController interaction points." \
  --json | jq -r '.id')
bd dep add "$P07_R" "$P06B"

P07_I=$(bd create "P07: Port snapshot/ package + handlers + routes + flags" \
  -t task -p 0 \
  --description "Copy snapshot/ package. Re-apply io_struct snapshot models, scheduler handlers, tokenizer_manager queues, http_server routes, server_args flags. Adapt to SessionController." \
  --json | jq -r '.id')
bd dep add "$P07_I" "$P07_R"

P07_V=$(bd create "P07: Validate ALL phases 0-8" \
  -t task -p 0 \
  --description "Run full phase suite 0-8. All must PASS. Snapshot save/restore cycle must complete. Stateful inference must recall semantic context." \
  --json | jq -r '.id')
bd dep add "$P07_V" "$P07_I"

echo "=== Creating sub-tasks for Phase 08 (validation) ==="

P08_CORE=$(bd create "P08: Run Phases 0-9 core validation" \
  -t task -p 0 \
  --description "Sequential run: Phase 0 (env), 1 (stateless), 2 (MambaPool 5/5), 3 (gauntlet 16/16), 4 (live server), 5 (metadata 5/5), 6 (extra_buffer), 7 (snapshot 6/6), 8 (stateful), 9 (stress)." \
  --json | jq -r '.id')
bd dep add "$P08_CORE" "$P07_V"

P08_10E=$(bd create "P08: Run Phase 10e context scaling (2K/8K/32K/64K/128K)" \
  -t task -p 0 \
  --description "Validate constant snapshot size and ~2ms restore across all context tiers on granite-4.0-h-tiny." \
  --json | jq -r '.id')
bd dep add "$P08_10E" "$P08_CORE"

P08_10F=$(bd create "P08: Run Phase 10f resilience/crash testing" \
  -t task -p 0 \
  --description "5 scenarios: client disconnect, SIGKILL mid-inference, SIGKILL during write, SIGTERM graceful, abort+save. Baseline: 4/5 PASS (SIGTERM hang is known Bug #16)." \
  --json | jq -r '.id')
bd dep add "$P08_10F" "$P08_CORE"

# ── Summary ──────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo "Phase chain: $P01 → $P02 → $P03 → $P04 → $P05 → $P06 → $P06B → $P07 → $P08"
echo ""
echo "Run 'bd ready --json' to see unblocked tasks."
echo "Run 'bd show <id>' for task details."
echo ""
bd ready
