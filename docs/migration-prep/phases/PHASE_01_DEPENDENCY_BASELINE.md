# PHASE 01 — Dependency Baseline

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

Reconcile build and packaging dependencies so the fork can compile and import against upstream's current dependency matrix before any source code merge begins.

## Upstream Commits to Integrate

| Commit | Topic |
|--------|-------|
| `682294151` | flashinfer-python bump to 0.6.6 |
| `93afe15b4` | flashinfer-cubin bump to 0.6.6 |
| `15097c5c3` | sgl-kernel bump to 0.4.0 |
| `d1e95af28` | transformers bump to 5.3.0 (includes Granite model evolution) |
| `f289d173a` | xgrammar bump to 0.1.32 |
| `4a757990a` | torchcodec 0.9.1 + video decode dependency splits |
| `654fc02cf` | gRPC packaging refactor to `smg-grpc-servicer>=0.5.0` |
| `025691cd9` | cache-dit bump to 1.3.0 |

## Files Touched

- `pyproject.toml`
- `python/pyproject.toml`
- `python/sglang/__init__.py` (possible version pins)
- Packaging/CI workflows (optional, low priority)

## Decision Points

1. **flashinfer 0.6.6 / sgl-kernel 0.4.0** — **ADOPT upstream**. These are performance-critical compiled deps; staying behind risks ABI mismatch with upstream kernels.
2. **transformers 5.3.0** — **ADOPT upstream**. Granite model configs evolved in this range; our inference tests depend on Granite.
3. **gRPC servicer packaging** — **ADOPT upstream**. Replace direct `grpcio` / `smg-grpc-proto` deps with `smg-grpc-servicer>=0.5.0`. Update any direct proto imports if they changed (research via DeepWiki if import errors occur).
4. **torchcodec / video decode** — **ADOPT upstream**. Conditional deps are harmless for our Mamba-only server path.
5. **xgrammar / cache-dit** — **ADOPT upstream**. No fork customizations in grammar/diffusion paths.

## Execution Steps

0. **Sync with origin and create the worktree** (first phase only):
   ```bash
   cd <sglang-mamba repo root>
   git fetch origin
   git pull origin main
   git log --oneline -5   # Verify latest PRs/pushes are present
   git worktree add worktrees/upstream-sync-2026-Q1 -b upstream-sync-2026-Q1 main
   cd worktrees/upstream-sync-2026-Q1
   ```
   Then run the Worktree Safety Check above to confirm you're in the right place.

1. **Add and fetch the upstream remote** (if not already present):
   ```bash
   git remote add upstream https://github.com/sgl-project/sglang.git 2>/dev/null || true
   git fetch upstream main
   ```

2. Cherry-pick or manually apply the dependency changes from the commits above.
3. Run `pip install -e "python/"` in a clean venv (or on the A100 host).
4. Verify that `python -c "import sglang"` succeeds.
5. Run Phase 0 and Phase 1 baselines.
6. Run Phase 2 unit tests (`pytest python/sglang/test/srt/test_mamba_unittest.py -v`).

## Validation Criteria

- `pip install -e "python/"` completes without dependency resolution errors.
- Phase 0 (environment verification) passes.
- Phase 1 (stateless inference baseline) passes.
- Phase 2 (MambaPool unit tests) passes (5/5).

## Rollback Plan

1. `git checkout -- pyproject.toml python/pyproject.toml`
2. `pip install -e "python/"` from `main` state.

## Estimated Complexity

**LOW-MEDIUM** — 2 to 4 hours. Most time is in installation and resolving any compile-time issues on sm80.

## Dependencies

None. This is the first phase.

## Team Structure

**Solo agent** with DeepWiki access (to verify gRPC import expectations if breakage occurs).

## bd Workflow

```bash
bd ready --json                    # Confirm Phase 01 is unblocked
bd update <phase-01-id> --claim    # Claim before starting
# ... do the work ...
bd close <phase-01-id> --reason "Phase 01 PASS. Tagged phase-01-pass."
```