<!-- ENGRAM_MODIFIED — Stateful Mamba documentation index -->
# Engram Docs Index

Index for the snapshot and statefulness documentation in this repository.

## Start Here

| I want to... | Go to... |
|--------------|----------|
| Read the user-friendly API docs | [API Guide](api_guide.md) |
| Understand the canonical technical contract | [HTTP API Spec](http_api_spec.md) |
| Learn the current usage model | [User Guide](user_guide.md) |
| Migrate an integration | [Migration Guide](migration_guide.md) |
| See current examples | [Examples](examples.md) |
| Understand the implementation shape | [Architecture](architecture.md) |
| Debug a failure | [Troubleshooting](troubleshooting.md) |
| Read a summary of the docs set | [SUMMARY.md](SUMMARY.md) |

## Canonical Technical Reference

### `api_guide.md`

The user-friendly API guide intended for most integrators.

It covers:

- quick start
- authentication
- common workflows
- curl, Python, and JavaScript examples
- practical error-handling guidance

### `http_api_spec.md`

This is the current canonical technical reference for Engram's added snapshot
HTTP surface.

It covers:

- the five snapshot endpoints
- auth behavior
- selector semantics
- restore-only vs restore-and-generate
- current quirks and contract oddities

## Active Guides

### `user_guide.md`

High-level guide to the current public snapshot surfaces:

- direct selector-based methods on `s`
- `SnapshotManager(runtime.endpoint)`
- conversation and turn selection model

### `migration_guide.md`

How to adopt Engram snapshot statefulness from a normal SGLang serving flow.

### `examples.md`

Current examples aligned to the live implementation.

### `architecture.md`

Implementation overview covering:

- `mamba_snapshot.py`
- scheduler integration
- tokenizer manager integration
- HTTP route layer

### `troubleshooting.md`

Common failure modes and current workarounds.

## Archived Material

Historical or speculative docs now live under
[`.archive/`](.archive/AGENTS.md).

Anything in that folder is historical reference only and should not be used as
the production source of truth.

## Notes

- This docs set covers the Engram-added statefulness layer.
- Standard SGLang serving APIs are documented elsewhere.
- If an active doc conflicts with code, trust the implementation and update the
  doc.
