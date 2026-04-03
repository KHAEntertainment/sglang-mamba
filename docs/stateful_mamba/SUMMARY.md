# Engram Docs Summary

This directory documents the snapshot and statefulness layer added in this
repository.

The key change in this pass is simple: the docs now treat
[http_api_spec.md](http_api_spec.md) as the canonical technical source for the
Engram HTTP contract, and the rest of the active docs are aligned to that
reality.

## Active Docs

### `api_guide.md`

User-friendly API documentation for most developers integrating Engram.

Includes:

- quick start
- authentication
- common workflows
- multi-language examples
- practical error handling

### `http_api_spec.md`

Canonical technical draft for:

- `/save_snapshot`
- `/list_snapshots`
- `/get_snapshot_info`
- `/restore_snapshot`
- `/delete_snapshot`

### `user_guide.md`

Current usage guide for:

- direct selector-based methods on `s`
- `SnapshotManager(runtime.endpoint)`

### `migration_guide.md`

Current migration path from ordinary serving to snapshot-aware serving.

### `examples.md`

Examples aligned to the live implementation.

### `architecture.md`

Implementation overview centered on:

- `mamba_snapshot.py`
- scheduler handling
- tokenizer manager forwarding
- HTTP routes

### `troubleshooting.md`

Current failure modes and practical debugging notes.

## Archived Docs

Historical or speculative material is now kept under `.archive/`.

That includes the old API reference and any support docs that fundamentally
described an API surface or architecture that does not match the current code.

## What This Summary Does Not Claim

This summary does not claim that:

- every historical document was correct
- the snapshot API is fully polished for end users yet
- the current behavior is perfectly consistent

Instead, this docs pass aims to make the active docs trustworthy, explicit, and
useful for technical integration work.
