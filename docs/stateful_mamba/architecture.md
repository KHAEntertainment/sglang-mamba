<!-- ENGRAM_MODIFIED — Engram snapshot architecture overview -->
# Engram Architecture

This document describes the implementation that exists today for the Engram
snapshot/statefulness layer in this repository.

It replaces the older speculative architecture that described nonexistent
registry and serializer modules.

## Overview

Engram adds snapshot persistence and restore flows for Mamba-capable SGLang
servers. The design is centered on five HTTP endpoints and the scheduler path
that backs them.

At a high level:

1. the client generates normally through SGLang
2. the client saves a snapshot for a request or conversation
3. snapshot metadata and state are stored on disk
4. later requests query, restore, or delete that stored state

## Main Components

### `mamba_snapshot.py`

`python/sglang/srt/snapshot/mamba_snapshot.py` is the core storage layer.

It is responsible for:

- persisting snapshot state to disk
- loading snapshots from disk
- listing snapshots for a conversation
- loading metadata
- deleting saved snapshot files

The metadata model is conversation-oriented and turn-oriented. It is not a
legacy in-memory registry of named snapshot objects.

### Scheduler Integration

`python/sglang/srt/managers/scheduler.py` is the behavioral center of the
feature.

It handles:

- save requests
- snapshot listing
- metadata lookup
- restore-only logic
- restore-and-generate logic
- delete requests

Important architectural detail:

- the scheduler is where live-request lookup happens via `rid`
- the scheduler is also where conversation selectors are resolved into snapshot
  files
- this is why `rid` and `conversation_id` have distinct roles

### Tokenizer Manager

`python/sglang/srt/managers/tokenizer_manager.py` forwards snapshot requests to
the scheduler and returns the result back to the HTTP layer.

It also detokenizes `output_ids` into `output_text` for successful
restore-and-generate responses.

### HTTP Server

`python/sglang/srt/entrypoints/http_server.py` exposes the public snapshot
routes:

- `/save_snapshot`
- `/list_snapshots`
- `/get_snapshot_info`
- `/restore_snapshot`
- `/delete_snapshot`

This layer is intentionally thin. It mostly:

- validates request bodies through the dataclasses
- forwards to the tokenizer manager
- translates results into HTTP responses

### Python Client Surfaces

There are two Python entry points:

#### Direct methods on `ProgramState`

Implemented in `python/sglang/lang/interpreter.py`:

- `save_snapshot`
- `list_snapshots`
- `get_snapshot_info`
- `restore_snapshot`
- `delete_snapshot`

These are selector-based methods using `conversation_id`, `turn_number`, and
`branch_name`.

#### `SnapshotManager`

Implemented in `python/sglang/snapshot.py`:

- `SnapshotManager(runtime.endpoint)`
- `list_conversation`
- `get_info`
- `restore`
- `delete`

This manager is a higher-level wrapper around the HTTP-backed runtime endpoint.

## Data Model

The live system is built around snapshot metadata records, not a persistent
Python `Snapshot` object exposed as the public API.

Core fields include:

- `conversation_id`
- `turn_number`
- `branch_name`
- `timestamp`
- `token_count`
- `model_name`

The metadata can also contain implementation details such as:

- `mamba_pool_idx`
- `req_pool_idx`
- `layer_config`
- `fill_ids`

Clients should treat these as implementation-defined metadata dictionaries, not
as a frozen public schema.

## Save Flow

1. client calls `/save_snapshot`
2. HTTP layer validates and forwards the request
3. tokenizer manager forwards to the scheduler
4. scheduler tries to resolve a live request through `rid`
5. if needed, scheduler can fall back to WARM-tier state
6. snapshot manager writes metadata and state to disk
7. response returns `success`, `snapshot_id`, and `message`

Architectural caveat:

- `rid` is the real live-request lookup key
- `conversation_id` is the grouping key for snapshot persistence

## Restore Flow

There are two different restore paths.

### Restore-Only

1. client sends selectors plus a live `rid`
2. scheduler loads snapshot state
3. scheduler injects state into the live request if it can resolve that request

If no live request is available, the server may report that snapshot state is
available for future use rather than performing an in-place restore.

### Restore-And-Generate

1. client sends `conversation_id`
2. client sets `create_new_request=true`
3. client supplies `continuation_ids` and `max_new_tokens`
4. scheduler loads the selected snapshot, often the latest one
5. scheduler validates model compatibility and metadata requirements
6. scheduler creates the new request and runs generation
7. tokenizer manager decodes `output_ids` into `output_text`

## Response Behavior

The HTTP layer currently preserves some non-ideal historical behavior:

- `/save_snapshot` can fail while still returning HTTP `200`
- `/restore_snapshot` returns HTTP `500` for application-level failure

This is why the JSON body is the real source of truth for success/failure.

## Current Boundaries

This architecture does not provide a public, stable, high-level object model
for:

- speculative persistence-helper APIs from the older docs set
- direct snapshot-id-only restore/delete APIs
- a separate `snapshot_registry.py` or `snapshot_serializer.py`

Those older concepts belonged to the speculative documentation model, not the
current implementation.

## Related Reading

- [http_api_spec.md](http_api_spec.md)
- [user_guide.md](user_guide.md)
- [migration_guide.md](migration_guide.md)
- [troubleshooting.md](troubleshooting.md)
