# Engram User Guide

This guide explains how to use the Engram snapshot and statefulness layer added
to this repository.

It focuses on the live API surface that exists today. For the canonical
technical contract, see [http_api_spec.md](http_api_spec.md).

## What Engram Adds

Engram adds snapshot-oriented state management for Mamba-capable SGLang
servers. The core ideas are:

- save model state after a turn
- inspect the saved snapshot catalog
- restore prior state for later use
- restore and immediately continue generation

This layer is separate from standard SGLang chat/completions APIs. Think of it
as a statefulness extension on top of normal request serving.

## Current Public Surfaces

There are two public entry points in the current codebase:

### 1. Direct state methods on `s`

`ProgramState` exposes these methods:

- `s.save_snapshot(...)`
- `s.list_snapshots(...)`
- `s.get_snapshot_info(...)`
- `s.restore_snapshot(...)`
- `s.delete_snapshot(...)`

These methods use conversation and snapshot selectors, not bare `snapshot_id`
restore flows.

### 2. `SnapshotManager(runtime.endpoint)`

The high-level manager wraps the HTTP-backed runtime endpoint:

```python
from sglang.snapshot import SnapshotManager

manager = SnapshotManager(runtime.endpoint)
```

The current manager surface is:

- `list_conversation(conversation_id)`
- `get_info(conversation_id, turn_number=None, branch_name=None)`
- `restore(rid, conversation_id, turn_number=None, branch_name=None)`
- `delete(conversation_id, turn_number=None, branch_name=None)`

## Enabling Snapshot Support

The server must be launched with snapshot persistence enabled:

```bash
python -m sglang.launch_server \
  --model-path /workspace/models/granite-4.0-h-small \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-snapshot-persistence \
  --snapshot-dir /workspace/snapshots \
  --trust-remote-code
```

Recommended production shape:

```bash
python -m sglang.launch_server \
  --model-path /workspace/models/granite-4.0-h-small \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-snapshot-persistence \
  --snapshot-dir /workspace/snapshots \
  --admin-api-key "$SGLANG_ADMIN_KEY" \
  --trust-remote-code
```

## Direct API Usage

### Save a snapshot

```python
@function
def capture_turn(s):
    s += gen("answer", max_tokens=64)

    snapshot_id = s.save_snapshot(
        conversation_id="chat-123",
        turn_number=2,
    )
    return snapshot_id
```

Notes:

- `conversation_id` defaults to the current session id if omitted
- `turn_number` is optional in the direct method signature, but supplying it is
  the safest way to make later lookup deterministic

### List snapshots for a conversation

```python
@function
def inspect(s):
    snapshots = s.list_snapshots(conversation_id="chat-123")
    return snapshots
```

Each result is a metadata dictionary produced by the snapshot layer.

### Get metadata for one snapshot

```python
@function
def inspect_one(s):
    return s.get_snapshot_info(
        conversation_id="chat-123",
        turn_number=2,
    )
```

Use either `turn_number` or `branch_name`.

### Restore snapshot state

```python
@function
def restore_previous(s):
    return s.restore_snapshot(
        conversation_id="chat-123",
        turn_number=2,
    )
```

This method delegates to the HTTP restore flow. The current implementation is
selector-based and returns a result dictionary; it is not the older
snapshot-id-only restore model.

### Delete a snapshot

```python
@function
def cleanup(s):
    deleted = s.delete_snapshot(
        conversation_id="chat-123",
        turn_number=2,
    )
    return deleted
```

## Working Model

The selector model is:

- `conversation_id` groups snapshots
- `turn_number` selects a turn on the main line
- `branch_name` selects a named branch

When choosing selectors:

- use `turn_number` for linear conversation checkpoints
- use `branch_name` for alternate branches
- avoid relying on invented client-side `snapshot_id` restore semantics

## HTTP-Oriented Workflow

The most common Engram integration pattern is:

1. generate normally through `/v1/chat/completions`
2. save a snapshot using the original `rid`
3. track `conversation_id` and `turn_number` on the client side
4. later call `/restore_snapshot` in restore-and-generate mode with
   `continuation_ids`

For the exact wire contract, request rules, and status code behavior, use
[http_api_spec.md](http_api_spec.md).

## Best Practices

- Treat `conversation_id` as your durable grouping key.
- Store your own turn numbering rather than assuming the server will infer it in
  every path.
- Keep the same tokenizer and chat template between original generation and
  restore-and-generate flows.
- Always inspect the JSON body for `success`, especially on restore failures.
- Treat the returned metadata dictionaries as implementation-defined records, not
  a frozen schema.

## What This Guide Does Not Cover

This guide does not try to replace the upstream SGLang docs for:

- standard chat/completion serving
- model loading beyond snapshot prerequisites
- general runtime usage unrelated to Engram's added statefulness layer

For lower-level endpoint details and current quirks, use
[http_api_spec.md](http_api_spec.md).
