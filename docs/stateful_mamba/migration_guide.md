# Migration Guide: Adopt Engram Snapshot Statefulness

This guide explains how to move from ordinary Mamba serving to the current
Engram snapshot workflow in this repository.

It reflects the implementation that exists today. It does not describe the
older Phase 1 / Phase 2 planning model.

## What Changes

Before Engram usage:

- requests are served normally
- clients resend full context on each turn
- no snapshot catalog is managed by the client

With Engram usage:

- the server is started with snapshot persistence enabled
- clients save snapshots after important turns
- clients track `conversation_id` plus `turn_number` or `branch_name`
- later turns can use restore-and-generate flows instead of replaying all prior
  tokens

## Step 1: Enable Snapshot Persistence On The Server

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

## Step 2: Decide Your Snapshot Keys

You need a durable selection strategy.

Recommended default:

- `conversation_id`: your stable chat/session key
- `turn_number`: your main-line turn counter
- `branch_name`: only when intentionally branching

Do not build your client around a legacy “restore by snapshot id only” mental
model. The live API is selector-based.

## Step 3: Save Snapshots After Important Turns

Typical pattern:

1. generate through `/v1/chat/completions`
2. keep the resulting `rid`
3. call `/save_snapshot` with that `rid`
4. persist your own `conversation_id` and `turn_number`

Example:

```python
save_resp = requests.post(
    "http://localhost:30000/save_snapshot",
    json={
        "rid": rid,
        "conversation_id": "chat-123",
        "turn_number": 3,
    },
    timeout=30,
)
save_body = save_resp.json()
```

## Step 4: Inspect The Snapshot Catalog

Use:

- `/list_snapshots` to inspect everything in a conversation
- `/get_snapshot_info` to inspect one selected snapshot

Remember:

- empty conversations currently return `snapshots: []`
- for info/delete flows, one of `turn_number` or `branch_name` is effectively
  required in practice

## Step 5: Migrate Later Turns To Restore-And-Generate

For stateful continuation:

1. tokenize only the new user turn
2. call `/restore_snapshot` with `create_new_request=true`
3. pass `continuation_ids` and `max_new_tokens`
4. read `output_text` from the response

Example:

```python
restore_resp = requests.post(
    "http://localhost:30000/restore_snapshot",
    json={
        "conversation_id": "chat-123",
        "create_new_request": True,
        "continuation_ids": continuation_ids,
        "max_new_tokens": 80,
    },
    timeout=120,
)
restore_body = restore_resp.json()
```

## Python Surface Changes

The current direct surface on `s` is:

- `save_snapshot`
- `list_snapshots`
- `get_snapshot_info`
- `restore_snapshot`
- `delete_snapshot`

The current high-level manager is:

```python
from sglang.snapshot import SnapshotManager

manager = SnapshotManager(runtime.endpoint)
```

Available manager methods:

- `list_conversation`
- `get_info`
- `restore`
- `delete`

## Behavior Differences Worth Noting

- `/save_snapshot` can fail but still return HTTP `200`
- `/restore_snapshot` returns HTTP `500` when `success=false`
- restore-only really works best with a live `rid`
- conversation-only restore can report state availability without injecting into
  a live request

## Migration Checklist

- server launched with `--enable-snapshot-persistence`
- snapshot directory is writable
- client owns a stable `conversation_id`
- client tracks `turn_number` or `branch_name`
- client uses the same tokenizer/template for restore-and-generate
- client checks response JSON for `success`

## Next Reading

- [http_api_spec.md](http_api_spec.md) for the canonical wire contract
- [user_guide.md](user_guide.md) for the current high-level usage model
- [troubleshooting.md](troubleshooting.md) for the most common failure modes
