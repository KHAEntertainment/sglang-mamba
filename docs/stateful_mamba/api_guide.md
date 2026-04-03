# Engram API Guide

User-friendly guide to Engram's snapshot and statefulness API.

This is the document intended for most integrators. If you need the exact
wire-level contract or current implementation quirks, use
[http_api_spec.md](http_api_spec.md).

Engram is built on the existing SGLang API surface and keeps OpenAI-compatible
serving behavior intact. For most teams, that means the API will feel familiar:
you still use the standard SGLang and OpenAI-shaped generation endpoints, and
Engram adds snapshot/statefulness features on top rather than replacing the
existing model-serving workflow with something brand new.

## What Engram Adds

Engram sits on top of standard SGLang serving and adds one thing SGLang does
not provide on its own: durable model state for Mamba-family models.

In practice that means you can:

- save conversation state after a turn
- inspect what snapshots exist for a conversation
- restore previous state
- restore and immediately continue generation from that saved state

You still use the standard SGLang/OpenAI-compatible APIs for normal generation.
Engram adds a snapshot layer on top of that workflow.

## Familiar By Design

Engram is not a brand new API platform with a separate mental model.

It keeps two important compatibility promises:

- standard OpenAI-compatible generation flows still work
- the snapshot layer is built off the existing SGLang server model, so teams
  already familiar with SGLang will recognize the shape immediately

In practice:

- you still generate through the standard chat/completions routes
- existing SGLang integration patterns stay relevant
- Engram adds extra endpoints and stateful workflows when you want persistent
  model state

## Quick Start

### 1. Start the server with snapshot persistence enabled

```bash
python -m sglang.launch_server \
  --model-path ibm-granite/granite-4.0-h-tiny \
  --enable-snapshot-persistence \
  --snapshot-dir ./snapshots \
  --mamba-scheduler-strategy no_buffer \
  --port 30000
```

Recommended for any remote or shared environment:

```bash
python -m sglang.launch_server \
  --model-path ibm-granite/granite-4.0-h-tiny \
  --enable-snapshot-persistence \
  --snapshot-dir ./snapshots \
  --admin-api-key "$SGLANG_ADMIN_KEY" \
  --mamba-scheduler-strategy no_buffer \
  --port 30000
```

### 2. Generate normally

Use the normal OpenAI-compatible chat API first. Keep the returned request id
(`rid`) if your client surface exposes it.

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite-4.0-h-tiny",
    "messages": [
      {"role": "user", "content": "My favorite color is blue. Remember that."}
    ],
    "max_tokens": 128
  }'
```

### 3. Save a snapshot

The most reliable save flow uses the original `rid` plus your own
`conversation_id` and `turn_number`.

```bash
curl http://localhost:30000/save_snapshot \
  -H "Content-Type: application/json" \
  -d '{
    "rid": "req-123",
    "conversation_id": "session-1",
    "turn_number": 1
  }'
```

### 4. Restore and continue from saved state

For later turns, restore the saved state and append only the new user turn as
`continuation_ids`.

```json
POST /restore_snapshot
{
  "conversation_id": "session-1",
  "create_new_request": true,
  "continuation_ids": [1, 2, 3, 4],
  "max_new_tokens": 128
}
```

If successful, the response can include `output_text`.

## Authentication

Snapshot routes are `ADMIN_OPTIONAL`, which means the required token depends on
server configuration.

- no `api_key` and no `admin_api_key`: snapshot routes are open
- only `api_key`: snapshot routes require `api_key`
- only `admin_api_key`: snapshot routes require `admin_api_key`
- both keys configured: snapshot routes require `admin_api_key`

Header format:

```http
Authorization: Bearer <token>
```

Example:

```bash
curl http://localhost:30000/list_snapshots \
  -H "Authorization: Bearer $SGLANG_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "session-1"}'
```

## Core Concepts

### `rid`

`rid` is the live request identifier. It matters most when:

- saving state from a live request
- restoring state back into a live request

### `conversation_id`

`conversation_id` is your durable grouping key for snapshots. You should treat
it as the stable id for a chat or session.

### `turn_number`

`turn_number` is the simplest way to select snapshots on the main line of a
conversation.

### `branch_name`

`branch_name` lets you create and retrieve alternate branches from the same
conversation.

## Endpoints At A Glance

| Endpoint | Method | What it does |
|----------|--------|--------------|
| `/save_snapshot` | `POST` | Saves state for a live request or recent WARM-tier state |
| `/list_snapshots` | `POST` | Lists snapshots for a conversation |
| `/get_snapshot_info` | `POST` | Fetches metadata for one selected snapshot |
| `/restore_snapshot` | `POST` | Restores state, or restores and immediately continues generation |
| `/delete_snapshot` | `POST` | Deletes one selected snapshot |

## Common Workflows

### Workflow 1: Save a checkpoint after an important turn

Use this when you want to revisit a point in the conversation later.

```bash
curl http://localhost:30000/save_snapshot \
  -H "Content-Type: application/json" \
  -d '{
    "rid": "req-123",
    "conversation_id": "session-1",
    "turn_number": 3
  }'
```

### Workflow 2: Inspect what snapshots exist

```bash
curl http://localhost:30000/list_snapshots \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "session-1"}'
```

Typical response:

```json
{
  "success": true,
  "snapshots": [
    {
      "conversation_id": "session-1",
      "turn_number": 1,
      "branch_name": null,
      "timestamp": 1712345678.0,
      "token_count": 128,
      "model_name": "/workspace/models/granite-4.0-h-tiny"
    }
  ]
}
```

### Workflow 3: Get details for one snapshot

```bash
curl http://localhost:30000/get_snapshot_info \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "session-1",
    "turn_number": 1
  }'
```

### Workflow 4: Restore and generate the next turn

This is the signature Engram flow.

```python
import requests
from transformers import AutoTokenizer

BASE_URL = "http://localhost:30000"
MODEL_PATH = "ibm-granite/granite-4.0-h-tiny"

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
formatted = tok.apply_chat_template(
    [{"role": "user", "content": "What is my favorite color?"}],
    tokenize=False,
    add_generation_prompt=True,
)
continuation_ids = tok.encode(formatted, add_special_tokens=False)

resp = requests.post(
    f"{BASE_URL}/restore_snapshot",
    json={
        "conversation_id": "session-1",
        "create_new_request": True,
        "continuation_ids": continuation_ids,
        "max_new_tokens": 64,
    },
    timeout=120,
)

body = resp.json()
if not body.get("success"):
    raise RuntimeError(body)

print(body.get("output_text"))
```

### Workflow 5: Delete an old snapshot

```bash
curl http://localhost:30000/delete_snapshot \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "session-1",
    "turn_number": 1
  }'
```

## JavaScript Example

```ts
const response = await fetch("http://localhost:30000/list_snapshots", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Authorization: `Bearer ${process.env.SGLANG_ADMIN_KEY}`,
  },
  body: JSON.stringify({ conversation_id: "session-1" }),
});

const body = await response.json();
if (!body.success) {
  throw new Error(body.message ?? "Snapshot list failed");
}

console.log(body.snapshots);
```

## Python API Surface

If you are working inside SGLang program code, the current direct methods on
`s` are:

- `save_snapshot`
- `list_snapshots`
- `get_snapshot_info`
- `restore_snapshot`
- `delete_snapshot`

If you want a higher-level wrapper around the HTTP-backed runtime endpoint:

```python
from sglang import RuntimeEndpoint
from sglang.snapshot import SnapshotManager

endpoint = RuntimeEndpoint("http://localhost:30000")
manager = SnapshotManager(endpoint)
```

Current manager methods:

- `list_conversation`
- `get_info`
- `restore`
- `delete`

## Error Handling

There are two important quirks to know up front:

### `/save_snapshot` can fail with HTTP `200`

Always inspect the JSON body:

```python
body = save_resp.json()
if not body.get("success"):
    raise RuntimeError(body)
```

### `/restore_snapshot` can fail with HTTP `500` but still return useful JSON

Do not rely on `raise_for_status()` alone for restore flows. Parse the body and
inspect `success` and `message`.

## Tips For Smooth Integrations

- Use your own durable `conversation_id`.
- Track `turn_number` explicitly for main-line conversations.
- Use `branch_name` only when you truly want branching behavior.
- Keep tokenizer and chat-template behavior consistent between original
  generation and restore-and-generate.
- Treat metadata dictionaries as implementation-defined rather than frozen.

## When To Use The Technical Spec Instead

Use [http_api_spec.md](http_api_spec.md) when you need:

- the exact request validation rules
- the precise status code quirks
- the distinction between typed-optional and effectively-required fields
- the implementation-backed contract for every endpoint
