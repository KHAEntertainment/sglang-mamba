# Engram Examples

These examples are intentionally aligned to the current implementation rather
than the older speculative snapshot API.

For the exact wire contract, use [http_api_spec.md](http_api_spec.md).

## Example 1: Save And Inspect Snapshots In A Function

```python
import sglang as sgl
from sglang import function, gen


@function
def one_turn(s, prompt):
    s += prompt
    s += gen("answer", max_tokens=64)

    snapshot_id = s.save_snapshot(
        conversation_id="chat-123",
        turn_number=1,
    )

    snapshots = s.list_snapshots(conversation_id="chat-123")
    info = s.get_snapshot_info(
        conversation_id="chat-123",
        turn_number=1,
    )

    return {
        "snapshot_id": snapshot_id,
        "snapshots": snapshots,
        "info": info,
    }
```

## Example 2: Use `SnapshotManager` With A Runtime Endpoint

```python
from sglang import RuntimeEndpoint
from sglang.snapshot import SnapshotManager

endpoint = RuntimeEndpoint("http://localhost:30000")
manager = SnapshotManager(endpoint)

snapshots = manager.list_conversation("chat-123")
info = manager.get_info("chat-123", turn_number=2)
deleted = manager.delete("chat-123", turn_number=1)
```

If you already have a runtime object with an endpoint, the intended pattern is:

```python
manager = SnapshotManager(runtime.endpoint)
```

## Example 3: Save Through Chat, Restore Through HTTP

This is the most realistic Engram integration pattern.

```python
import requests
from transformers import AutoTokenizer

BASE_URL = "http://localhost:30000"
MODEL_PATH = "/workspace/models/granite-4.0-h-small"

tok = AutoTokenizer.from_pretrained(MODEL_PATH)

# After a normal /v1/chat/completions call, keep the request id.
rid = "req-123"

save_resp = requests.post(
    f"{BASE_URL}/save_snapshot",
    json={
        "rid": rid,
        "conversation_id": "chat-123",
        "turn_number": 1,
    },
    timeout=30,
)
save_body = save_resp.json()
if not save_body.get("success"):
    raise RuntimeError(save_body)

formatted = tok.apply_chat_template(
    [{"role": "user", "content": "What is my favorite color?"}],
    tokenize=False,
    add_generation_prompt=True,
)
continuation_ids = tok.encode(formatted, add_special_tokens=False)

restore_resp = requests.post(
    f"{BASE_URL}/restore_snapshot",
    json={
        "conversation_id": "chat-123",
        "create_new_request": True,
        "continuation_ids": continuation_ids,
        "max_new_tokens": 64,
    },
    timeout=120,
)
restore_body = restore_resp.json()
if not restore_body.get("success"):
    raise RuntimeError(restore_body)

print(restore_body.get("output_text"))
```

## Example 4: Branch-Oriented Snapshot Lookup

```python
metadata = manager.get_info(
    "chat-123",
    branch_name="alternate-ending",
)

deleted = manager.delete(
    "chat-123",
    branch_name="alternate-ending",
)
```

## Example 5: Minimal TypeScript Client

```ts
const baseUrl = "http://localhost:30000";

const response = await fetch(`${baseUrl}/list_snapshots`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ conversation_id: "chat-123" }),
});

const body = await response.json();
if (!body.success) {
  throw new Error(body.message ?? "list failed");
}

console.log(body.snapshots);
```

## Notes

- These examples use the selector-based API that exists today.
- They do not use the old speculative persistence helpers, per-request snapshot
  enable hooks, or the obsolete `SnapshotManager` constructor pattern that took
  an engine object directly.
- They assume the server is already running with `--enable-snapshot-persistence`.
