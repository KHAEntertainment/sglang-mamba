# Engram Troubleshooting

This guide covers the failure modes most likely to confuse integrators working
with the current Engram snapshot implementation.

For the exact request and response contract, use
[http_api_spec.md](http_api_spec.md).

## Snapshot Save Failed But HTTP Status Was 200

Symptom:

- `/save_snapshot` returned HTTP `200`
- the body still says `success: false`

Cause:

- the route currently returns HTTP `200` for handler-level save failures

What to do:

- always parse the response body
- treat `success` as authoritative
- log the returned message

## Restore Failed And Returned HTTP 500

Symptom:

- `/restore_snapshot` returned HTTP `500`
- the body still includes a structured JSON failure payload

Cause:

- the route maps application-level restore failure to HTTP `500`

What to do:

- do not rely on `raise_for_status()` alone
- parse the body and inspect `success` plus `message`

## Restore-Only Did Not Inject State

Symptom:

- restore-only call succeeds
- response says state is available for future requests
- expected in-place restore did not happen

Cause:

- restore-only is effectively a live-request operation
- without a resolvable live `rid`, the server may locate snapshot state but not
  inject it into a running request

What to do:

- use a live `rid` for restore-only flows
- for later-turn inference, prefer restore-and-generate with
  `create_new_request=true`

## `get_snapshot_info` Or `delete_snapshot` Fails With Selectors That Look Optional

Symptom:

- request model suggests `turn_number` and `branch_name` are optional
- runtime still fails when both are omitted

Cause:

- the lower snapshot path logic effectively needs one of those selectors

What to do:

- always send either `turn_number` or `branch_name`

## Empty Snapshot List

Symptom:

- `/list_snapshots` returns `success: true`
- `snapshots` is an empty list

Cause:

- current behavior for nonexistent or empty conversations is a successful empty
  response, not an application error

What to do:

- treat `[]` as “no snapshots found yet”
- verify that your `conversation_id` matches the one used during save

## Restore-And-Generate Fails Validation

Symptom:

- `/restore_snapshot` fails before useful generation starts

Common causes:

- neither `rid` nor `conversation_id` was provided
- `create_new_request=true` without `continuation_ids`
- `create_new_request=true` without `max_new_tokens`
- `continuation_ids` were provided while `create_new_request=false`

What to do:

- validate the request body on the client side before sending it

## Snapshot Exists But Generation Still Fails

Possible causes:

- model mismatch between the saved snapshot and the running server
- missing `fill_ids` in snapshot metadata
- Mamba pool unavailable
- context-length constraints exceeded

What to do:

- log the full restore failure body
- confirm the same model is running
- confirm you are using the correct snapshot selectors

## Tokenization Mismatch In Restore-And-Generate

Symptom:

- restore succeeds
- generated answer is clearly wrong or detached from the prior state

Cause:

- `continuation_ids` were not encoded the same way as the original server-side
  prompt formatting

What to do:

- use the same tokenizer as the server model
- use the correct chat template
- use `add_generation_prompt=True` for chat-style continuation encoding

## Python API Confusion

Current reality:

- the direct methods on `s` are selector-based
- `SnapshotManager` is constructed from `runtime.endpoint`
- the current manager surface is `list_conversation`, `get_info`, `restore`,
  and `delete`

If a document or example tells you to use the older engine-based manager
constructor, per-request snapshot enable hooks, speculative persistence helpers,
or bare snapshot-id restore flows, that document is stale or archival.

## Still Stuck?

Check these next:

- [http_api_spec.md](http_api_spec.md)
- [user_guide.md](user_guide.md)
- [architecture.md](architecture.md)
