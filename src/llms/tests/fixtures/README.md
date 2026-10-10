# Stream fixtures (captured, not handcrafted)

- `responses_incomplete.sse` — live `POST /zen/v1/responses` (stream:true,
  title-shaped, 32-token cap), Sep 2026. Exercises created/in_progress,
  output_item.added, reasoning item, and the `response.incomplete` +
  usage path. 8.5KB complete stream.
- `responses_rich_prefix.sse` — prefix of a live agent turn from the
  mitm capture (`response.completed` tail and >2KB reasoning blobs cut).
  Exercises item added/done, content_part, text + function_call deltas,
  and a `{"type":"ping"}` keepalive the gateway ignores. Truncated
  streams must still terminate cleanly (trailing `StreamDone`).

Regenerate: point a streaming request at the gateway or Zen with
`curl -N`, saving raw bytes. Keep fixtures small (<10KB): drop
>2KB lines (reasoning blobs) and cap at ~14 data events.

- `claude-schema.jsonl` — the installed claude CLI's live tool
  declaration (single-line JSON: `{"body": {"tools": [...], "model":
  ...}}`), captured via the repo stub. Pins the claude rename rows
  (shell/read/write/edit onto Bash/Read/Write/Edit) in
  test_claude_capture.py, test_compat_table.py, test_client_tools.py
  (notice shape) and test_stream_translate.py (streaming renames).

Regenerate: `python scripts/refresh_claude_schema.py` (stub server,
zero quota — nothing leaves the machine). Re-run when the CLI
version changes the declared toolset; the tests assert the count, so
drift fails LOUDLY instead of testing a stale declaration.
