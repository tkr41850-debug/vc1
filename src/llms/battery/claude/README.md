# Claude battery — spec index

Single-turn specs live in `test_single_turn.py`, multi-turn specs in
`test_multi_turn.py`. All run the messages leg (CLI-native haiku).
See `battery/README.md` for the harness.

Run: `just battery claude/`

## Single-turn

| id | exercises | prompt | assertion |
|----|-----------|--------|-----------|
| `test_cl_single_print` | print mode, no-tool echo (TextBlock round-trip) | Reply exactly CLAUDE-PRINT-OK | stdout stripped == marker; stream-json rerun: all lines parse, zero tool_use |
| `test_cl_single_bash` | Bash tool round-trip (shell→claude-bash rewrite) | echo CLAUDE-BASH-OK via Bash, reply verbatim | marker in stdout; stream-json: one Bash tool_use with exact input.command + tool_result |
| `test_cl_single_write` | Write tool byte-exact create | write 15 bytes CLAUDE-WRITE-OK, no newline, reply DONE | file bytes exact, DONE in stdout |
| `test_cl_single_read` | Read tool exact content | read pre-seeded cl-read.txt, reply exact content | stdout stripped == CLAUDE-READ-OK (marker not in prompt — proves Read ran) |
| `test_cl_single_edit` | Edit tool exact replace | replace V1 with V2 in cl-edit.txt, reply DONE | file bytes `b"CLAUDE-EDIT-V2"`, DONE in stdout |
| `test_cl_single_multi_tool` | parallel multi-tool turn | Write cl-multi.txt + Bash echo in one turn | file bytes + marker; stream-json: one turn, two tool_use {Write, Bash} |
| `test_cl_single_output_format` | text vs stream-json envelope parity | same exact-echo prompt, both formats | text: stripped == marker; stream-json: all lines parse, deltas concat == marker, message_stop present |
| `test_cl_single_allowed_tools` | client-side tool gating | Bash prompt under allow-Bash then allow-Read-only | allow: Bash tool_use + marker; deny: non-empty output, zero Bash tool_use |
| `test_cl_single_json_structured` | `--json-schema` (doc-sourced) | BLUEBIRD extract + commentary-tempting prompt | payload structured_output == {"answer": "BLUEBIRD"} |
| `test_cl_single_partial_messages` | `--include-partial-messages` (doc-sourced) | exact echo under stream-json + verbose | stream_event/content_block_delta present, marker in output |
| `test_cl_single_session_id_and_name` | `--session-id` + `-n` (doc-sourced) | exact echo under json with uuid sid | session_id == sid, marker in payload |
| `test_cl_single_fork_session` | `--resume` + `--fork-session` (doc-sourced) | repeat turn-1 marker in fork | new session_id, marker echoed |
| `test_cl_single_stdin_prompt` | prompt via stdin pipe (doc-sourced) | `-` positional fed exact-echo prompt | stdout stripped == STDIN-PIPE-OK |
| `test_cl_single_input_format_stream_json` | `--input-format=stream-json` (doc-sourced) | piped user message, exact echo | result FMT-IN-OK in stream-json output |
| `test_cl_single_task_subagent` | Task subagent delegation | subagent replies SUBAGENT-OK, report verbatim | SUBAGENT-OK in stdout |
| `test_cl_single_hook_pretooluse` | PreToolUse command hook via `--settings` | hook appends HOOK-FIRED on Bash, echo marker | HOOK-BASH-OK in stdout, hook.log == HOOK-FIRED |
| `test_cl_single_no_space_strip` | regression pin: tool args keep interior spaces | Bash echo prompt under allow-Read x3 | spaceless token never appears (fails loud + saves evidence) |
| `test_cl_single_mcp_tool` | MCP stdio tool via `--mcp-config` + pre-grant | marker_echo from battery-probe server, relay verbatim | stdout stripped == MCP-MARKER-OK (skips when uv/fastmcp missing) |
| `test_cl_single_plugin_skill` | plugin skill via `--plugin-dir` (doc-sourced) | invoke marker-skill, relay verbatim | stdout stripped == PLUGIN-MARKER-OK |

Dropped by adversarial verify (not written — fix noted):
- `cl-followup-tool-output`: turn-1 token appears verbatim in its own
  prompt, so the model can echo it without ever calling Bash. Fix:
  assert stream-json Bash tool_use on turn 1 (mirroring
  `test_cl_single_bash`), or make the echoed value unpredictable
  (command substitution).

## Multi-turn

See `multi_turn.md`.
