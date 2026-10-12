# Codex battery — spec index

Single-turn specs live in `test_single_turn.py`, multi-turn specs in
`test_multi_turn.py`. All run the responses leg (spark model) through
the gateway. See `battery/README.md` for the harness.

Run: `just battery codex/`

## Single-turn

| id | exercises | prompt | assertion |
|----|-----------|--------|-----------|
| `test_cx_single_file_write` | file write via tools (`translate_to_client`, genuine write overlay) | Write exactly MARKER-WRITE-1 to write_plain.txt, reply DONE-WRITE-1 | file bytes `b"MARKER-WRITE-1\n"`, stdout ends DONE-WRITE-1 |
| `test_cx_single_shell_redirect_write` | shell-redirect write path (`_shell_to_cmd`, `_write_to_shell_redirect`) | printf REDIRECT-OK > redir.txt via shell tool, reply DONE-REDIR-1 | file bytes `b"REDIRECT-OK"` (empty = heredoc/stdin regression), stdout ends DONE-REDIR-1 |
| `test_cx_single_chained_multitool` | chained multi-tool turn (2+ function_call + history items) | echo CHAIN-A/B to a/b.txt, cat both, reply CHAIN-DONE | both files byte-exact, stdout ends CHAIN-DONE |
| `test_cx_single_image_attach` | `-i` image attach with vision-content assertion (stop sign) | attached stop-sign PNG (in-memory PIL); STOP never in prompt | ATTACH-IMG-1 + STOP in stdout — proves image bytes reached vision |
| `test_cx_single_review_diff` | apply_patch grammar + bare-patch rewrap | create patch_target.txt via Begin/Add grammar, reply DONE-PATCH-1 | file bytes `b"BASE-LINE\n"`, no SyntaxError/Unexpected token |
| `test_cx_single_sandbox_workspace` | sandbox modes (tool-list shaping per mode) | same write prompt under `--sandbox read-only` then `workspace-write` | RO: no file + DONE-SANDBOX-RO; WW: bytes + DONE-SANDBOX-WW |
| `test_cx_single_stdin_vs_arg` | prompt ingestion (instructions vs input mapping) | arg-form marker vs `echo ... | codex exec -` piped marker | each run ends with its own marker, never the other's |
| `test_cx_single_fork_session` | `exec fork` (doc-sourced) | fork turn-1 `--json` thread, read fork.txt in child | FORKED-FORK-SEED-5H1N, file bytes match |
| `test_cx_single_json_structured` | `--output-schema` (doc-sourced; violation-tempted fix) | schema + prose-tempting prompt | last line parses to exactly `{"ok": "schema-ok"}` |
| `test_cx_single_ephemeral` | `--ephemeral` (doc-sourced) | ephemeral echo, then `resume --last` | first ends EPHEMERAL-OK; resume lacks marker or errors |
| `test_cx_single_output_last_message` | `-o` file write (doc-sourced) | reply LASTMSG-OK with -o path | out file contains LASTMSG-OK |
| `test_cx_single_search_flag` | `--search` (doc-sourced) | reply SEARCH-FLAG-OK with web-search on | stdout ends SEARCH-FLAG-OK |
| `test_cx_single_mcp_tool` | MCP stdio tool via `codex mcp add` (doc-sourced) | marker_echo from battery-probe server, relay verbatim | CX-MCP-MARKER-OK in stdout (skips when uv/fastmcp missing; no pre-grant needed) |
| `test_cx_single_read_tool` | genuine `read` steer (read-via-shell-cat) | read seed_read.txt, reply exact contents (marker never in prompt) | marker in stdout — proves a real read, not an echo |
| `test_cx_single_edit_tool` | genuine `edit` (generic steer, no TO_CLIENT row) | replace EDIT-BEFORE with EDIT-AFTER, reply DONE-EDIT-1 | file bytes `b"EDIT-AFTER-8Q2M\n"`, stdout ends DONE-EDIT-1 |
| `test_cx_single_grep_tool` | content search (generic steer → shell grep/rg) | search *.txt for GREP-NEEDLE, reply FOUND-GREP-1 + full line verbatim | marker + GREP-NEEDLE-3X9P in stdout |
| `test_cx_single_glob_tool` | file finding (generic steer → shell find/ls) | find cx-glob-target.json under workspace, reply FOUND-GLOB-1 + path | marker + sub-nest-7 in stdout |
| `test_cx_single_question_direct` | `question` has no codex-plain route (notice bans it) | reply QUESTION-PATH-OK, no tools | stdout ends QUESTION-PATH-OK (pins no stall on banned tool) |
| `test_cx_single_webfetch_tool` | content fetch (generic steer → shell curl) | fetch example.com, reply FETCH-OK-1 + heading | marker + "Example Domain" in stdout |
| `test_cx_single_websearch_dispatch` | web-search dispatch: no-hang pin, not result-exact | search for Codex CLI docs, reply SEARCH-DISPATCH-OK | marker in stdout (results vary by day — only completion pinned) |
| `test_cx_single_view_image` | harness-native `view_image` dispatch (disk → vision) | view stop.png via image tool, reply VIEW-IMG-1 or NO-IMAGE | stdout ends VIEW-IMG-1 |
| `test_cx_single_spawn_agent` | harness-native `spawn_agent` + `wait_agent` delegation (ports cl-task) | subagent echoes SUBAGENT-OK (never in prompt), wait + relay | SUBAGENT-OK in stdout |
| `test_cx_single_explore_chain` | explore→edit→verify chain (traffic-dominant Read→Edit bigram) | read marker, swap V1→V2, re-read to verify, reply CHAIN-VERIFY-OK | file bytes `b"CHAIN-V2-6T2N\n"`, marker in stdout |
| `test_cx_single_request_user_input` | `request_user_input` has no exec route (Plan-mode only): no-stall pin | reply NOINPUT-PATH-OK, no tools | stdout ends NOINPUT-PATH-OK |

## Multi-turn

See `multi_turn.md`.
