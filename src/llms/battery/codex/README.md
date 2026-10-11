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
| `test_cx_single_image_attach` | `-i` image attach (input_image parts) | red 1x1 PNG attached; reply ATTACH-IMG-1 iff image received | stdout ends ATTACH-IMG-1 (NO-IMAGE = dropped part) |
| `test_cx_single_review_diff` | apply_patch grammar + bare-patch rewrap | create patch_target.txt via Begin/Add grammar, reply DONE-PATCH-1 | file bytes `b"BASE-LINE\n"`, no SyntaxError/Unexpected token |
| `test_cx_single_sandbox_workspace` | sandbox modes (tool-list shaping per mode) | same write prompt under `--sandbox read-only` then `workspace-write` | RO: no file + DONE-SANDBOX-RO; WW: bytes + DONE-SANDBOX-WW |
| `test_cx_single_stdin_vs_arg` | prompt ingestion (instructions vs input mapping) | arg-form marker vs `echo ... | codex exec -` piped marker | each run ends with its own marker, never the other's |
| `test_cx_single_fork_session` | `exec fork` (doc-sourced) | fork turn-1 `--json` thread, read fork.txt in child | FORKED-FORK-SEED-5H1N, file bytes match |
| `test_cx_single_json_structured` | `--output-schema` (doc-sourced; violation-tempted fix) | schema + prose-tempting prompt | last line parses to exactly `{"ok": "schema-ok"}` |
| `test_cx_single_ephemeral` | `--ephemeral` (doc-sourced) | ephemeral echo, then `resume --last` | first ends EPHEMERAL-OK; resume lacks marker or errors |
| `test_cx_single_output_last_message` | `-o` file write (doc-sourced) | reply LASTMSG-OK with -o path | out file contains LASTMSG-OK |
| `test_cx_single_search_flag` | `--search` (doc-sourced) | reply SEARCH-FLAG-OK with web-search on | stdout ends SEARCH-FLAG-OK |

## Multi-turn

See `multi_turn.md`.
