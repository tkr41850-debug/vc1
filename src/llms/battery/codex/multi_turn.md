# Codex battery — multi-turn specs

All in `test_multi_turn.py`. Seed a unique marker turn 1, recall it
turn 2 via the stated resume mechanism. Same CODEX_HOME across turns.

| id | mechanism | turns | assertion |
|----|-----------|-------|-----------|
| `test_cx_multi_resume_last` | `exec resume --last` | write marker to marker.txt, resume and read back | stdout has RESUMED-MARKER-RESUME-LAST-7Q2T, file bytes match |
| `test_cx_multi_resume_session_id` | `exec resume <id>` (id parsed from turn-1 `--json`) | same, explicit id | stdout has RESUMED-MARKER-SID-9X4P, file bytes match |
| `test_cx_multi_tool_chain` | one exec, dependent steps | write 1..10 → shell sum → read → double → answer | session-sum=110, numbers/sum/doubled files byte-exact (55/110) |
| `test_cx_multi_thread_continuation` | `resume --last` thread | write marker, resume and echo CONTINUED-\<marker\> | CONTINUED-THREAD-CONT-3K8R, file bytes match |
| `test_cx_multi_previous_response_id` | direct API, no CLI | turn 2 carries turn 1's response id | ECHOED-CHAIN-5M2N; both legs share one upstream ses_ |
| `test_cx_multi_review_modes` | top-level `review --uncommitted` ×2 (doc-sourced; no prompt — flag conflicts with positional) | committed repo + dirty file, review twice | both outputs mention review.txt |

Note: `resume --last` and explicit-id resume intentionally overlap
(different markers) — they pin distinct proxy paths (alias lookup vs
thread-id mapping). Merging would save quota but lose that split.
