# Claude battery — multi-turn specs

All in `test_multi_turn.py`. Seed a unique marker turn 1 (isolated
workspace per test), recall turn 2+ via `--continue` / `--resume`.

| id | mechanism | turns | assertion |
|----|-----------|-------|-----------|
| `test_cl_continue_recall` | `-p --continue` | echo MARKER-7F3A, continue and repeat | turn-2 stdout stripped == marker exactly |
| `test_cl_resume_session` | `-p --resume <session-id>` (id from turn-1 json) | echo MARKER-RESUME-51C2, resume and repeat | same session_id both turns, marker echoed |
| `test_cl_file_write_read` | `--continue`, Write then Read | write HELLO-BATTERY-42 (no newline), read back | file bytes exact + turn-2 stdout stripped == bytes |
| `test_cl_file_modify` | `--continue` ×2, Write → Read+Edit → Read | write LINE-ONE-1111, append second line, verify | file bytes `b"LINE-ONE-1111\nLINE-TWO-2222"`, turn-3 echo exact |
