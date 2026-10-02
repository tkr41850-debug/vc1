# Token attribution across compaction — investigation closeout (not-our-bug)

Symptom (user-reported): statusline shows 0 after first compaction;
autocompaction never triggers. Timeboxed to one session per plan Task 7.

## Evidence

1. The statusline (`~/.claude/cc-statusline.sh`) reads token counts from
   Claude Code's **transcript JSONL** (`message.usage` blocks), never from
   the gateway's `usage.json`. Gateway-side attribution cannot move the
   statusline number directly — only the usage blocks we return to the
   client can, and those are upstream bytes (passthrough) or faithful
   re-emits of upstream frames (translate legs).
2. `scripts/stream_usage_probe.py` passes against a live gateway:
   `requests=1 in=380 out=30 cached=320` — streamed turns attribute.
3. `scripts/claude_probe.py` passes (turn-one, turn-two, tool-ok):
   multi-turn conversations attribute across turns.
4. Post-compaction turns are ordinary turns with summary text. The
   `4c5ebba` commit already locks that compacted shapes flow on every
   leg (Claude summary + redacted thinking + document blocks; Codex
   thread-id continuity keeps the session). Nothing in the pipeline
   branches on compaction.
5. Usage is keyed by secret key, not session: even a genuinely new
   post-compaction session attributes its tokens to the same key entry.
   A new session only changes `prompt_cache_key`, never attribution.
6. One real zero-token path exists but does not match the symptom: a
   200 response with a bare (non-SSE) JSON body on the anonymous
   synthesize path parses to zero deltas, folds to `incomplete` with
   `None` tokens, and records request-only. Real Zen answers
   `stream:true` with SSE (verified in the fold probe: realistic SSE
   yields `in: 9000 out: 50`), so this path needs a non-SSE 200 from
   upstream — an upstream-shape anomaly, not compaction.

## Conclusion

No gateway-side mechanism drops attribution at the compaction
boundary. If the statusline reads 0 it is reading transcript usage
blocks (client-side accounting), and autocompaction triggering is a
pure client decision off the same blocks. Close as not-our-bug unless
a repro shows our returned usage blocks themselves at 0 on a
post-compaction turn — that would reopen against the translate legs.
