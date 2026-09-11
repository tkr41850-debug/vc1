from __future__ import annotations

from llms.proxy.usage import UsageTracker, extract_usage


def test_extract_usage_all_dialects():
    assert extract_usage(
        "responses", {"usage": {"input_tokens": 4, "output_tokens": 2}}
    ) == (4, 2, None, None)
    assert extract_usage(
        "messages",
        {
            "usage": {
                "input_tokens": 4,
                "output_tokens": 2,
                "cache_read_input_tokens": 3,
            }
        },
    ) == (4, 2, 3, None)
    assert extract_usage(
        "chat",
        {"usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}},
    ) == (4, 2, None, None)
    assert extract_usage("responses", {}) == (None, None, None, None)
    assert extract_usage("responses", {"usage": None}) == (None, None, None, None)


def test_extract_usage_details_shapes():
    assert extract_usage(
        "responses",
        {
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "input_tokens_details": {"cached_tokens": 7},
                "output_tokens_details": {"reasoning_tokens": 2},
            }
        },
    ) == (10, 5, 7, 2)
    assert extract_usage(
        "chat",
        {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 7},
                "completion_tokens_details": {"reasoning_tokens": 2},
            }
        },
    ) == (10, 5, 7, 2)


def test_tracker_aggregates_per_key_and_model():
    t = UsageTracker()
    t.record("ak-a", "m1", 4, 2, 1, 0)
    t.record("ak-a", "m1", 1, 1, 1, 1)
    t.record("ak-a", "m2", None, None)
    snap = t.snapshot()
    a = snap["keys"]["ak-a"]
    assert a["requests"] == 3
    assert a["input_tokens"] == 5
    assert a["output_tokens"] == 3
    assert a["cached_tokens"] == 2
    assert a["reasoning_tokens"] == 1
    assert a["models"]["m1"] == {
        "requests": 2,
        "input_tokens": 5,
        "output_tokens": 3,
        "cached_tokens": 2,
        "reasoning_tokens": 1,
    }
    assert a["models"]["m2"]["requests"] == 1


def test_tracker_migrates_old_snapshots(tmp_path):
    import json

    old = {
        "keys": {
            "ak-a": {
                "requests": 1,
                "input_tokens": 4,
                "output_tokens": 2,
                "models": {
                    "m1": {"requests": 1, "input_tokens": 4, "output_tokens": 2}
                },
            }
        }
    }
    (tmp_path / "usage.json").write_text(json.dumps(old))
    t = UsageTracker()
    t.load_file(tmp_path)
    a = t.snapshot()["keys"]["ak-a"]
    assert a["cached_tokens"] == 0
    assert a["reasoning_tokens"] == 0
    assert a["models"]["m1"]["cached_tokens"] == 0


def test_tracker_persists_roundtrip(tmp_path):
    t = UsageTracker()
    t.record("ak-a", "m1", 4, 2)
    t.save_file(tmp_path)
    t2 = UsageTracker()
    t2.load_file(tmp_path)
    assert t2.snapshot() == t.snapshot()


def test_extract_usage_malformed_values_fail_open():
    # Malformed upstream usage never 500s the request: unparseable fields
    # come back None and record as request-only.
    assert extract_usage(
        "responses",
        {
            "usage": {
                "input_tokens": "lots",
                "output_tokens": None,
                "input_tokens_details": {"cached_tokens": "many"},
                "output_tokens_details": {"reasoning_tokens": [1]},
            }
        },
    ) == (None, None, None, None)
    assert extract_usage(
        "chat",
        {
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "prompt_tokens_details": "nope",
            }
        },
    ) == (4, 2, None, None)


def test_tracker_rekey_merges_attribution():
    t = UsageTracker()
    t.record("sk-old", "m1", 4, 2, 1, 0)
    t.record("sk-new", "m1", 1, 1, 0, 1)
    t.rekey("sk-old", "sk-new")
    snap = t.snapshot()["keys"]
    assert "sk-old" not in snap
    assert snap["sk-new"]["requests"] == 2
    assert snap["sk-new"]["input_tokens"] == 5
    assert snap["sk-new"]["cached_tokens"] == 1
    assert snap["sk-new"]["reasoning_tokens"] == 1


def _post(tc, affinity, path, body):
    from tests.conftest import TEST_HEADERS

    prefix = f"/{affinity}" if affinity else ""
    return tc.post(f"{prefix}/{path}", json=body, headers=TEST_HEADERS)


def test_usage_recorded_for_responses(app_client):
    from tests.conftest import TEST_SECRET

    tc, _ = app_client
    r = _post(
        tc,
        None,
        "v1/responses",
        {"model": "muse-spark-1.3-contributor-free", "input": "hi"},
    )
    assert r.status_code == 200
    usage = tc.app.state.usage.snapshot()["keys"][TEST_SECRET]
    assert usage["requests"] == 1
    assert usage["input_tokens"] == 4
    assert usage["output_tokens"] == 2
    assert usage["cached_tokens"] == 0
    assert usage["reasoning_tokens"] == 0


def test_usage_recorded_for_chat_and_messages(app_client):
    from tests.conftest import TEST_SECRET

    tc, _ = app_client
    assert (
        _post(
            tc,
            None,
            "v1/chat/completions",
            {
                "model": "mimo-v2.5-free",
                "messages": [{"role": "user", "content": "hi"}],
            },
        ).status_code
        == 200
    )
    assert (
        _post(
            tc,
            None,
            "v1/messages",
            {
                "model": "claude-haiku-4-5",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
            },
        ).status_code
        == 200
    )
    snap = tc.app.state.usage.snapshot()["keys"][TEST_SECRET]
    assert snap["requests"] == 2
    assert snap["input_tokens"] == 8
    assert snap["output_tokens"] == 4


def test_stream_counts_request_without_tokens(app_client, mock_upstream):
    from tests.conftest import TEST_SECRET

    tc, _ = app_client
    _, seen_dict = mock_upstream
    seen_dict["mode"] = "stream"
    r = _post(
        tc,
        None,
        "v1/responses",
        {"model": "muse-spark-1.3-contributor-free", "input": "hi", "stream": True},
    )
    assert r.status_code == 200
    snap = tc.app.state.usage.snapshot()["keys"][TEST_SECRET]
    assert snap["requests"] == 1
    assert snap["input_tokens"] == 0


def test_usage_attributed_to_secret_not_affinity(app_client):
    # Two requests with different unauthenticated ak- prefixes but the same
    # sk- header aggregate under one usage key.
    from tests.conftest import TEST_SECRET

    tc, _ = app_client
    assert _post(tc, "ak-a", "v1/responses", {"input": "hi"}).status_code == 200
    assert _post(tc, "ak-b", "v1/responses", {"input": "hi"}).status_code == 200
    snap = tc.app.state.usage.snapshot()["keys"]
    assert list(snap) == [TEST_SECRET]
    assert snap[TEST_SECRET]["requests"] == 2


def test_rejected_requests_record_nothing(app_client):
    tc, _ = app_client
    # ak- prefix alone is not auth; wrong-scheme and unknown sk- also 401
    assert tc.post("/ak-nope/v1/responses", json={"input": "hi"}).status_code == 401
    assert tc.post("/v1/responses", json={"input": "hi"}).status_code == 401
    assert (
        tc.post(
            "/v1/responses",
            json={"input": "hi"},
            headers={"Authorization": "Bearer ak-nope"},
        ).status_code
        == 401
    )
    assert (
        tc.post(
            "/v1/responses",
            json={"input": "hi"},
            headers={"Authorization": "Bearer sk-nope"},
        ).status_code
        == 401
    )
    assert tc.app.state.usage.snapshot() == {"keys": {}}
