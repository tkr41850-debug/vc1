from __future__ import annotations

from llms.proxy.usage import UsageTracker, extract_usage


def test_extract_usage_all_dialects():
    assert extract_usage(
        "responses", {"usage": {"input_tokens": 4, "output_tokens": 2}}
    ) == (4, 2)
    assert extract_usage(
        "messages", {"usage": {"input_tokens": 4, "output_tokens": 2}}
    ) == (4, 2)
    assert extract_usage(
        "chat",
        {"usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}},
    ) == (4, 2)
    assert extract_usage("responses", {}) == (None, None)
    assert extract_usage("responses", {"usage": None}) == (None, None)


def test_tracker_aggregates_per_key_and_model():
    t = UsageTracker()
    t.record("ak-a", "m1", 4, 2)
    t.record("ak-a", "m1", 1, 1)
    t.record("ak-a", "m2", None, None)
    snap = t.snapshot()
    a = snap["keys"]["ak-a"]
    assert a["requests"] == 3
    assert a["input_tokens"] == 5
    assert a["output_tokens"] == 3
    assert a["models"]["m1"] == {
        "requests": 2,
        "input_tokens": 5,
        "output_tokens": 3,
    }
    assert a["models"]["m2"]["requests"] == 1


def test_tracker_persists_roundtrip(tmp_path):
    t = UsageTracker()
    t.record("ak-a", "m1", 4, 2)
    t.save_file(tmp_path)
    t2 = UsageTracker()
    t2.load_file(tmp_path)
    assert t2.snapshot() == t.snapshot()


def _post(tc, key, path, body):
    return tc.post(f"/{key}/{path}", json=body)


def test_usage_recorded_for_responses(app_client):
    from tests.conftest import TEST_KEY

    tc, _ = app_client
    r = _post(
        tc,
        TEST_KEY,
        "v1/responses",
        {"model": "muse-spark-1.3-contributor-free", "input": "hi"},
    )
    assert r.status_code == 200
    usage = tc.app.state.usage.snapshot()["keys"][TEST_KEY]
    assert usage["requests"] == 1
    assert usage["input_tokens"] == 4
    assert usage["output_tokens"] == 2


def test_usage_recorded_for_chat_and_messages(app_client):
    from tests.conftest import TEST_KEY

    tc, _ = app_client
    assert (
        _post(
            tc,
            TEST_KEY,
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
            TEST_KEY,
            "v1/messages",
            {
                "model": "claude-haiku-4-5",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
            },
        ).status_code
        == 200
    )
    snap = tc.app.state.usage.snapshot()["keys"][TEST_KEY]
    assert snap["requests"] == 2
    assert snap["input_tokens"] == 8
    assert snap["output_tokens"] == 4


def test_stream_counts_request_without_tokens(app_client, mock_upstream):
    from tests.conftest import TEST_KEY

    tc, _ = app_client
    _, seen_dict = mock_upstream
    seen_dict["mode"] = "stream"
    r = _post(
        tc,
        TEST_KEY,
        "v1/responses",
        {"model": "muse-spark-1.3-contributor-free", "input": "hi", "stream": True},
    )
    assert r.status_code == 200
    snap = tc.app.state.usage.snapshot()["keys"][TEST_KEY]
    assert snap["requests"] == 1
    assert snap["input_tokens"] == 0


def test_rejected_requests_record_nothing(app_client):
    tc, _ = app_client
    assert tc.post("/ak-nope/v1/responses", json={"input": "hi"}).status_code == 401
    assert tc.post("/v1/responses", json={"input": "hi"}).status_code == 401
    assert tc.app.state.usage.snapshot() == {"keys": {}}
