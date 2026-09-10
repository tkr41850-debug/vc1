from __future__ import annotations

from llms.proxy.affinity import bucket_for, parse_affinity_prefix


def test_parse_affinity_prefix():
    assert parse_affinity_prefix("/ak-team1/v1/responses") == (
        "ak-team1",
        "/v1/responses",
    )
    assert parse_affinity_prefix("/ak-a_1-2/v1/chat/completions") == (
        "ak-a_1-2",
        "/v1/chat/completions",
    )
    assert parse_affinity_prefix("/ak-team1/messages") == ("ak-team1", "/messages")


def test_non_affinity_paths_pass_through():
    assert parse_affinity_prefix("/v1/responses") == (None, "/v1/responses")
    assert parse_affinity_prefix("/healthz") == (None, "/healthz")
    assert parse_affinity_prefix("/bk-team1/v1/responses") == (
        None,
        "/bk-team1/v1/responses",
    )
    assert parse_affinity_prefix("/ak-/v1/responses") == (None, "/ak-/v1/responses")
    assert parse_affinity_prefix("/") == (None, "/")


def test_bucket_stable_and_bounded():
    first = bucket_for("ak-team1", "muse-spark-1.3-contributor-free")
    assert bucket_for("ak-team1", "muse-spark-1.3-contributor-free") == first
    assert 0 <= first < 1024


def test_bucket_varies_with_inputs():
    buckets = {bucket_for(f"ak-team{i}", f"model-{i}") for i in range(20)}
    assert len(buckets) > 1
    assert bucket_for(None, "model-a") == bucket_for("", "model-a")
    assert bucket_for(None, "model-a") == bucket_for(None, "MODEL-A")


def test_bucket_spreads_uniformly():
    seen = {bucket_for(f"ak-{i}", "m") for i in range(200)}
    assert len(seen) > 150


def test_bucket_mixes_secret_key_with_affinity():
    # Same affinity + model but different sk- keys spread across buckets,
    # affinities still spread under one sk-, and omitting the secret is stable.
    seen = {bucket_for("ak-team1", "m", 1024, f"sk-{i}") for i in range(50)}
    assert len(seen) > 1
    aff = {bucket_for(f"ak-{i}", "m", 1024, "sk-same") for i in range(50)}
    assert len(aff) > 1
    assert bucket_for("ak-team1", "m") == bucket_for("ak-team1", "m", 1024, None)
    assert bucket_for("ak-team1", "m", 1024, "sk-x") == bucket_for(
        "ak-team1", "m", 1024, "sk-x"
    )
