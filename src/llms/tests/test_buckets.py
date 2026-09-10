from __future__ import annotations

from llms.proxy.buckets import BucketTable


def test_initial_mapping_spreads_buckets():
    table = BucketTable(num_buckets=1024, num_slots=8)
    assert table.slot_for(0) == 0
    assert table.slot_for(7) == 7
    assert table.slot_for(8) == 0
    assert table.slot_for(1025) == table.slot_for(1)


def test_ratelimit_advances_slot():
    table = BucketTable(num_buckets=16, num_slots=4)
    assert table.slot_for(3) == 3
    assert table.note_ratelimited(3) == 0
    assert table.slot_for(3) == 0


def test_ratelimit_wraps_around():
    table = BucketTable(num_buckets=4, num_slots=2)
    table.note_ratelimited(1)
    assert table.slot_for(1) == 0
    table.note_ratelimited(1)
    assert table.slot_for(1) == 1


def test_single_slot_stays_put():
    table = BucketTable(num_buckets=4, num_slots=1)
    assert table.note_ratelimited(2) == 0
    assert table.slot_for(2) == 0


def test_cooldown_honors_retry_after():
    now = [100.0]
    table = BucketTable(
        num_buckets=4, num_slots=2, slot_cooldown_s=60.0, now=lambda: now[0]
    )
    assert table.cooldown_remaining(1) == 0.0
    table.note_ratelimited(1, retry_after=120.0)
    assert table.cooldown_remaining(1) == 120.0
    now[0] = 220.0
    assert table.cooldown_remaining(1) == 0.0


def test_other_buckets_unaffected():
    table = BucketTable(num_buckets=8, num_slots=4)
    table.note_ratelimited(2)
    assert table.slot_for(3) == 3
    assert table.cooldown_remaining(3) == 0.0
