from __future__ import annotations

import threading
import time
from collections.abc import Callable


class BucketTable:
    def __init__(
        self,
        num_buckets: int = 1024,
        num_slots: int = 8,
        slot_cooldown_s: float = 60.0,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.num_buckets = num_buckets
        self.num_slots = num_slots
        self.slot_cooldown_s = slot_cooldown_s
        self._now = now
        self._lock = threading.Lock()
        self._slots = [b % num_slots for b in range(num_buckets)]
        self._cooldown_until = [0.0] * num_buckets

    def slot_for(self, bucket: int) -> int:
        with self._lock:
            return self._slots[bucket % self.num_buckets]

    def set_num_slots(self, num_slots: int) -> bool:
        num_slots = max(1, int(num_slots))
        with self._lock:
            if num_slots == self.num_slots:
                return False
            self.num_slots = num_slots
            self._slots = [b % num_slots for b in range(self.num_buckets)]
            return True

    def cooldown_remaining(self, bucket: int) -> float:
        return max(0.0, self._cooldown_until[bucket % self.num_buckets] - self._now())

    def note_ratelimited(self, bucket: int, retry_after: float | None = None) -> int:
        idx = bucket % self.num_buckets
        with self._lock:
            current = self._slots[idx]
            if self.num_slots > 1:
                self._slots[idx] = (current + 1) % self.num_slots
            wait = self.slot_cooldown_s
            if retry_after is not None:
                wait = max(wait, retry_after)
            self._cooldown_until[idx] = self._now() + wait
            return self._slots[idx]
