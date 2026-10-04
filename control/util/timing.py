"""Monotonic deadlines shared by acquisition and inference."""

import math
import time


def positive_rate(value, name="frequency"):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


class FixedRate:
    def __init__(self, hz, *, clock=time.monotonic_ns, sleep=time.sleep):
        self.period_ns = round(1e9 / positive_rate(hz))
        self.clock, self.sleep = clock, sleep
        self.deadline = clock()
        self.overruns = 0

    def tick(self):
        now = self.clock()
        if now < self.deadline:
            self.sleep((self.deadline - now) / 1e9)
        now = self.clock()
        if now - self.deadline >= self.period_ns:
            self.overruns += 1
            self.deadline = now
        self.deadline += self.period_ns
        return now


def wait_ready(read, timeout_s):
    positive_rate(timeout_s, "timeout_s")
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return read()
        except (RuntimeError, ValueError) as exc:
            if time.monotonic() >= deadline:
                raise TimeoutError("Robot state unavailable") from exc
            time.sleep(0.01)
