import copy
import time

import numpy as np
import pytest

from control.config import load_config
from control.inference.loop import run_inference
from control.inference.policy import ActionPlan, validate_metadata


class SlowPolicy:
    server_metadata = {
        "action_space": "ee",
        "action_fps": 30.0,
        "action_horizon": 16,
        "action_dim": 7,
    }

    def __init__(self):
        self.closed = False
        self.calls = 0

    def infer(self, obs):
        self.calls += 1
        time.sleep(0.1)
        assert "observation/ee_pose" in obs
        pose = obs["observation/ee_pose"].copy()
        pose[0] += 0.001
        return {"actions": np.tile(np.r_[pose, 1.0], (16, 1))}

    def close(self):
        self.closed = True


def test_async_inference_preserves_servo_rate():
    cfg = copy.deepcopy(load_config())
    policy = SlowPolicy()
    start = time.monotonic()
    stats = run_inference(cfg, dry_run=True, max_steps=60, policy=policy)
    assert stats["ticks"] == 60 and policy.closed
    assert 2 <= stats["requests"] <= 3
    assert time.monotonic() - start < 1.1


def test_action_timing_and_metadata_validation():
    plan = ActionPlan(0, np.arange(16 * 7).reshape(16, 7), np.zeros(6))
    assert plan.at(0, 30)[0] is None
    assert plan.at(33_333_334, 30)[1] == 0
    assert plan.at(100_000_000, 30)[1] == 2
    assert plan.at(1_000_000_000, 30)[0] is None
    assert not validate_metadata(SlowPolicy.server_metadata, load_config())
    with pytest.raises(ValueError):
        validate_metadata({**SlowPolicy.server_metadata, "action_fps": 100}, load_config())
