"""Metadata-checked asynchronous action chunks; requests never pace robot control."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionPlan:
    observation_ns: int
    actions: np.ndarray
    initial_ee: np.ndarray

    def at(self, now_ns, fps):
        index = (now_ns - self.observation_ns) * fps / 1e9 - 1
        if index < 0 or index >= len(self.actions):
            return None, int(np.floor(index))
        i = int(index)
        return self.actions[i].copy(), i


def validate_metadata(metadata, config):
    cfg = config["inference"]
    space = metadata.get("action_space")
    expected = 8 if space == "joint" else 7
    force = metadata.get("rlt_protocol") == "pytorch-rl-token-v1"
    if force:
        expected = 7
        if metadata.get("group") == "C":
            raise ValueError("Force RLT C needs a synchronized finger-history model adapter")
        if metadata.get("rlt_actor_protocol") != "pytorch-force-rlt-actor-v1" or not metadata.get(
            "actor_sha256"
        ):
            raise ValueError("Force RLT actor metadata is invalid")
    if (
        space not in ("ee", "joint")
        or space != cfg["action_space"]
        or float(metadata.get("action_fps", 0)) != float(cfg["action_fps"])
        or int(metadata.get("action_horizon", 0)) != int(cfg["action_horizon"])
        or int(metadata.get("action_dim", 0)) != expected
    ):
        raise ValueError("Policy metadata action space, dimension, horizon or fps mismatch")
    return force


def build_observation(state, pair, prompt, space, *, tactile=None, step=0):
    result = {
        "observation/exterior_image_1_left": pair.front.copy(),
        "observation/wrist_image_left": pair.wrist.copy(),
        "observation/gripper_position": np.array(
            [state.gripper.commanded_open_ratio], dtype=np.float32
        ),
        "prompt": prompt,
    }
    key = "ee_pose" if space == "ee" else "joint_position"
    result[f"observation/{key}"] = (
        state.ee_pose.vector if space == "ee" else state.joint_positions.astype(np.float32)
    )
    if tactile is not None:
        left, right = tactile
        if left is None or right is None:
            raise ValueError("Tactile policy inputs are missing")
        result.update(
            {
                "observation/gripper_image_left": left.copy(),
                "observation/gripper_image_right": right.copy(),
            }
        )
    result.update(
        monotonic_ts=np.array(state.sampled_monotonic_ns / 1e9, dtype=np.float64),
        O_T_TCP=state.end_effector_pose.copy(),
        step_id=step,
    )
    return result


def parse_response(response, observation_ns, state, metadata):
    force = metadata.get("rlt_protocol") == "pytorch-rl-token-v1"
    actions = np.asarray(response.get("proposed_q" if force else "actions", []), dtype=np.float64)
    shape = (int(metadata["action_horizon"]), int(metadata["action_dim"]))
    if actions.shape != shape or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite policy action chunk {shape}, got {actions.shape}")
    reference = float(response.get("ref_obs_ts", float("nan")))
    if force and (
        not np.isfinite(reference)
        or response.get("group") != metadata.get("group")
        or response.get("actor_sha256") != metadata.get("actor_sha256")
        or abs(reference - observation_ns / 1e9) > 1e-6
    ):
        raise ValueError("Force RLT response reference mismatch")
    return ActionPlan(observation_ns, actions, state.ee_pose.vector)


class PolicyWorker:
    def __init__(self, policy):
        self.policy = policy
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy")
        self.future = None
        self.generation = 0

    def submit(self, observation, state):
        if self.future is not None:
            return False
        generation = self.generation

        def request():
            response = self.policy.infer(observation)
            return generation, parse_response(
                response, state.sampled_monotonic_ns, state, self.policy.server_metadata
            )

        self.future = self.pool.submit(request)
        return True

    def take(self):
        if self.future is None or not self.future.done():
            return None
        generation, result = self.future.result()
        self.future = None
        return result if generation == self.generation else None

    def invalidate(self):
        self.generation += 1

    def close(self):
        self.policy.close()
        self.pool.shutdown(wait=True, cancel_futures=True)
