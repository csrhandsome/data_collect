"""100 Hz EE executor, 30 Hz policy action timebase and optional VR takeover."""

import logging
import time
from contextlib import ExitStack

import numpy as np

from control.collection.devices import make_cameras, make_vr
from control.collection.vr import EEMapper, VRResampler
from control.inference.policy import PolicyWorker, build_observation, validate_metadata
from control.recording_writer import FreshCameraPair
from control.robot_state import EEPose
from control.robotic_arm_controller import RoboticArmControler
from control.util.pose import quat_angle_xyzw
from control.util.robot import finish_stream, handle_gripper
from control.util.timing import FixedRate

logger = logging.getLogger(__name__)


def run_inference(config, *, dry_run=False, max_steps=0, policy=None):
    from control._panda.fake import FakeBackend
    from control.inference.client import PolicyClient

    cfg = config["inference"]
    with ExitStack() as stack:
        client = policy or PolicyClient(
            host=cfg["host"],
            port=int(cfg["port"]),
            connect_timeout_s=cfg.get("connect_timeout_s", 10),
            request_timeout_s=cfg.get("request_timeout_s", 5),
        )
        stack.callback(client.close)
        force = validate_metadata(client.server_metadata, config)
        if force and not dry_run:
            raise ValueError("Force RLT real execution needs a calibrated TCP safety projector")
        arm = stack.enter_context(
            RoboticArmControler(config=config, backend=FakeBackend() if dry_run else None)
        )
        cameras = make_cameras(config, dry_run)
        stack.callback(cameras.close)
        vr = make_vr(config, dry_run) if cfg.get("enable_vr_takeover", False) else None
        if vr is not None:
            stack.callback(vr.stop)
        if cfg.get("move_to_start", False):
            arm.move_to_start()
        arm.start_stream()
        worker = PolicyWorker(client)
        stack.callback(worker.close)
        resampler, mapper = VRResampler(), EEMapper(config.get("vr", {}))
        rate = FixedRate(config["control"]["frequency_hz"])
        gate = FreshCameraPair()
        plan = None
        steps = 0
        last_request_ns = 0
        takeover = False
        requests = 0
        last_axis_ns = 0
        horizon_ns = round(1e9 * cfg["replan_every_steps"] / cfg["action_fps"])
        limit = max_steps or int(cfg.get("max_steps", 0))
        try:
            while limit <= 0 or steps < limit:
                rate.tick()
                state = arm.get_state()
                pair = cameras.get_frames()
                raw_sample = vr.latest if vr else None
                now = time.monotonic_ns()
                if (
                    pair is None
                    or now - min(pair.front_ns, pair.wrist_ns)
                    > config["camera"].get("max_age_s", 0.2) * 1e9
                ):
                    raise RuntimeError("Inference camera frame is unavailable or stale")
                sample = resampler.sample(raw_sample, now) if vr else None
                enabled = sample is not None and sample.enabled
                if enabled != takeover:
                    worker.invalidate()
                    plan = None
                    mapper.reset()
                    takeover = enabled
                if enabled:
                    position, quat = mapper.map(sample, state, now)
                    if not arm.gripper_busy:
                        arm.send_ee_target(position, quat)
                        _, last_axis_ns = handle_gripper(
                            arm, sample, state, now, last_axis_ns, config
                        )
                else:
                    result = worker.take()
                    if result is not None:
                        plan = result
                    action, _ = plan.at(now, cfg["action_fps"]) if plan else (None, 0)
                    if plan and now - plan.observation_ns > cfg.get("max_plan_age_s", 0.6) * 1e9:
                        action = None
                    if not arm.gripper_busy:
                        if action is None:
                            arm.hold()
                        else:
                            desired = (
                                EEPose.from_vector(action[:6])
                                if cfg["action_space"] == "ee"
                                else arm.pose_from_joints(action[:7])
                            )
                            if np.linalg.norm(desired.xyz_m - state.ee_position) > cfg.get(
                                "max_translation_step_m", 0.03
                            ) or quat_angle_xyzw(
                                desired.quaternion_xyzw(), state.ee_quaternion_xyzw
                            ) > cfg.get("max_rotation_step_rad", 0.15):
                                raise ValueError(
                                    "Policy target exceeds configured measured pose guard"
                                )
                            arm.send_ee_target(desired.xyz_m, desired.quaternion_xyzw())
                            if not force:
                                ratio = float(action[-1])
                                if not np.isfinite(ratio) or not 0 <= ratio <= 1:
                                    raise ValueError("Policy gripper ratio outside 0..1")
                                if abs(ratio - state.gripper.commanded_open_ratio) > 0.05:
                                    arm.set_gripper(ratio, wait=False)
                    if now - last_request_ns >= horizon_ns and gate.accept(
                        pair.front_ns, pair.wrist_ns
                    ):
                        observation = build_observation(
                            state,
                            pair,
                            config["dataset"]["instruction"],
                            cfg["action_space"],
                            tactile=arm.get_tactile_images()
                            if config.get("tactile", {}).get("enabled", False)
                            else None,
                            step=steps,
                        )
                        if worker.submit(observation, state):
                            requests += 1
                            last_request_ns = now
                steps += 1
        except KeyboardInterrupt:
            logger.info("Stopping inference")
        finally:
            finish_stream(arm)
    return {"ticks": steps, "requests": requests, "overruns": rate.overruns}
