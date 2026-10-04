"""Main acquisition workflow. No Panda/libfranka calls outside the controller."""

from __future__ import annotations

import logging
import time
from contextlib import ExitStack

import numpy as np

from control.collection.dataset import open_dataset
from control.collection.devices import make_cameras, make_microphone, make_vr
from control.collection.recording import EpisodeRecorder
from control.collection.vr import EEMapper, VRResampler
from control.control_diagnostics import ControlDiagnostics
from control.reactive_desk_client import ScenePublisher
from control.robotic_arm_controller import RoboticArmControler
from control.util.pose import quat_angle_xyzw
from control.util.robot import finish_stream, handle_gripper, reset_robot
from control.util.timing import FixedRate

logger = logging.getLogger(__name__)


def run_collection(config, *, dry_run=False, max_steps=0):
    from control._panda.fake import FakeBackend

    recorder = None
    with ExitStack() as stack:
        diagnostics = stack.enter_context(ControlDiagnostics(config))
        arm = stack.enter_context(
            RoboticArmControler(config=config, backend=FakeBackend() if dry_run else None)
        )
        cameras = make_cameras(config, dry_run)
        stack.callback(cameras.close)
        vr = make_vr(config, dry_run)
        stack.callback(vr.stop)
        microphone = (
            make_microphone(config, dry_run)
            if config["dataset"].get("enable_logging", True)
            else None
        )
        if microphone is not None:
            stack.callback(microphone.stop)
        scene = ScenePublisher(config, dry_run)
        stack.callback(scene.close)
        if config["dataset"].get("enable_logging", True):
            dataset, root = open_dataset(config)
            recorder = EpisodeRecorder(dataset, root, config, microphone)
            stack.callback(recorder.close)
        if config.get("robot", {}).get("move_to_start", True):
            arm.move_to_start()
        arm.start_stream()
        resampler = VRResampler(float(config.get("vr", {}).get("stale_after_s", 0.5)))
        mapper = EEMapper(config.get("vr", {}))
        state = arm.get_state()
        mapper.reset(state)
        if diagnostics.enabled:
            diagnostics.event("stream_start", hold_position=state.ee_position.tolist())
        rate = FixedRate(config["control"]["frequency_hz"])
        steps = 0
        last_axis_ns = 0
        failed = False
        try:
            while max_steps <= 0 or steps < max_steps:
                tick_ns = rate.tick()
                diagnostics.begin_tick(steps, tick_ns, rate.overruns)
                reset_requested = False
                command_sent = False
                state = arm.get_state()
                diagnostics.mark("state_read")
                raw_sample = vr.latest
                diagnostics.mark("vr_read")
                now = time.monotonic_ns()
                sample = resampler.sample(raw_sample, now)
                if arm.gripper_busy:
                    # The controller is paused for the gripper; discard unsent VR motion.
                    mapper.reset(state)
                position, quaternion = mapper.map(sample, state, now)
                diagnostics.mark("resample_and_map")
                scene.publish(state.ee_position, now, config["dataset"]["instruction"])
                diagnostics.mark("scene_publish")
                # Idle pose corrections are not operator actions that start an episode.
                changed = sample.enabled and (
                    np.linalg.norm(position - state.ee_position) > 1e-6
                    or quat_angle_xyzw(quaternion, state.ee_quaternion_xyzw) > 1e-6
                )
                if not arm.gripper_busy:
                    arm.send_ee_target(position, quaternion)
                    command_sent = True
                    gripper_changed, last_axis_ns = handle_gripper(
                        arm, sample, state, now, last_axis_ns, config
                    )
                    changed = changed or gripper_changed
                diagnostics.mark("command_and_gripper")
                if recorder is not None:
                    if changed and not recorder.active:
                        recorder.start()
                    recorder.trace(state, sample, position, quaternion)
                    pair = cameras.get_frames()
                    if pair is not None:
                        if now - min(pair.front_ns, pair.wrist_ns) > int(
                            config["camera"].get("max_age_s", 0.2) * 1e9
                        ):
                            raise RuntimeError("Camera frame is stale")
                        recorder.frame(
                            state,
                            pair,
                            arm.get_tactile_images()
                            if config.get("tactile", {}).get("enabled", False)
                            else (None, None),
                        )
                    expired = (
                        recorder.active
                        and now - recorder.started_ns
                        > config["dataset"].get("max_duration_s", 3600) * 1e9
                    )
                    if sample.save or sample.discard or expired:
                        recorder.finish(
                            arm.get_state(),
                            save=not sample.discard,
                            success=True if sample.save else None,
                        )
                        reset_requested = True
                diagnostics.mark("recording_and_camera")
                diagnostics.record(
                    raw=raw_sample,
                    sample=sample,
                    state=state,
                    mapper=mapper,
                    position=position,
                    quaternion=quaternion,
                    command_sent=command_sent,
                )
                if reset_requested:
                    reset_robot(arm)
                    state = arm.get_state()
                    mapper.reset(state)
                    if diagnostics.enabled:
                        diagnostics.event("stream_reset", hold_position=state.ee_position.tolist())
                steps += 1
        except KeyboardInterrupt:
            diagnostics.event("interrupt", step=steps)
            logger.info("Stopping acquisition")
        except BaseException as exc:
            failed = True
            diagnostics.event(
                "error",
                step=steps,
                error_type=type(exc).__name__,
                message=str(exc),
                timings_ms=dict(diagnostics.timings_ms),
            )
            raise
        finally:
            diagnostics.event("loop_end", ticks=steps, overruns=rate.overruns, failed=failed)
            finish_stream(arm)
            if recorder is not None:
                recorder.finish(arm.get_state(), save=not failed)
        logger.info("Acquisition finished: %s ticks, %s overruns", steps, rate.overruns)
    if recorder is not None and config.get("audio", {}).get("vad_enabled", False) and not dry_run:
        audio_files = (recorder.root / "audio").glob("episode_*.wav")
        if recorder.dataset.meta.total_episodes > 0 and any(path.is_file() for path in audio_files):
            from data_analysis.preprocess_vad import compute_vad_for_dataset

            compute_vad_for_dataset(recorder.root)
        else:
            logger.info("Skipping VAD: no saved episode audio")
    return {"ticks": steps, "overruns": rate.overruns}
