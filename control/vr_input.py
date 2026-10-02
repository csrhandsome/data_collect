"""VR controller input reader via teleop_xr.

Provides two modes:
  - VRInputReader : in-process (thread).  Good for standalone testing.
  - VRInputProcess: out-of-process via multiprocessing.  Immune to GIL
    contention from camera / IK / panda threads in the main process.

Optionally applies lightweight pose interpolation on the right controller
while the deadman switch is engaged. This smooths single-frame WebXR jitter
before downstream IK / joint control consumes the pose stream.

Deadman switch: both triggers (button 0) held for >long_press_s.
X (left controller button 4): switch to the second prompt.
Y (left controller button 5): start/stop recording.
A/B (right controller buttons 4/5): close/open gripper.
Right thumbstick Y: gripper velocity axis (up=close, down=open).
"""

import multiprocessing as mp
import threading
import time
from dataclasses import dataclass

from teleop_xr import Teleop
from teleop_xr.config import TeleopSettings
from teleop_xr.messages import XRState

# Shared-memory layout (16 doubles, lock-free):
#  [0]  arm_enabled   (0.0 / 1.0)
#  [1]  pos_x
#  [2]  pos_y
#  [3]  pos_z
#  [4]  quat_x
#  [5]  quat_y
#  [6]  quat_z
#  [7]  quat_w
#  [8]  x_pressed     (0.0 / 1.0)
#  [9]  y_pressed     (0.0 / 1.0)
#  [10] gripper_close (0.0 / 1.0)
#  [11] gripper_open  (0.0 / 1.0)
#  [12] right_stick_x
#  [13] right_stick_y
#  [14] gripper_velocity_axis  (+close / -open)
#  [15] timestamp
_SHM_SIZE = 18


def _normalize_quaternion(quaternion_xyzw):
    try:
        return tuple(normalize_quat_xyzw(quaternion_xyzw))
    except ValueError:
        return (0., 0., 0., 1.)


def _nlerp_quaternion(start_xyzw, end_xyzw, alpha):
    return tuple(nlerp_quat_xyzw(_normalize_quaternion(start_xyzw), _normalize_quaternion(end_xyzw), alpha))


@dataclass(slots=True)
class VRInput:
    """Processed VR controller state — all scalars, no arrays."""

    # Deadman: both triggers held > threshold
    arm_enabled: bool = False

    # Right controller grip pose (world frame)
    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_z: float = 0.0
    quat_x: float = 0.0
    quat_y: float = 0.0
    quat_z: float = 0.0
    quat_w: float = 1.0

    # Button states (active while pressed)
    x_pressed: bool = False
    y_pressed: bool = False
    save_pressed: bool = False
    gripper_close: bool = False  # A (right btn 4)
    gripper_open: bool = False  # B (right btn 5)
    right_stick_x: float = 0.0
    right_stick_y: float = 0.0
    gripper_velocity_axis: float = 0.0  # +close (stick up), -open (stick down)

    timestamp: float = 0.0
    pose_monotonic_ns: int = 0
    pose_seq: int = 0


class VRInputReader:
    """Non-blocking VR input reader.

    Call ``run()`` in a daemon thread. Read ``latest`` from the main thread.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 4443,
        long_press_s: float = 0.5,
        position_alpha: float = 1.0,
        rotation_alpha: float = 1.0,
    ) -> None:
        if not 0.0 < position_alpha <= 1.0:
            raise ValueError("position_alpha must be in (0, 1]")
        if not 0.0 < rotation_alpha <= 1.0:
            raise ValueError("rotation_alpha must be in (0, 1]")

        self._long_press_s = long_press_s
        self._position_alpha = float(position_alpha)
        self._rotation_alpha = float(rotation_alpha)
        self._latest: VRInput = VRInput()

        # Internal timing for deadman detection
        self._left_trigger_since: float = 0.0
        self._right_trigger_since: float = 0.0

        self._filtered_pos: tuple[float, float, float] | None = None
        self._filtered_quat: tuple[float, float, float, float] | None = None

        settings = TeleopSettings(host=host, port=port)
        self._teleop = Teleop(settings=settings)
        self._teleop.subscribe(self._on_xr_update)

    @property
    def latest(self) -> VRInput:
        return self._latest

    def run(self) -> None:
        """Blocking — run in a daemon thread."""
        self._teleop.run()

    # ------------------------------------------------------------------

    def _set_filtered_pose(
        self,
        pos_x: float,
        pos_y: float,
        pos_z: float,
        quat_x: float,
        quat_y: float,
        quat_z: float,
        quat_w: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        raw_pos = (pos_x, pos_y, pos_z)
        raw_quat = _normalize_quaternion((quat_x, quat_y, quat_z, quat_w))
        self._filtered_pos = raw_pos
        self._filtered_quat = raw_quat
        return raw_pos, raw_quat

    def _filter_pose(
        self,
        *,
        arm_enabled: bool,
        pos_x: float,
        pos_y: float,
        pos_z: float,
        quat_x: float,
        quat_y: float,
        quat_z: float,
        quat_w: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        if not arm_enabled or self._filtered_pos is None or self._filtered_quat is None:
            return self._set_filtered_pose(
                pos_x,
                pos_y,
                pos_z,
                quat_x,
                quat_y,
                quat_z,
                quat_w,
            )

        raw_pos = (pos_x, pos_y, pos_z)
        raw_quat = _normalize_quaternion((quat_x, quat_y, quat_z, quat_w))

        filtered_pos = tuple(
            prev_component + self._position_alpha * (raw_component - prev_component)
            for prev_component, raw_component in zip(self._filtered_pos, raw_pos)
        )
        filtered_quat = _nlerp_quaternion(
            self._filtered_quat,
            raw_quat,
            self._rotation_alpha,
        )

        self._filtered_pos = filtered_pos
        self._filtered_quat = filtered_quat
        return filtered_pos, filtered_quat

    def _on_xr_update(self, _pose, message) -> None:
        xr_data = message.get("data", message)
        state = XRState.model_validate(xr_data)

        now = time.monotonic()
        left_dev = None
        right_dev = None

        for dev in state.devices:
            if not dev.handedness:
                continue
            hand = dev.handedness.value
            if hand == "left":
                left_dev = dev
            elif hand == "right":
                right_dev = dev

        # --- Trigger state (button 0) for deadman ---
        left_trigger = False
        right_trigger = False
        if left_dev and left_dev.gamepad and len(left_dev.gamepad.buttons) > 0:
            left_trigger = left_dev.gamepad.buttons[0].pressed
        if right_dev and right_dev.gamepad and len(right_dev.gamepad.buttons) > 0:
            right_trigger = right_dev.gamepad.buttons[0].pressed

        if left_trigger:
            if self._left_trigger_since == 0.0:
                self._left_trigger_since = now
        else:
            self._left_trigger_since = 0.0

        if right_trigger:
            if self._right_trigger_since == 0.0:
                self._right_trigger_since = now
        else:
            self._right_trigger_since = 0.0

        arm_enabled = (
            self._left_trigger_since > 0.0
            and self._right_trigger_since > 0.0
            and (now - self._left_trigger_since) >= self._long_press_s
            and (now - self._right_trigger_since) >= self._long_press_s
        )

        # --- Right controller grip pose ---
        pos_x = pos_y = pos_z = 0.0
        quat_x = quat_y = quat_z = 0.0
        quat_w = 1.0
        if right_dev:
            pose = right_dev.gripPose or right_dev.pose
            if pose:
                p = pose.position
                pos_x = float(p.get("x", 0.0))
                pos_y = float(p.get("y", 0.0))
                pos_z = float(p.get("z", 0.0))
                o = pose.orientation
                quat_x = float(o.get("x", 0.0))
                quat_y = float(o.get("y", 0.0))
                quat_z = float(o.get("z", 0.0))
                quat_w = float(o.get("w", 1.0))

        # --- Buttons ---
        x_pressed = False
        y_pressed = False
        gripper_close = False
        gripper_open = False
        right_stick_x = 0.0
        right_stick_y = 0.0

        # Left controller: X (btn 4) -> second prompt, Y (btn 5) -> start/stop
        if left_dev and left_dev.gamepad:
            btns = left_dev.gamepad.buttons
            if len(btns) > 4 and btns[4].pressed:
                x_pressed = True
            if len(btns) > 5 and btns[5].pressed:
                y_pressed = True

        save_pressed = y_pressed

        # Right controller: A (btn 4) -> close, B (btn 5) -> open
        if right_dev and right_dev.gamepad:
            btns = right_dev.gamepad.buttons
            axes = right_dev.gamepad.axes
            if len(btns) > 4 and btns[4].pressed:
                gripper_close = True
            if len(btns) > 5 and btns[5].pressed:
                gripper_open = True
            if len(axes) >= 2:
                right_stick_x = float(axes[-2])
                right_stick_y = float(axes[-1])

        gripper_velocity_axis = -right_stick_y

        (filtered_pos, filtered_quat) = self._filter_pose(
            arm_enabled=arm_enabled,
            pos_x=pos_x,
            pos_y=pos_y,
            pos_z=pos_z,
            quat_x=quat_x,
            quat_y=quat_y,
            quat_z=quat_z,
            quat_w=quat_w,
        )

        # Atomic reference swap — no lock needed
        self._latest = VRInput(
            arm_enabled=arm_enabled,
            pos_x=filtered_pos[0],
            pos_y=filtered_pos[1],
            pos_z=filtered_pos[2],
            quat_x=filtered_quat[0],
            quat_y=filtered_quat[1],
            quat_z=filtered_quat[2],
            quat_w=filtered_quat[3],
            x_pressed=x_pressed,
            y_pressed=y_pressed,
            save_pressed=save_pressed,
            gripper_close=gripper_close,
            gripper_open=gripper_open,
            right_stick_x=right_stick_x,
            right_stick_y=right_stick_y,
            gripper_velocity_axis=gripper_velocity_axis,
            timestamp=now,
            pose_monotonic_ns=time.monotonic_ns(),
            pose_seq=self._latest.pose_seq+1,
        )


class VRInputProcess:
    """Out-of-process VR reader.  Immune to main-process GIL contention.

    Usage::

        vr = VRInputProcess()
        vr.start()
        ...
        v = vr.latest          # zero-copy read of 12 doubles
        vr.stop()              # on shutdown
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 4443,
        long_press_s: float = 0.5,
        position_alpha: float = 1.0,
        rotation_alpha: float = 1.0,
    ) -> None:
        self._shm: mp.Array = mp.Array("d", _SHM_SIZE, lock=True)
        self._shm[7] = 1.0  # quat_w default
        self._proc = mp.Process(
            target=self._vr_worker,
            args=(
                self._shm,
                host,
                port,
                long_press_s,
                position_alpha,
                rotation_alpha,
            ),
            daemon=True,
        )

    def start(self) -> None:
        self._proc.start()

    def stop(self) -> None:
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2.0)

    def _vr_worker(
        self,
        shm: mp.Array,
        host: str,
        port: int,
        long_press_s: float,
        position_alpha: float,
        rotation_alpha: float,
    ) -> None:
        """Child-process entry point.  Has its own GIL — teleop_xr runs free."""
        reader = VRInputReader(
            host=host,
            port=port,
            long_press_s=long_press_s,
            position_alpha=position_alpha,
            rotation_alpha=rotation_alpha,
        )
        t = threading.Thread(target=reader.run, daemon=True)
        t.start()
        while True:
            v = reader.latest
            with shm.get_lock():
                shm[0] = 1.0 if v.arm_enabled else 0.0
                shm[1] = v.pos_x
                shm[2] = v.pos_y
                shm[3] = v.pos_z
                shm[4] = v.quat_x
                shm[5] = v.quat_y
                shm[6] = v.quat_z
                shm[7] = v.quat_w
                shm[8] = 1.0 if v.x_pressed else 0.0
                shm[9] = 1.0 if v.y_pressed else 0.0
                shm[10] = 1.0 if v.gripper_close else 0.0
                shm[11] = 1.0 if v.gripper_open else 0.0
                shm[12] = v.right_stick_x
                shm[13] = v.right_stick_y
                shm[14] = v.gripper_velocity_axis
                shm[15] = v.timestamp
                shm[16] = v.pose_monotonic_ns
                shm[17] = v.pose_seq
            time.sleep(0.001)  # 1 kHz — <1 ms latency, negligible CPU

    @property
    def latest(self) -> VRInput:
        with self._shm.get_lock():
            s = list(self._shm)
        return VRInput(
            arm_enabled=s[0] > 0.5,
            pos_x=s[1],
            pos_y=s[2],
            pos_z=s[3],
            quat_x=s[4],
            quat_y=s[5],
            quat_z=s[6],
            quat_w=s[7],
            x_pressed=s[8] > 0.5,
            y_pressed=s[9] > 0.5,
            save_pressed=(s[8] > 0.5) or (s[9] > 0.5),
            gripper_close=s[10] > 0.5,
            gripper_open=s[11] > 0.5,
            right_stick_x=s[12],
            right_stick_y=s[13],
            pose_monotonic_ns=int(s[16]),
            pose_seq=int(s[17]),
            gripper_velocity_axis=s[14],
            timestamp=s[15],
        )


def main() -> None:
    vr = VRInputProcess()
    vr.start()
    print("VR Input Process started, waiting for XR data...")
    try:
        while True:
            v = vr.latest
            print(
                f"enabled={v.arm_enabled}  "
                f"quat=({v.quat_x:+.3f}, {v.quat_y:+.3f}, {v.quat_z:+.3f}, {v.quat_w:+.3f})  "
                f"x={v.x_pressed}  y={v.y_pressed}  save={v.save_pressed}  "
                f"grip_close={v.gripper_close}  grip_open={v.gripper_open}  "
                f"right_stick=({v.right_stick_x:+.3f}, {v.right_stick_y:+.3f})  "
                f"grip_vel_axis={v.gripper_velocity_axis:+.3f}",
                end="\r",
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        vr.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
