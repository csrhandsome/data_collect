"""VR controller input reader via teleop_xr.

Provides two modes:
  - VRInputReader : in-process (thread).  Good for standalone testing.
  - VRInputProcess: out-of-process via multiprocessing.  Immune to GIL
    contention from camera / IK / panda threads in the main process.

Deadman switch: both triggers (button 0) held for >long_press_s.
X/Y (left controller buttons 4/5): save episode.
A/B (right controller buttons 4/5): close/open gripper.
"""

import multiprocessing as mp
import threading
import time
from dataclasses import dataclass

from teleop_xr import Teleop
from teleop_xr.config import TeleopSettings
from teleop_xr.messages import XRState

# Shared-memory layout (12 doubles, lock-free):
#  [0]  arm_enabled   (0.0 / 1.0)
#  [1]  pos_x
#  [2]  pos_y
#  [3]  pos_z
#  [4]  quat_x
#  [5]  quat_y
#  [6]  quat_z
#  [7]  quat_w
#  [8]  save_pressed  (0.0 / 1.0)
#  [9]  gripper_close (0.0 / 1.0)
#  [10] gripper_open  (0.0 / 1.0)
#  [11] timestamp
_SHM_SIZE = 12


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
    save_pressed: bool = False
    gripper_close: bool = False  # A (right btn 4)
    gripper_open: bool = False  # B (right btn 5)

    timestamp: float = 0.0


class VRInputReader:
    """Non-blocking VR input reader.

    Call ``run()`` in a daemon thread. Read ``latest`` from the main thread.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 4443,
        long_press_s: float = 0.5,
    ) -> None:
        self._long_press_s = long_press_s
        self._latest: VRInput = VRInput()

        # Internal timing for deadman detection
        self._left_trigger_since: float = 0.0
        self._right_trigger_since: float = 0.0

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
        save_pressed = False
        gripper_close = False
        gripper_open = False

        # Left controller: X (btn 4) / Y (btn 5) -> save
        if left_dev and left_dev.gamepad:
            btns = left_dev.gamepad.buttons
            if len(btns) > 4 and btns[4].pressed:
                save_pressed = True
            if len(btns) > 5 and btns[5].pressed:
                save_pressed = True

        # Right controller: A (btn 4) -> close, B (btn 5) -> open
        if right_dev and right_dev.gamepad:
            btns = right_dev.gamepad.buttons
            if len(btns) > 4 and btns[4].pressed:
                gripper_close = True
            if len(btns) > 5 and btns[5].pressed:
                gripper_open = True

        # Atomic reference swap — no lock needed
        self._latest = VRInput(
            arm_enabled=arm_enabled,
            pos_x=pos_x,
            pos_y=pos_y,
            pos_z=pos_z,
            quat_x=quat_x,
            quat_y=quat_y,
            quat_z=quat_z,
            quat_w=quat_w,
            save_pressed=save_pressed,
            gripper_close=gripper_close,
            gripper_open=gripper_open,
            timestamp=now,
        )


def _vr_worker(
    shm: mp.Array,
    host: str,
    port: int,
    long_press_s: float,
) -> None:
    """Child-process entry point.  Has its own GIL — teleop_xr runs free."""
    reader = VRInputReader(host=host, port=port, long_press_s=long_press_s)
    t = threading.Thread(target=reader.run, daemon=True)
    t.start()
    while True:
        v = reader.latest
        shm[0] = 1.0 if v.arm_enabled else 0.0
        shm[1] = v.pos_x
        shm[2] = v.pos_y
        shm[3] = v.pos_z
        shm[4] = v.quat_x
        shm[5] = v.quat_y
        shm[6] = v.quat_z
        shm[7] = v.quat_w
        shm[8] = 1.0 if v.save_pressed else 0.0
        shm[9] = 1.0 if v.gripper_close else 0.0
        shm[10] = 1.0 if v.gripper_open else 0.0
        shm[11] = v.timestamp
        time.sleep(0.001)  # 1 kHz — <1 ms latency, negligible CPU


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
    ) -> None:
        self._shm: mp.Array = mp.Array("d", _SHM_SIZE, lock=False)
        self._shm[7] = 1.0  # quat_w default
        self._proc = mp.Process(
            target=_vr_worker,
            args=(self._shm, host, port, long_press_s),
            daemon=True,
        )

    def start(self) -> None:
        self._proc.start()

    def stop(self) -> None:
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2.0)

    @property
    def latest(self) -> VRInput:
        s = self._shm
        return VRInput(
            arm_enabled=s[0] > 0.5,
            pos_x=s[1],
            pos_y=s[2],
            pos_z=s[3],
            quat_x=s[4],
            quat_y=s[5],
            quat_z=s[6],
            quat_w=s[7],
            save_pressed=s[8] > 0.5,
            gripper_close=s[9] > 0.5,
            gripper_open=s[10] > 0.5,
            timestamp=s[11],
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
                f"pos=({v.pos_x:+.3f}, {v.pos_y:+.3f}, {v.pos_z:+.3f})  "
                f"quat=({v.quat_x:+.3f}, {v.quat_y:+.3f}, {v.quat_z:+.3f}, {v.quat_w:+.3f})  "
                f"save={v.save_pressed}  "
                f"grip_close={v.gripper_close}  grip_open={v.gripper_open}",
                end="\r",
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        vr.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
