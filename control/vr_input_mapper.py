"""Map VRInput to DROID-style cartesian velocity (6,) in [-1, 1].

Snapshot-based: when arm becomes enabled, record the VR pose as the
neutral point. The displacement from that neutral point is proportional
to the output velocity, like a virtual joystick. This is robust to
GIL contention / slow VR update rates because the *total* displacement
from neutral is always correct regardless of how often we sample.

Important:
teleop_xr already converts raw WebXR coordinates from RUB to ROS FLU.
`VRInput` therefore arrives in FLU axes:
  +X = forward, +Y = left, +Z = up

Franka base frame in this pipeline uses the same axis convention, so we
must NOT remap axes again in this mapper.
"""

import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from control.vr_input import VRInput, VRInputReader


class VRInputMapper:
    """Snapshot-based mapper: displacement from neutral → velocity.

    Input pose is already in FLU (forward-left-up), so translation and
    rotation deltas are used directly as robot-base-frame commands.

    Rotation is **disabled by default** because natural wrist rotation
    during hand movement overwhelms the IK solver.  Enable with
    ``enable_rotation=True`` once translation is verified.
    """

    def __init__(
        self,
        translation_scale: float = 0.05,
        rotation_scale: float = 0.35,
        enable_rotation: bool = True,
    ) -> None:
        """
        Args:
            translation_scale: VR displacement (m) from neutral that maps
                               to 1.0 in cart_vel (full DROID speed).
                               0.05 = 5 cm displacement → full speed.
            rotation_scale:    VR rotation (rad) from neutral that maps
                               to 1.0 in cart_vel.
                               0.35 ≈ 20° rotation → full speed.
            enable_rotation:   Whether to include rotation in cart_vel.
        """
        self._ts = translation_scale
        self._rs = rotation_scale
        self._enable_rot = enable_rotation
        self._snap_pos: np.ndarray | None = None
        self._snap_rot: Rotation | None = None
        self._dbg_count = 0

    def reset(self) -> None:
        self._snap_pos = None
        self._snap_rot = None

    def map(self, vr: VRInput) -> np.ndarray:
        """Return cartesian_velocity (6,) in [-1, 1] for DroidIKSolver."""
        if not vr.arm_enabled:
            self._snap_pos = None
            self._snap_rot = None
            return np.zeros(6, dtype=np.float64)

        pos = np.array([vr.pos_x, vr.pos_y, vr.pos_z])
        rot = Rotation.from_quat([vr.quat_x, vr.quat_y, vr.quat_z, vr.quat_w])

        # First enabled frame: record neutral pose, output zero
        if self._snap_pos is None:
            self._snap_pos = pos
            self._snap_rot = rot
            return np.zeros(6, dtype=np.float64)

        # Displacement from neutral (not frame-to-frame)
        dp = pos - self._snap_pos

        # Translation: input is already FLU (robot base aligned)
        cart_vel = np.zeros(6, dtype=np.float64)
        cart_vel[0] = dp[0] / self._ts
        cart_vel[1] = dp[1] / self._ts
        cart_vel[2] = dp[2] / self._ts

        if self._enable_rot:
            dr = (rot * self._snap_rot.inv()).as_rotvec()
            cart_vel[3] = dr[0] / self._rs
            cart_vel[4] = dr[1] / self._rs
            cart_vel[5] = dr[2] / self._rs

        # Debug: print every ~1s at 15 Hz
        self._dbg_count += 1
        if self._dbg_count % 15 == 0:
            print(
                f"[VR→Robot] dp_vr=({dp[0]:+.4f},{dp[1]:+.4f},{dp[2]:+.4f}) "
                f"→ cv=({cart_vel[0]:+.2f},{cart_vel[1]:+.2f},{cart_vel[2]:+.2f},"
                f"{cart_vel[3]:+.2f},{cart_vel[4]:+.2f},{cart_vel[5]:+.2f})"
            )

        return np.clip(cart_vel, -1.0, 1.0)


def main() -> None:
    reader = VRInputReader()
    mapper = VRInputMapper()
    t = threading.Thread(target=reader.run, daemon=True)
    t.start()
    print("VR Input Mapper started, waiting for XR data...")
    try:
        while True:
            vr = reader.latest
            cv = mapper.map(vr)
            print(
                f"enabled={vr.arm_enabled}  "
                f"cart_vel=[{cv[0]:+.3f} {cv[1]:+.3f} {cv[2]:+.3f} "
                f"{cv[3]:+.3f} {cv[4]:+.3f} {cv[5]:+.3f}]  "
                f"save={vr.save_pressed}  "
                f"grip_close={vr.gripper_close}  grip_open={vr.gripper_open}",
                end="\r",
            )
            time.sleep(0.066)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
