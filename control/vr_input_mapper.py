"""Map VRInput to DROID-style cartesian velocity (6,) in [-1, 1].

Snapshot-based: when arm becomes enabled, record the VR pose as the
neutral point.  The displacement from that neutral point is proportional
to the output velocity — like a virtual joystick.  This is robust to
GIL contention / slow VR update rates because the *total* displacement
from neutral is always correct regardless of how often we sample.

VR (WebXR):  X=right, Y=up,  Z=back (toward user)
Robot (Panda): X=forward, Y=left, Z=up

Default mapping (user faces robot):
  VR -Z  -> Robot +X  (forward)
  VR -X  -> Robot +Y  (left)
  VR +Y  -> Robot +Z  (up)
"""

import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from control.vr_input import VRInput, VRInputReader


class VRInputMapper:
    """Snapshot-based mapper: displacement from neutral → velocity."""

    def __init__(
        self,
        translation_scale: float = 0.05,
        rotation_scale: float = 0.3,
    ) -> None:
        """
        Args:
            translation_scale: VR displacement (m) from neutral that maps
                               to 1.0 in cart_vel (full DROID speed).
                               0.05 = 5 cm displacement → full speed.
            rotation_scale:    VR rotation (rad) from neutral that maps
                               to 1.0 in cart_vel.
                               0.3 ≈ 17° rotation → full speed.
        """
        self._ts = translation_scale
        self._rs = rotation_scale
        self._snap_pos: np.ndarray | None = None
        self._snap_rot: Rotation | None = None

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
        dr = (rot * self._snap_rot.inv()).as_rotvec()  # (3,) radians

        # Coordinate transform VR -> Robot and scale to [-1, 1]
        cart_vel = np.array(
            [
                -dp[2] / self._ts,  # VR -Z -> Robot +X
                -dp[0] / self._ts,  # VR -X -> Robot +Y
                dp[1] / self._ts,   # VR +Y -> Robot +Z
                -dr[2] / self._rs,  # rotation around Robot X
                -dr[0] / self._rs,  # rotation around Robot Y
                dr[1] / self._rs,   # rotation around Robot Z
            ],
            dtype=np.float64,
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
