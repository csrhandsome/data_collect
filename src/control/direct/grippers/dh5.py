"""DH5 serial gripper implementation using its native position register."""

from control.direct.grippers.backend import GripperBackend


class DH5GripperBackend(GripperBackend):
    kind = "dh5"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.enable_cameras = config.get("tactile", {}).get("enabled", False)
        self.tactile = config.get("tactile", {})

    def connect(self) -> None:
        from control.direct.soft_gripper_control import DH5Gripper

        if self.device is None:
            try:
                options = dict(self.config.get("dh5", {}))
                if self.enable_cameras:
                    from control.direct.camera_connector import (
                        RealSenseConnector,
                        RGBCameraConnector,
                    )

                    for side in ("left", "right"):
                        if side not in self.tactile:
                            continue
                        camera = dict(self.tactile[side])
                        kind = camera.pop("type", "rgb")
                        camera.setdefault("fps", options.get("camera_fps", 30))
                        camera.setdefault("width", options.get("camera_width", 640))
                        camera.setdefault("height", options.get("camera_height", 480))
                        if kind == "realsense":
                            camera["enable_depth"] = False
                            connector = RealSenseConnector(**camera)
                        elif kind == "rgb":
                            connector = RGBCameraConnector(**camera)
                        else:
                            raise ValueError(f"Unknown tactile.{side}.type: {kind}")
                        options[f"{side}_camera"] = connector
                self.device = DH5Gripper(**options, enable_cameras=self.enable_cameras)
                self.device.set_force(int(self.config.get("force", 50)))
                self.device.set_velocity(int(self.config.get("velocity", 100)))
                self.commanded_open_ratio = 1.0
            except BaseException:
                self.close()
                raise

    def close(self) -> None:
        try:
            if self.device is not None:
                self.device.close()
        finally:
            self.device = None

    def command(self, ratio: float, *, speed: float, force: float) -> bool:
        self.require_connected()
        position = ratio if self.config.get("open_position_high", False) else 1 - ratio
        self.device.set_position(round(position * 1000), wait=True)
        self.commanded_open_ratio = ratio
        return True

    def stop(self) -> None:
        raise NotImplementedError("DH5 driver has no confirmed stop command")

    def tactile_images(self):
        self.require_connected()
        observation = self.device.observation
        return observation.wrist_img, observation.external_img

    def tactile_frames(self):
        self.require_connected()
        cameras = self.device.dual_camera_manager
        if cameras is None:
            return None, None, None, None
        right, left, right_ns, left_ns = cameras.get_frames()
        return left, right, left_ns, right_ns
