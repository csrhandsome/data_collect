"""Optional publisher-only ROS 2 telemetry, independent of robot hardware.

Construct before opening devices so missing ROS dependencies fail at startup.
``submit`` copies a single latest snapshot; message creation, RGB serialization
and DDS calls run on the worker. No executor, subscriptions or commands exist.
ROS imports (and a private ROS context) are only created when enabled.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass

_LOG = logging.getLogger(__name__)
_NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class _Snapshot:
    submitted_ns: int
    robot: dict
    vr: dict | None
    target: tuple | None
    cameras: dict


class Ros2Publisher:
    """Read-only telemetry context manager accepting the full application config.

    State QoS is reliable/volatile/keep-last(1); images use sensor-data QoS
    (best-effort/volatile/keep-last(1)). Headers use a fixed host-monotonic to
    ROS SYSTEM_TIME anchor, independent of /clock. ``timestamps`` describes the
    anchor and each source clock; camera sensor clocks are never used as ROS time.

    VR pose reflects the supplied VRInput's position/quaternion fields. Optional
    pre-filter raw pose fields are also preserved in ``vr/input``. Target headers
    use the submit receipt time, since the target tuple has no source timestamp.
    After close or a runtime failure, submit is a no-op. If DDS blocks during
    close, the daemon worker retains ownership and cleans up when it returns.
    """

    _CLOSE_TIMEOUT_S = 2.0

    def __init__(self, config):
        self._condition = threading.Condition()
        self._enabled = False
        self._stopping = False
        self._pending = None
        self._thread = None
        self._context = None
        self._node = None
        self._publishers = {}
        self._last_image_ns = {}
        cfg = config.get("ros2", {})
        self._publish_ee_wrench = (
            config.get("robot", {}).get("backend") == "ros2"
            and config.get("observation", {}).get("ee_wrench", False)
        )
        if not cfg.get("enabled", False):
            return

        self._publish_hz = float(cfg.get("publish_hz", 30.0))
        if not math.isfinite(self._publish_hz) or self._publish_hz <= 0:
            raise ValueError("ros2.publish_hz must be finite and positive")
        self._publish_images = bool(cfg.get("publish_images", True))
        self._vr_frame = cfg.get("vr_frame", "vr_world")
        self._camera_frames = {
            "external": cfg.get("external_camera_frame", "external_camera_optical_frame"),
            "wrist": cfg.get("wrist_camera_frame", "wrist_camera_optical_frame"),
        }
        self._arm_id = cfg.get("arm_id", "panda")
        try:
            self._initialize(cfg)
            self._enabled = True
            self._thread = threading.Thread(target=self._run, name="ros2-telemetry", daemon=True)
            self._thread.start()
        except BaseException as exc:
            self._enabled = False
            self._cleanup()
            if not isinstance(exc, Exception):
                raise
            raise RuntimeError(
                "ROS 2 telemetry initialization failed. Install/source a ROS 2 distribution "
                "with rclpy, sensor_msgs, geometry_msgs and std_msgs for this Python interpreter "
                "(for example, source /opt/ros/jazzy/setup.bash), and check the ROS namespace, "
                "node name and DDS configuration; or set ros2.enabled=false. "
                f"Cause: {exc}"
            ) from exc

    def _initialize(self, cfg):
        # Keep every ROS import here: importing this module while disabled is safe
        # in collection environments that have neither ROS nor its native libraries.
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.clock import Clock, ClockType
        from rclpy.context import Context
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import Image, JointState
        from std_msgs.msg import String

        self._types = {"pose": PoseStamped, "image": Image, "joints": JointState, "json": String}
        if self._publish_ee_wrench:
            from geometry_msgs.msg import WrenchStamped

            self._types["wrench"] = WrenchStamped
        self._context = Context()
        # No default context, global ROS arguments, /clock, or signal handlers.
        rclpy.init(args=[], context=self._context, signal_handler_options=SignalHandlerOptions.NO)
        self._node = rclpy.create_node(
            cfg.get("node_name", "data_collect_bridge"),
            namespace=cfg.get("telemetry_namespace", "/data_collect"),
            context=self._context,
            use_global_arguments=False,
            enable_rosout=False,
            start_parameter_services=False,
        )
        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        topics = {
            "robot/joint_states": "joints",
            "robot/ee_pose": "pose",
            "robot/gripper_state": "json",
            "robot/target_ee_pose": "pose",
            "vr/pose": "pose",
            "vr/input": "json",
            "timestamps": "json",
        }
        if self._publish_ee_wrench:
            topics["robot/ee_wrench"] = "wrench"
        if self._publish_images:
            topics.update({f"camera/{name}/image": "image" for name in self._camera_frames})
        for topic, kind in topics.items():
            self._publishers[topic] = self._node.create_publisher(
                self._types[kind], topic, image_qos if kind == "image" else state_qos
            )

        # Bracket the wall-clock read to reduce sampling skew. This explicit fixed
        # anchor preserves source intervals even if the system clock later jumps.
        before = time.monotonic_ns()
        self._anchor_ros_ns = Clock(clock_type=ClockType.SYSTEM_TIME).now().nanoseconds
        after = time.monotonic_ns()
        self._anchor_monotonic_ns = (before + after) // 2

    @property
    def enabled(self) -> bool:
        with self._condition:
            return self._enabled

    def submit(self, state, *, vr=None, target=None, cameras=None):
        """Replace the pending snapshot; never perform ROS calls or encode images.

        Inputs must not be mutated concurrently *during* this call. On return all
        mutable arrays/VR fields are owned copies, so callers can reuse buffers.
        ``cameras`` must contain the current preprocessed RGB CameraPair.
        """
        if not self.enabled:
            return
        try:
            snapshot = self._snapshot(state, vr, target, cameras)
        except Exception:
            self._disable("ROS 2 telemetry snapshot failed; telemetry disabled")
            return
        with self._condition:
            if self._enabled:
                self._pending = snapshot
                self._condition.notify()

    def _snapshot(self, state, vr, target, cameras):
        submitted_ns = time.monotonic_ns()
        robot = {
            "sampled_monotonic_ns": int(state.sampled_monotonic_ns),
            "base_frame": str(state.base_frame),
            "joint_positions": tuple(float(x) for x in state.joint_positions),
            "joint_velocities": tuple(float(x) for x in state.joint_velocities),
            "position": tuple(float(x) for x in state.ee_position),
            "quaternion_xyzw": tuple(float(x) for x in state.ee_quaternion_xyzw),
            "robot_time_s": getattr(state, "robot_time_s", None),
            "gripper": {
                "kind": state.gripper.kind,
                "commanded_open_ratio": float(state.gripper.commanded_open_ratio),
                "measured_width_m": (
                    None
                    if state.gripper.measured_width_m is None
                    else float(state.gripper.measured_width_m)
                ),
                "busy": bool(getattr(state.gripper, "busy", False)),
            },
        }
        if self._publish_ee_wrench:
            if state.ee_wrench is None:
                raise ValueError("ROS 2 external wrench feedback is unavailable")
            robot["ee_wrench"] = tuple(float(x) for x in state.ee_wrench)
        vr_copy = None
        if vr is not None:
            vr_copy = {
                "frame_id": self._vr_frame,
                "position": [float(vr.pos_x), float(vr.pos_y), float(vr.pos_z)],
                "quaternion_xyzw": [float(getattr(vr, f"quat_{axis}")) for axis in "xyzw"],
                "enabled": bool(vr.arm_enabled),
                "pose_monotonic_ns": int(getattr(vr, "pose_monotonic_ns", 0)),
                "pose_seq": int(getattr(vr, "pose_seq", 0)),
                "timestamp": getattr(vr, "timestamp", None),
                "buttons": {
                    key: bool(getattr(vr, key, False))
                    for key in (
                        "x_pressed",
                        "y_pressed",
                        "save_pressed",
                        "critical_pressed",
                        "gripper_open",
                        "gripper_close",
                    )
                },
                "right_stick_x": float(getattr(vr, "right_stick_x", 0.0)),
                "right_stick_y": float(getattr(vr, "right_stick_y", 0.0)),
                "gripper_velocity_axis": float(getattr(vr, "gripper_velocity_axis", 0.0)),
            }
            for key in ("raw_position", "raw_quaternion_xyzw"):
                value = getattr(vr, key, None)
                vr_copy[key] = None if value is None else tuple(float(x) for x in value)
        target_copy = None if target is None else tuple(tuple(float(x) for x in v) for v in target)
        camera_copy = {}
        if cameras is not None:
            for name, prefix in (("external", "front"), ("wrist", "wrist")):
                image = getattr(cameras, prefix)
                camera_copy[name] = {
                    "host_capture_monotonic_ns": int(getattr(cameras, f"{prefix}_ns")),
                    "sensor_timestamp": getattr(cameras, f"{prefix}_sensor_timestamp"),
                    "image": image.copy(order="C")
                    if self._publish_images and image is not None
                    else None,
                }
        return _Snapshot(submitted_ns, robot, vr_copy, target_copy, camera_copy)

    def _run(self):
        next_publish = 0.0
        try:
            while True:
                with self._condition:
                    while not self._stopping:
                        delay = next_publish - time.monotonic()
                        if self._pending is not None and delay <= 0:
                            break
                        self._condition.wait(delay if self._pending is not None else None)
                    if self._stopping:
                        return
                    snapshot, self._pending = self._pending, None
                next_publish = time.monotonic() + 1.0 / self._publish_hz
                self._publish(snapshot)
        except Exception:
            self._disable("ROS 2 telemetry worker failed; telemetry disabled")
        finally:
            self._cleanup()

    def _stamp(self, monotonic_ns):
        ros_ns = self._anchor_ros_ns + int(monotonic_ns) - self._anchor_monotonic_ns
        sec, nanosec = divmod(ros_ns, _NS_PER_S)
        return {"sec": sec, "nanosec": nanosec}

    @staticmethod
    def _header(message, frame, stamp):
        message.header.frame_id = frame
        message.header.stamp.sec = stamp["sec"]
        message.header.stamp.nanosec = stamp["nanosec"]

    def _json(self, topic, payload):
        message = self._types["json"]()
        message.data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        self._publishers[topic].publish(message)

    def _pose(self, topic, position, quaternion, frame, stamp):
        message = self._types["pose"]()
        self._header(message, frame, stamp)
        message.pose.position.x, message.pose.position.y, message.pose.position.z = position
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = quaternion
        self._publishers[topic].publish(message)

    def _publish(self, snapshot):
        robot = snapshot.robot
        robot_stamp = self._stamp(robot["sampled_monotonic_ns"])
        metadata = {
            "clock": {
                "header_clock": "ros_system_time",
                "anchor_monotonic_ns": self._anchor_monotonic_ns,
                "anchor_ros_ns": self._anchor_ros_ns,
            },
            "submitted_monotonic_ns": snapshot.submitted_ns,
            "robot": {
                "sampled_monotonic_ns": robot["sampled_monotonic_ns"],
                "robot_time_s": robot["robot_time_s"],
                "frame_id": robot["base_frame"],
                "stamp": robot_stamp,
            },
            "vr": None,
            "target": None,
            "cameras": {},
        }
        joints = self._types["joints"]()
        self._header(joints, robot["base_frame"], robot_stamp)
        joints.name = [f"{self._arm_id}_joint{i}" for i in range(1, 8)]
        joints.position = list(robot["joint_positions"])
        joints.velocity = list(robot["joint_velocities"])
        self._publishers["robot/joint_states"].publish(joints)
        self._pose(
            "robot/ee_pose",
            robot["position"],
            robot["quaternion_xyzw"],
            robot["base_frame"],
            robot_stamp,
        )
        if self._publish_ee_wrench:
            message = self._types["wrench"]()
            self._header(message, robot["base_frame"], robot_stamp)
            message.wrench.force.x, message.wrench.force.y, message.wrench.force.z = (
                robot["ee_wrench"][:3]
            )
            message.wrench.torque.x, message.wrench.torque.y, message.wrench.torque.z = (
                robot["ee_wrench"][3:]
            )
            self._publishers["robot/ee_wrench"].publish(message)
        self._json(
            "robot/gripper_state",
            {
                **robot["gripper"],
                "sampled_monotonic_ns": robot["sampled_monotonic_ns"],
                "frame_id": robot["base_frame"],
                "stamp": robot_stamp,
            },
        )
        if snapshot.target is not None:
            target_stamp = self._stamp(snapshot.submitted_ns)
            self._pose("robot/target_ee_pose", *snapshot.target, robot["base_frame"], target_stamp)
            metadata["target"] = {
                "submitted_monotonic_ns": snapshot.submitted_ns,
                "stamp": target_stamp,
                "frame_id": robot["base_frame"],
            }
        if snapshot.vr is not None:
            vr = snapshot.vr
            # Zero is the VRInput startup sentinel. Preserve it as provenance,
            # but use submit time for the header until a pose receipt exists.
            stamp_ns = (
                vr["pose_monotonic_ns"] if vr["pose_monotonic_ns"] > 0 else snapshot.submitted_ns
            )
            vr_stamp = self._stamp(stamp_ns)
            self._pose("vr/pose", vr["position"], vr["quaternion_xyzw"], self._vr_frame, vr_stamp)
            self._json("vr/input", {**vr, "stamp": vr_stamp})
            metadata["vr"] = {
                "pose_monotonic_ns": vr["pose_monotonic_ns"],
                "pose_seq": vr["pose_seq"],
                "timestamp": vr["timestamp"],
                "stamp_monotonic_ns": stamp_ns,
                "frame_id": self._vr_frame,
                "stamp": vr_stamp,
            }
        for name, camera in snapshot.cameras.items():
            source_ns = camera["host_capture_monotonic_ns"]
            stamp = self._stamp(source_ns)
            image = camera["image"]
            published = False
            # Host capture receipts are monotonic. A high-water mark also avoids
            # re-publishing a stale frame if an older pair is later submitted.
            if image is not None and source_ns > self._last_image_ns.get(name, -1):
                if image.ndim != 3 or image.shape[2] != 3 or image.dtype.name != "uint8":
                    raise ValueError("ROS camera images must be preprocessed HxWx3 RGB uint8")
                message = self._types["image"]()
                self._header(message, self._camera_frames[name], stamp)
                message.height, message.width = image.shape[:2]
                message.encoding = "rgb8"
                message.is_bigendian = 0
                message.step = message.width * 3
                message.data = image.tobytes(order="C")
                self._publishers[f"camera/{name}/image"].publish(message)
                self._last_image_ns[name] = source_ns
                published = True
            metadata["cameras"][name] = {
                "host_capture_monotonic_ns": source_ns,
                "sensor_timestamp": camera["sensor_timestamp"],
                "frame_id": self._camera_frames[name],
                "stamp": stamp,
                "image_published": published,
            }
        self._json("timestamps", metadata)

    def _disable(self, message):
        with self._condition:
            self._enabled = False
            self._stopping = True
            self._pending = None
            self._condition.notify_all()
        _LOG.exception(message)

    def _cleanup(self):
        # Only the constructor (before thread start) or worker owns ROS resources.
        # A blocked publisher must never race close() destroying its native node.
        node, self._node = self._node, None
        context, self._context = self._context, None
        self._publishers.clear()
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                _LOG.exception("Could not destroy ROS 2 telemetry node")
        if context is not None:
            try:
                context.try_shutdown()
            except Exception:
                _LOG.exception("Could not shut down ROS 2 telemetry context")
            finally:
                try:
                    context.destroy()
                except Exception:
                    _LOG.exception("Could not destroy ROS 2 telemetry context")

    def close(self):
        with self._condition:
            self._enabled = False
            self._stopping = True
            self._pending = None
            self._condition.notify_all()
        thread = self._thread
        if (
            thread is not None
            and thread is not threading.current_thread()
            and thread.ident is not None
        ):
            thread.join(timeout=self._CLOSE_TIMEOUT_S)
            if thread.is_alive():
                _LOG.warning("ROS 2 telemetry is still publishing; worker will clean up on return")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
