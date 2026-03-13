"""新的控制夹爪的sdk,需要更新robotic_arm_controller这个类里面的控制夹爪的函数(datacollect里面的都不要多线程控制夹爪了哈哈哈)"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import time
from typing import Optional

import crcmod
import numpy as np
import serial

from control.dual_camera_manager import DualRGBCameraManager


class ControlRoot(object):
    def __init__(self, com="/dev/ttyUSB0", timeout=0.2):
        self.sc = serial.Serial(
            port=com,
            baudrate=115200,
            timeout=timeout,
            write_timeout=timeout,
        )
        self.crc16 = crcmod.mkCrcFun(0x18005, rev=True, initCrc=0xFFFF, xorOut=0x0000)

    def calCrc(self, array):
        bytes_ = b""
        for i in range(array.__len__()):
            bytes_ = bytes_ + array[i].to_bytes(1, byteorder="big", signed=True)
        crc = self.crc16(bytes_).to_bytes(2, byteorder="big", signed=False)
        crcH = int.from_bytes(crc[0:1], byteorder="big", signed=False)
        crcQ = int.from_bytes(crc[1:2], byteorder="big", signed=False)
        return crcQ, crcH

    def clear_input_buffer(self) -> None:
        try:
            self.sc.reset_input_buffer()
        except AttributeError:
            self.sc.read_all()

    def readSerial(self, expected_length=None, timeout_s: Optional[float] = None):
        if expected_length is None:
            return self.sc.read_all()

        if timeout_s is None:
            timeout_s = self.sc.timeout
        timeout_s = 0.0 if timeout_s is None else max(float(timeout_s), 0.0)
        deadline = time.monotonic() + timeout_s
        received = bytearray()

        while len(received) < expected_length:
            remaining = expected_length - len(received)
            chunk = self.sc.read(remaining)
            if chunk:
                received.extend(chunk)
                continue
            if time.monotonic() >= deadline:
                break
            time.sleep(0.002)

        return bytes(received)

    def sendCmd(
        self,
        ModbusHighAddress,
        ModbusLowAddress,
        Value=0x01,
        isSet=True,
        isReadSerial=True,
    ):
        if isSet:
            SetAddress = 0x06
        else:
            SetAddress = 0x03
        Value = Value if Value >= 0 else Value - 1
        bytes_ = Value.to_bytes(2, byteorder="big", signed=True)
        ValueHexQ = int.from_bytes(bytes_[0:1], byteorder="big", signed=True)
        ValueHexH = int.from_bytes(bytes_[1:2], byteorder="big", signed=True)
        array = [
            0x01,
            SetAddress,
            ModbusHighAddress,
            ModbusLowAddress,
            ValueHexQ,
            ValueHexH,
        ]
        currentValueQ, currentValueH = self.calCrc(array)
        setValueCmd = [
            0x01,
            SetAddress,
            ModbusHighAddress,
            ModbusLowAddress,
            ValueHexQ,
            ValueHexH,
            currentValueQ,
            currentValueH,
        ]
        for i in range(setValueCmd.__len__()):
            setValueCmd[i] = (
                setValueCmd[i] if setValueCmd[i] >= 0 else setValueCmd[i] + 256
            )
        self.sc.write(bytes(setValueCmd))

        if isReadSerial:
            expected_length = 8 if isSet else 7
            back = self.readSerial(expected_length=expected_length)
            if len(back) < expected_length:
                raise RuntimeError(
                    f"No or short response from gripper on {self.sc.port}: "
                    f"expected {expected_length} bytes, got {len(back)} bytes ({back.hex(' ')})"
                )
            if back[1] & 0x80:
                raise RuntimeError(
                    f"Gripper returned Modbus exception frame: {back.hex(' ')}"
                )
            if isSet:
                value = int.from_bytes(back[4:6], byteorder="big", signed=True)
            else:
                value = int.from_bytes(back[3:5], byteorder="big", signed=True)
            if value < 0:
                value = value + 1
            self.sc.flush()
            return value
        time.sleep(0.005)
        return


def isRange(value, min_, max_):
    if not min_ <= value <= max_:
        raise RuntimeError("Out of range")


@dataclass
class DHGripperObservation:
    external_img: Optional[np.ndarray] = None
    wrist_img: Optional[np.ndarray] = None
    gripper_open_ratio: np.ndarray = field(
        default_factory=lambda: np.ones((1,), dtype=np.float32)
    )


class DH5Gripper(object):
    def __init__(
        self,
        com="/dev/ttyUSB0",
        right_camera_device: int | str = 1,
        left_camera_device: int | str = 2,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        camera_timeout_ms: int = 1000,
        camera_poll_interval_s: float = 0.01,
        discrete_level_count: int = 28,
        vr_axis_threshold: float = 0.55,
        vr_step_interval_s: float = 0.08,
    ):
        if int(discrete_level_count) <= 1:
            raise ValueError("discrete_level_count must be > 1")
        if not 0.0 < float(vr_axis_threshold) <= 1.0:
            raise ValueError("vr_axis_threshold must be in (0, 1]")
        if float(vr_step_interval_s) <= 0.0:
            raise ValueError("vr_step_interval_s must be > 0")

        self.Hand = ControlRoot(com=com)
        self._position_command = 0
        self._gripper_level_count = int(discrete_level_count)
        self._gripper_level = 0
        self._vr_axis_threshold = float(vr_axis_threshold)
        self._vr_step_interval_s = float(vr_step_interval_s)
        self._last_gripper_step_time = 0.0
        self._last_gripper_step_dir = 0
        self.dual_camera_manager = DualRGBCameraManager(
            right_camera_device=right_camera_device,
            left_camera_device=left_camera_device,
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
            background_poll=True,
            background_poll_interval_s=camera_poll_interval_s,
            background_timeout_ms=camera_timeout_ms,
        )
        self._observation = DHGripperObservation(
            gripper_open_ratio=self.gripper_open_ratio
        )
        self.initialize()
        self.dual_camera_manager.connect()
        self._gripper_level = self._position_to_level(self._position_command)

    def _build_observation(
        self,
        *,
        external_img: Optional[np.ndarray],
        wrist_img: Optional[np.ndarray],
    ) -> DHGripperObservation:
        return DHGripperObservation(
            external_img=external_img,
            wrist_img=wrist_img,
            gripper_open_ratio=self.gripper_open_ratio,
        )

    def _set_register(self, high_address, low_address, value, is_read_serial=True):
        return self.Hand.sendCmd(
            ModbusHighAddress=high_address,
            ModbusLowAddress=low_address,
            Value=value,
            isReadSerial=is_read_serial,
        )

    def _get_register(self, high_address, low_address):
        return self.Hand.sendCmd(
            ModbusHighAddress=high_address,
            ModbusLowAddress=low_address,
            isSet=False,
        )

    def initialize(self):
        self._set_register(0x01, 0x00, 1)
        self.wait_until_ready()

    def set_force(self, value):
        isRange(value, 20, 100)
        self._set_register(0x01, 0x01, value)

    def set_position(self, value, *, wait: bool = True):
        isRange(value, 0, 1000)
        if not wait:
            self.Hand.clear_input_buffer()
        self._set_register(0x01, 0x03, value, is_read_serial=wait)
        self._position_command = int(value)

    def set_velocity(self, value):
        isRange(value, 0, 1000)
        self._set_register(0x01, 0x04, value)

    def set_absolute_rotation(self, cmd):
        isRange(cmd, -32768, 32767)
        self._set_register(0x01, 0x05, cmd, is_read_serial=False)

    def set_rotation_velocity(self, value):
        isRange(value, 1, 100)
        self._set_register(0x01, 0x07, value, is_read_serial=False)

    def set_rotation_force(self, value):
        isRange(value, 20, 100)
        self._set_register(0x01, 0x08, value)

    def set_relative_rotation(self, cmd):
        isRange(cmd, -32768, 32767)
        self._set_register(0x01, 0x09, cmd)

    @property
    def initialization_status(self):
        return self._get_register(0x02, 0x00)

    @property
    def rotate_angle(self):
        return self._get_register(0x02, 0x08)

    @property
    def gripper_level_count(self) -> int:
        return self._gripper_level_count

    @property
    def gripper_level(self) -> int:
        return self._gripper_level

    @property
    def position_command(self) -> int:
        return int(self._position_command)

    @property
    def max_gripper_level(self) -> int:
        return self._gripper_level_count - 1

    def _level_to_position(self, level: int) -> int:
        clipped = int(np.clip(level, 0, self.max_gripper_level))
        if self.max_gripper_level <= 0:
            return 0
        return int(round((float(clipped) / float(self.max_gripper_level)) * 1000.0))

    def _position_to_level(self, position: int) -> int:
        if self.max_gripper_level <= 0:
            return 0
        clipped = float(np.clip(position, 0, 1000))
        return int(
            np.clip(
                np.rint((clipped / 1000.0) * float(self.max_gripper_level)),
                0,
                self.max_gripper_level,
            )
        )

    def set_gripper_level(self, level: int, *, wait: bool = True) -> bool:
        next_level = int(np.clip(level, 0, self.max_gripper_level))
        if next_level == self._gripper_level:
            return False
        self.set_position(self._level_to_position(next_level), wait=wait)
        self._gripper_level = next_level
        return True

    def update_from_vr(
        self,
        vr_input,
        *,
        axis_threshold: Optional[float] = None,
        step_interval_s: Optional[float] = None,
        now: Optional[float] = None,
        wait: bool = False,
    ) -> bool:
        axis = float(getattr(vr_input, "gripper_velocity_axis", 0.0))
        threshold = (
            self._vr_axis_threshold if axis_threshold is None else float(axis_threshold)
        )
        step_interval = (
            self._vr_step_interval_s
            if step_interval_s is None
            else float(step_interval_s)
        )
        now = time.monotonic() if now is None else float(now)

        if axis >= threshold:
            step_dir = 1
        elif axis <= -threshold:
            step_dir = -1
        else:
            self._last_gripper_step_dir = 0
            return False

        should_step = (
            step_dir != self._last_gripper_step_dir
            or self._last_gripper_step_time == 0.0
            or (now - self._last_gripper_step_time) >= step_interval
        )
        if not should_step:
            return False

        self._last_gripper_step_time = now
        self._last_gripper_step_dir = step_dir
        return self.set_gripper_level(self._gripper_level + step_dir, wait=wait)

    @property
    def gripper_open_ratio(self) -> np.ndarray:
        normalized = 1.0 - (float(self._position_command) / 1000.0)
        normalized = np.clip(normalized, 0.0, 1.0)
        return np.asarray([normalized], dtype=np.float32)

    @property
    def observation(self) -> DHGripperObservation:
        external_img, wrist_img = self.dual_camera_manager.get_images()
        self._observation = self._build_observation(
            external_img=external_img,
            wrist_img=wrist_img,
        )
        return self._observation

    def wait_until_ready(self, max_wait_seconds=5.0):
        deadline = time.time() + max_wait_seconds
        back = self.initialization_status
        while back == 0:
            if time.time() >= deadline:
                raise TimeoutError("Gripper initialization timed out with status 0")
            self._set_register(0x01, 0x00, 1)
            time.sleep(0.1)
            back = self.initialization_status
            print(f"initialization_status={back}")
        while back == 2:
            if time.time() >= deadline:
                raise TimeoutError("Gripper initialization timed out with status 2")
            time.sleep(0.1)
            back = self.initialization_status
            print(f"initialization_status={back}")

    def close(self) -> None:
        self.dual_camera_manager.close()
        if self.Hand.sc.is_open:
            self.Hand.sc.close()

    def __enter__(self) -> "DH5Gripper":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False


DHgripper = DH5Gripper


def main():
    import cv2
    from control.vr_input import VRInputProcess

    def show_observation(observation: DHGripperObservation) -> None:
        if observation.external_img is not None:
            external_bgr = cv2.cvtColor(observation.external_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("DHGripper External", external_bgr)
        if observation.wrist_img is not None:
            wrist_bgr = cv2.cvtColor(observation.wrist_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("DHGripper Wrist", wrist_bgr)

    parser = argparse.ArgumentParser(description="DHgripper VR 遥杆速度控制")
    parser.add_argument("--com", default="/dev/ttyUSB0", help="串口设备")
    parser.add_argument("--force", type=int, default=50, help="夹持力 20~100")
    parser.add_argument("--velocity", type=int, default=100, help="夹爪速度 0~1000")
    parser.add_argument(
        "--step-interval",
        type=float,
        default=0.08,
        help="VR 档位步进最小间隔（秒）。摇杆持续推住时按该频率逐档变化。",
    )
    parser.add_argument(
        "--axis-threshold",
        type=float,
        default=0.55,
        help="VR 摇杆触发一步档位变化的阈值，绝对值需达到该值才会动作。",
    )
    parser.add_argument(
        "--loop-hz",
        type=float,
        default=60.0,
        help="VR 模式主循环频率",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=0.2,
        help="VR 模式状态打印周期（秒）",
    )
    args = parser.parse_args()

    print(f"[DHGripper] open {args.com}")

    cv2.namedWindow("DHGripper External", cv2.WINDOW_NORMAL)
    cv2.namedWindow("DHGripper Wrist", cv2.WINDOW_NORMAL)

    with DH5Gripper(com=args.com) as gripper:
        print(f"[DHGripper] initialization_status={gripper.initialization_status}")

        gripper.set_force(args.force)
        gripper.set_velocity(args.velocity)
        print(f"[DHGripper] force={args.force}, velocity={args.velocity}")
        print("[DHGripper] 按 'q' 或 ESC 可提前退出")

        if args.loop_hz <= 0:
            raise ValueError("--loop-hz must be > 0")
        if args.step_interval <= 0:
            raise ValueError("--step-interval must be > 0")
        if not 0.0 < args.axis_threshold <= 1.0:
            raise ValueError("--axis-threshold must be in (0, 1]")

        vr = VRInputProcess()
        vr.start()
        print(
            "[DHGripper][VR] 右手柄摇杆控制夹爪: 上=闭合一档, 下=打开一档, "
            f"levels={gripper.gripper_level_count}, axis_threshold={args.axis_threshold:.2f}, "
            f"step_interval={args.step_interval:.2f}s"
        )
        print(
            "[DHGripper][VR] 当前使用 gripper_velocity_axis: "
            "正值=闭合一档, 负值=打开一档"
        )

        try:
            last_status_time = 0.0
            loop_period = 1.0 / float(args.loop_hz)

            while True:
                loop_start = time.monotonic()

                vr_state = vr.latest
                axis = float(vr_state.gripper_velocity_axis)
                gripper.update_from_vr(
                    vr_state,
                    axis_threshold=args.axis_threshold,
                    step_interval_s=args.step_interval,
                    now=loop_start,
                    wait=False,
                )

                observation = gripper.observation
                show_observation(observation)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

                if (loop_start - last_status_time) >= args.status_interval:
                    print(
                        "[DHGripper][VR] "
                        f"axis={axis:+.3f}  "
                        f"right_stick=({vr_state.right_stick_x:+.3f}, {vr_state.right_stick_y:+.3f})  "
                        f"level={gripper.gripper_level:02d}/{gripper.max_gripper_level:02d}  "
                        f"target_position={gripper.position_command:4d}",
                        end="\r",
                    )
                    last_status_time = loop_start

                elapsed = time.monotonic() - loop_start
                remaining = loop_period - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            vr.stop()
            print()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
