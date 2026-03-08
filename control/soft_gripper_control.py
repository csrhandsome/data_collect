"""新的控制夹爪的sdk,需要更新robotic_arm_controller这个类里面的控制夹爪的函数(datacollect里面的都不要多线程控制夹爪了哈哈哈)"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import threading
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

    def readSerial(self, expected_length=None):
        time.sleep(0.02)
        if expected_length is None:
            return self.sc.read_all()
        return self.sc.read(expected_length)

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


DEFAULT_POSITION_SEQUENCE = [0, 1000]


@dataclass
class DHGripperObservation:
    external_img: Optional[np.ndarray] = None
    wrist_img: Optional[np.ndarray] = None
    gripper_open_ratio: np.ndarray = field(
        default_factory=lambda: np.ones((1,), dtype=np.float32)
    )


class DHGripper(object):
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
    ):
        self.Hand = ControlRoot(com=com)
        self._position_command = 0
        self._camera_timeout_ms = int(camera_timeout_ms)
        self._camera_poll_interval_s = max(float(camera_poll_interval_s), 0.0)
        self.dual_camera_manager = DualRGBCameraManager(
            right_camera_device=right_camera_device,
            left_camera_device=left_camera_device,
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
        )
        self._observation_lock = threading.Lock()
        self._camera_stop_event = threading.Event()
        self._camera_thread: Optional[threading.Thread] = None
        self._camera_thread_error: Optional[Exception] = None
        self._observation = DHGripperObservation(
            gripper_open_ratio=self.gripper_open_ratio
        )
        self.initialize()
        self._start_camera_thread()

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

    def _refresh_observation_once(self) -> bool:
        external_img = None
        wrist_img = None

        try:
            self.dual_camera_manager.update(timeout_ms=self._camera_timeout_ms)
            external_img, wrist_img = self.dual_camera_manager.get_images()
        except Exception as exc:
            self._camera_thread_error = exc
            return False

        with self._observation_lock:
            previous = self._observation
            self._observation = self._build_observation(
                external_img=(
                    external_img if external_img is not None else previous.external_img
                ),
                wrist_img=wrist_img if wrist_img is not None else previous.wrist_img,
            )
        self._camera_thread_error = None
        return True

    def _camera_worker(self) -> None:
        try:
            self.dual_camera_manager.connect()
        except Exception as exc:
            self._camera_thread_error = exc
            return

        while not self._camera_stop_event.is_set():
            self._refresh_observation_once()
            if self._camera_stop_event.wait(self._camera_poll_interval_s):
                break

    def _start_camera_thread(self) -> None:
        if self._camera_thread is not None:
            return

        self._camera_stop_event.clear()
        self._camera_thread = threading.Thread(
            target=self._camera_worker,
            name="dh-gripper-camera",
            daemon=True,
        )
        self._camera_thread.start()

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

    def set_position(self, value):
        isRange(value, 0, 1000)
        self._set_register(0x01, 0x03, value)
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
    def gripper_open_ratio(self) -> np.ndarray:
        normalized = 1.0 - (float(self._position_command) / 1000.0)
        normalized = np.clip(normalized, 0.0, 1.0)
        return np.asarray([normalized], dtype=np.float32)

    @property
    def observation(self) -> DHGripperObservation:
        with self._observation_lock:
            cached = self._observation

        return self._build_observation(
            external_img=cached.external_img,
            wrist_img=cached.wrist_img,
        )

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
        self._camera_stop_event.set()
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=max(self._camera_timeout_ms / 1000.0, 1.0))
            self._camera_thread = None
        self.dual_camera_manager.close()
        if self.Hand.sc.is_open:
            self.Hand.sc.close()

    def __enter__(self) -> "DHGripper":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False


DHgripper = DHGripper


def main():
    import cv2

    def show_observation(observation: DHGripperObservation) -> None:
        if observation.external_img is not None:
            external_bgr = cv2.cvtColor(observation.external_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("DHGripper External", external_bgr)
        if observation.wrist_img is not None:
            wrist_bgr = cv2.cvtColor(observation.wrist_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("DHGripper Wrist", wrist_bgr)

    def pump_display(gripper: DHGripper, duration_s: float) -> bool:
        deadline = time.time() + max(duration_s, 0.0)
        while time.time() < deadline:
            show_observation(gripper.observation)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                return False
            time.sleep(0.01)
        return True

    parser = argparse.ArgumentParser(description="DHgripper 开合测试")
    parser.add_argument("--com", default="/dev/ttyUSB0", help="串口设备")
    parser.add_argument("--force", type=int, default=50, help="夹持力 20~100")
    parser.add_argument("--velocity", type=int, default=100, help="夹爪速度 0~1000")
    parser.add_argument("--sleep", type=float, default=0.5, help="每次动作后的附加等待")
    args = parser.parse_args()

    print(f"[DHGripper] open {args.com}")
    print(f"[DHGripper] test positions = {DEFAULT_POSITION_SEQUENCE}")

    cv2.namedWindow("DHGripper External", cv2.WINDOW_NORMAL)
    cv2.namedWindow("DHGripper Wrist", cv2.WINDOW_NORMAL)

    with DHGripper(com=args.com) as gripper:
        print(f"[DHGripper] initialization_status={gripper.initialization_status}")

        gripper.set_force(args.force)
        gripper.set_velocity(args.velocity)
        print(f"[DHGripper] force={args.force}, velocity={args.velocity}")
        print("[DHGripper] 按 'q' 或 ESC 可提前退出")

        for _ in range(3):
            for position in DEFAULT_POSITION_SEQUENCE:
                print(f"[DHGripper] move -> {position}")
                gripper.set_position(position)
                observation = gripper.observation
                print(f"[DHGripper] gripper_open_ratio={gripper.gripper_open_ratio}")

                show_observation(observation)
                if not pump_display(gripper, args.sleep):
                    cv2.destroyAllWindows()
                    return

                print(f"[DHGripper] rotate_angle={gripper.rotate_angle}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
