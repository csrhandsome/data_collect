"""Gamepad input adapter for the same EE acquisition workflow."""

import time
from types import SimpleNamespace

import numpy as np

from control.pygame_gamepad import PygameGamepadTeleop


class GamepadInput:
    def __init__(self, config):
        self.config = config
        self.reader = PygameGamepadTeleop(
            deadzone=float(config.get("deadzone", 0.1)),
            joystick_index=int(config.get("joystick_index", 0)),
        )
        self.reader.connect()
        self.position = np.zeros(3)
        self.last_ns = time.monotonic_ns()
        self.sequence = 0

    @property
    def latest(self):
        action = self.reader.get_action()
        now = time.monotonic_ns()
        dt = min((now - self.last_ns) / 1e9, 0.05)
        self.last_ns = now
        joystick = self.reader.joystick

        def button(index):
            return bool(
                joystick is not None
                and index < joystick.get_numbuttons()
                and joystick.get_button(index)
            )

        enabled = button(int(self.config.get("deadman_button", 4)))
        if enabled:
            self.position += (
                np.array([action.get(f"delta_{axis}", 0) for axis in "xyz"])
                * float(self.config.get("speed_m_s", 0.1))
                * dt
            )
        self.sequence += 1
        return SimpleNamespace(
            pos_x=self.position[0],
            pos_y=self.position[1],
            pos_z=self.position[2],
            quat_x=0.0,
            quat_y=0.0,
            quat_z=0.0,
            quat_w=1.0,
            arm_enabled=enabled,
            gripper_close=button(0),
            gripper_open=button(1),
            x_pressed=button(2),
            y_pressed=button(3),
            gripper_velocity_axis=0.0,
            pose_seq=self.sequence,
            pose_monotonic_ns=now,
        )

    def stop(self):
        self.reader.disconnect()
