from __future__ import annotations

class PygameGamepadTeleop:
    """Minimal pygame-based gamepad reader compatible with existing teleop usage."""

    def __init__(self, *, deadzone: float = 0.1, joystick_index: int = 0) -> None:
        self.deadzone = float(deadzone)
        self.joystick_index = int(joystick_index)
        self.joystick = None
        self._pygame = None

    def connect(self) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "pygame is required for gamepad input; install pygame first."
            ) from exc

        self._pygame = pygame
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() <= self.joystick_index:
            self._cleanup_pygame()
            raise RuntimeError("No gamepad detected. Connect a controller and retry.")

        self.joystick = pygame.joystick.Joystick(self.joystick_index)
        self.joystick.init()

    def disconnect(self) -> None:
        pygame = self._pygame
        if pygame is None:
            self.joystick = None
            return

        if self.joystick is not None:
            try:
                self.joystick.quit()
            except Exception:
                pass
            self.joystick = None

        if pygame.joystick.get_init():
            pygame.joystick.quit()
        pygame.quit()
        self._pygame = None

    def update(self) -> None:
        pygame = self._pygame
        if pygame is None:
            return
        pygame.event.pump()

    def get_action(self) -> dict[str, float]:
        if self.joystick is None:
            return {}
        self.update()
        delta_x, delta_y, delta_z = self._get_deltas()
        return {"delta_x": delta_x, "delta_y": delta_y, "delta_z": delta_z}

    def _cleanup_pygame(self) -> None:
        pygame = self._pygame
        if pygame is None:
            return
        if pygame.joystick.get_init():
            pygame.joystick.quit()
        pygame.quit()
        self._pygame = None

    def _read_axis(self, index: int, *, apply_deadzone: bool = True) -> float:
        if self.joystick is None:
            return 0.0
        if index < 0 or index >= self.joystick.get_numaxes():
            return 0.0
        try:
            value = float(self.joystick.get_axis(index))
        except Exception:
            return 0.0
        if apply_deadzone and abs(value) < self.deadzone:
            return 0.0
        return value

    def _get_deltas(self) -> tuple[float, float, float]:
        # Match the lerobot pygame mapping to keep behavior consistent.
        y_input = self._read_axis(0)
        x_input = self._read_axis(1)
        z_input = self._read_axis(3)

        delta_x = -x_input
        delta_y = -y_input
        delta_z = -z_input

        return delta_x, delta_y, delta_z
