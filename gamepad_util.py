from lerobot.teleoperators.gamepad import GamepadTeleop, GamepadTeleopConfig


def _get_gamepad_inputs(teleop: GamepadTeleop) -> tuple[dict, dict]:
    gamepad = getattr(teleop, "gamepad", None)
    joystick = getattr(gamepad, "joystick", None)
    deadzone = float(getattr(gamepad, "deadzone", 0.1))

    axes: dict[str, float] = {
        "right_x": 0.0,
        "right_y": 0.0,
        "lt": 0.0,
        "rt": 0.0,
    }
    buttons: dict[str, bool] = {
        "l1": False,
        "r1": False,
        "a": False,
        "b": False,
        "x": False,
        "o": False,
    }

    if joystick is None:
        return axes, buttons

    num_axes = joystick.get_numaxes()
    num_buttons = joystick.get_numbuttons()

    def read_axis(index: int, *, apply_deadzone: bool = True) -> float:
        if index < 0 or index >= num_axes:
            return 0.0
        value = float(joystick.get_axis(index))
        if apply_deadzone and abs(value) < deadzone:
            return 0.0
        return value

    def read_button(index: int) -> bool:
        if index < 0 or index >= num_buttons:
            return False
        return bool(joystick.get_button(index))

    if num_axes >= 5:
        axes["right_y"] = read_axis(3)
        axes["right_x"] = read_axis(4)
    elif num_axes >= 4:
        axes["right_x"] = read_axis(2)
        axes["right_y"] = read_axis(3)

    def normalize_trigger(value: float) -> float:
        if value < 0.0:
            return (value + 1.0) / 2.0
        return value

    if num_axes >= 6:
        axes["lt"] = normalize_trigger(read_axis(2, apply_deadzone=False))
        axes["rt"] = normalize_trigger(read_axis(5, apply_deadzone=False))
    elif num_axes >= 5:
        combined = read_axis(2, apply_deadzone=False)
        axes["lt"] = max(0.0, -combined)
        axes["rt"] = max(0.0, combined)

    buttons["l1"] = read_button(4)
    buttons["r1"] = read_button(5)
    buttons["a"] = read_button(0)
    buttons["b"] = read_button(1)
    buttons["x"] = read_button(2)
    buttons["o"] = read_button(3)

    return axes, buttons
