"""Shared Cartesian impedance settings for the SDK and ROS controllers."""

import math


def impedance_parameters(control):
    parameters = {}
    for name, key, default in (
        ("filter_coefficient", "ee_filter_coeff", 0.35),
        ("translational_stiffness", "ee_translational_stiffness", 600.0),
        ("rotational_stiffness", "ee_rotational_stiffness", 30.0),
        ("nullspace_stiffness", "ee_nullspace_stiffness", 0.5),
    ):
        value = float(control.get(key, default))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"control.{key} must be finite and non-negative")
        if name == "filter_coefficient" and not 0 < value <= 1:
            raise ValueError(f"control.{key} must be in (0, 1]")
        parameters[name] = value
    return parameters
