"""Canonical Panda observation/action keys shared by recording and inference."""

OBSERVATION_FIELDS = {
    "exterior_image": None,
    "wrist_image_left": None,
    "gripper_image_left": None,
    "gripper_image_right": None,
    "joint_position": [f"joint_{i}" for i in range(7)],
    "ee_pose": ["x", "y", "z", "roll", "pitch", "yaw"],
    "ee_position": ["x", "y", "z"],
    "gripper_position": ["commanded_open_ratio"],
}
ACTION_FIELDS = {
    key: OBSERVATION_FIELDS[key] for key in ("joint_position", "ee_pose", "gripper_position")
}
FIELD_NAMES = frozenset(
    f"{group}.{key}"
    for group, fields in (("observation", OBSERVATION_FIELDS), ("action", ACTION_FIELDS))
    for key in fields
)


def validate_fields(fields, group):
    if fields is None:
        return
    known = OBSERVATION_FIELDS if group == "observation" else ACTION_FIELDS
    if not isinstance(fields, dict):
        raise ValueError(f"{group} must be a mapping of field names to booleans")
    for key, enabled in fields.items():
        if key not in known:
            raise ValueError(f"Unknown {group} field: {key}")
        if not isinstance(enabled, bool):
            raise ValueError(f"{group}.{key} must be true or false")


def features(image_hw, action_space, tactile, observation=None, action=None):
    if action_space not in ("joint", "ee"):
        raise ValueError("dataset.action_space must be joint or ee")
    result = {}
    for group, fields, selection in (
        ("observation", OBSERVATION_FIELDS, observation),
        ("action", ACTION_FIELDS, action),
    ):
        validate_fields(selection, group)
        for key, names in fields.items():
            if not (selection or {}).get(key, True):
                continue
            if key.startswith("gripper_image") and not tactile:
                continue
            result[f"{group}.{key}"] = (
                {
                    "dtype": "image",
                    "shape": (image_hw, image_hw, 3),
                    "names": ["height", "width", "channel"],
                }
                if names is None
                else {"dtype": "float32", "shape": (len(names),), "names": names}
            )
    if not result:
        raise ValueError("observation/action must enable at least one available field")
    return result
