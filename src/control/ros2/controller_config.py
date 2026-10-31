"""Generate launch parameters from the same tuning used by panda-py."""

import tempfile
from pathlib import Path

import yaml

from control.common.impedance import impedance_parameters


def write_controller_config(base_path, config, output_path=None):
    controllers = yaml.safe_load(Path(base_path).read_text())
    controllers["cartesian_pose_impedance_controller"]["ros__parameters"].update(
        impedance_parameters(config.get("control", {}))
    )
    if output_path is None:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="data-collect-controllers-", suffix=".yaml", delete=False
        ) as output:
            yaml.safe_dump(controllers, output)
            return output.name
    Path(output_path).write_text(yaml.safe_dump(controllers))
    return str(output_path)
