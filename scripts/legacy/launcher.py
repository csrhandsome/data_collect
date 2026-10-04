"""Shared compatibility entry. Set vr.source=xbox in YAML for a gamepad."""

import warnings

from vr_collect import main


def launch():
    warnings.warn(
        "Historical entry: use vr_collect.py --config config/panda.yaml; all device options are in YAML",
        FutureWarning,
    )
    main()
