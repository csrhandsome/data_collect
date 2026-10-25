"""Compatibility imports; new code uses BaseRunner and create_runner."""

from control.base_runner import BaseRunner
from control.runners import create_runner

RoboticArmControler = create_runner

__all__ = ["BaseRunner", "RoboticArmControler", "create_runner"]
