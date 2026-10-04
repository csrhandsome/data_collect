"""CLI/helper for the collector's independent 100 Hz robot/VR stream."""

from .read_series import read_series


def read_action_trace(root, episode_index, feature="joint_position", max_points=2000):
    return read_series(root, episode_index, f"trace_{feature}", max_points)
