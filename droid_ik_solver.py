"""DROID-style IK solver using dm_control Cartesian6dVelocityEffector.

Mirrors the exact IK pipeline from the DROID codebase:
  cartesian_velocity [-1,1] → cartesian_delta → dm_control IK → joint_delta → normalized [-1,1]

Parameters match DROID defaults:
  max_lin_delta  = 0.075 m/step
  max_rot_delta  = 0.15  rad/step
  max_joint_delta = 0.2  rad/step (per joint)
  control_hz     = 15
"""

import os

import numpy as np
from dm_control import mjcf
from dm_robotics.moma.effectors import arm_effector, cartesian_6d_velocity_effector
from dm_robotics.moma.models.robots.robot_arms import robot_arm


class _FrankaArm(robot_arm.RobotArm):
    """Minimal Franka arm for dm_robotics IK, matching DROID's robot_ik/arm.py."""

    def _build(self):
        self._name = "franka"
        model_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "franka_mjcf", "panda.xml"
        )
        self._mjcf_root = mjcf.from_path(model_file)
        self._joints = self._mjcf_root.find_all("joint")
        self._bodies = self.mjcf_model.find_all("body")
        self._actuators = self.mjcf_model.find_all("actuator")
        self._wrist_site = self.mjcf_model.find("site", "wrist_site")
        self._base_site = self.mjcf_model.find("site", "base_site")

    @property
    def name(self):
        return self._name

    @property
    def joints(self):
        return self._joints

    @property
    def actuators(self):
        return self._actuators

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def base_site(self):
        return self._base_site

    @property
    def wrist_site(self):
        return self._wrist_site

    def update_state(self, physics, qpos, qvel):
        physics.bind(self._joints).qpos[:] = qpos
        physics.bind(self._joints).qvel[:] = qvel

    def set_joint_angles(self, physics, qpos):
        physics.bind(self._joints).qpos[:] = qpos

    def initialize_episode(self, physics, random_state):
        pass


class DroidIKSolver:
    """DROID-identical IK solver with joint limits, nullspace optimization, and velocity clamping."""

    def __init__(self, control_hz: float = 15.0):
        self.max_joint_delta = 0.2
        self.relative_max_joint_delta = np.array([0.2] * 7)
        self.max_lin_delta = 0.075
        self.max_rot_delta = 0.15
        self.control_hz = control_hz

        self._arm = _FrankaArm()
        self._physics = mjcf.Physics.from_mjcf_model(self._arm.mjcf_model)
        self._effector = arm_effector.ArmEffector(
            arm=self._arm, action_range_override=None, robot_name=self._arm.name
        )
        self._effector_model = cartesian_6d_velocity_effector.ModelParams(
            self._arm.wrist_site, self._arm.joints
        )
        self._effector_control = cartesian_6d_velocity_effector.ControlParams(
            control_timestep_seconds=1.0 / self.control_hz,
            max_lin_vel=self.max_lin_delta,
            max_rot_vel=self.max_rot_delta,
            joint_velocity_limits=self.relative_max_joint_delta,
            nullspace_joint_position_reference=[0] * 7,
            nullspace_gain=0.025,
            regularization_weight=1e-2,
            enable_joint_position_limits=True,
            minimum_distance_from_joint_position_limit=0.3,
            joint_position_limit_velocity_scale=0.95,
            max_cartesian_velocity_control_iterations=300,
            max_nullspace_control_iterations=300,
        )
        self._cart_effector = cartesian_6d_velocity_effector.Cartesian6dVelocityEffector(
            self._arm.name,
            self._effector,
            self._effector_model,
            self._effector_control,
        )
        self._cart_effector.after_compile(self._arm.mjcf_model, self._physics)

    def solve(
        self, cartesian_velocity: np.ndarray, qpos: np.ndarray, qvel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert cartesian velocity [-1,1] to joint delta and normalized action.

        Args:
            cartesian_velocity: (6,) in [-1, 1] — same semantics as DROID Oculus input.
            qpos: (7,) current joint positions (rad).
            qvel: (7,) current joint velocities (rad/s).

        Returns:
            joint_delta: (7,) position increment (rad/step), feed to IntegratedVelocity.set_control().
            normalized_action: (7,) in [-1, 1], record as action in dataset.
        """
        cv = np.asarray(cartesian_velocity, dtype=np.float64)
        lin, rot = cv[:3], cv[3:6]
        lin_norm = np.linalg.norm(lin)
        rot_norm = np.linalg.norm(rot)
        if lin_norm > 1:
            lin = lin / lin_norm
        if rot_norm > 1:
            rot = rot / rot_norm
        cartesian_delta = np.concatenate([
            lin * self.max_lin_delta,
            rot * self.max_rot_delta,
        ])

        self._arm.update_state(self._physics, np.asarray(qpos), np.asarray(qvel))
        self._cart_effector.set_control(self._physics, cartesian_delta)
        joint_delta = self._physics.bind(self._arm.actuators).ctrl.copy()

        normalized_action = joint_delta / self.max_joint_delta
        return joint_delta, normalized_action
