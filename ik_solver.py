"""Franka IK solvers.

Includes:
  - ``DroidIKSolver``: DROID-style Cartesian velocity IK that returns joint deltas.
  - ``FrankaJointIKSolver``: target end-effector pose IK that returns joint positions.
"""

import os

import numpy as np
from dm_control import mjcf
from dm_robotics.geometry import geometry
from dm_robotics.moma.effectors import arm_effector, cartesian_6d_velocity_effector
from dm_robotics.moma.models.robots.robot_arms import robot_arm
from dm_robotics.moma.utils.ik_solver import IkSolver as _PoseIKSolver
from dm_robotics.transformations import transformations as tr


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


class FrankaJointIKSolver:
    """Solve Franka end-effector pose IK and return a 7-DoF joint configuration.

    Notes:
        - The target pose is expressed in the world / robot-base frame.
        - Quaternions default to ``wxyz`` order, which is what
          ``dm_robotics.geometry.Pose`` expects.
        - The default target element is ``wrist_site`` from ``franka_mjcf/panda.xml``.
    """

    def __init__(
        self,
        site_name: str = "wrist_site",
        *,
        linear_tol: float = 1e-3,
        angular_tol: float = 1e-3,
        max_steps: int = 100,
        num_attempts: int = 30,
        nullspace_gain: float = 0.4,
        random_seed: int = 0,
    ):
        self.linear_tol = float(linear_tol)
        self.angular_tol = float(angular_tol)
        self.max_steps = int(max_steps)
        self.num_attempts = int(num_attempts)
        self.site_name = site_name

        self._arm = _FrankaArm()
        self._physics = mjcf.Physics.from_mjcf_model(self._arm.mjcf_model)
        self._target_site = self._arm.mjcf_model.find("site", site_name)
        if self._target_site is None:
            raise ValueError(f"Unknown Franka site: {site_name}")

        self._solver = _PoseIKSolver(
            model=self._arm.mjcf_model,
            controllable_joints=self._arm.joints,
            element=self._target_site,
            nullspace_gain=float(nullspace_gain),
        )
        self._random_state = np.random.RandomState(random_seed)

    @property
    def joint_limits(self) -> np.ndarray:
        return self._physics.bind(self._arm.joints).range.copy()

    def forward_kinematics(
        self,
        qpos: np.ndarray,
        *,
        quaternion_order: str = "wxyz",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return target-site pose for a given Franka joint configuration."""
        qpos_arr = self._as_joint_array(qpos, name="qpos")
        self._arm.set_joint_angles(self._physics, qpos_arr)
        self._physics.forward()

        site_binding = self._physics.bind(self._target_site)
        position = np.array(site_binding.xpos, dtype=np.float64)
        quaternion_wxyz = np.array(
            tr.mat_to_quat(np.array(site_binding.xmat, dtype=np.float64).reshape(3, 3)),
            dtype=np.float64,
        )
        return position, self._format_quaternion(quaternion_wxyz, quaternion_order)

    def solve(
        self,
        ee_position: np.ndarray,
        ee_quaternion: np.ndarray,
        initial_joint_configuration: np.ndarray | None = None,
        nullspace_reference: np.ndarray | None = None,
        *,
        quaternion_order: str = "wxyz",
        linear_tol: float | None = None,
        angular_tol: float | None = None,
        max_steps: int | None = None,
        early_stop: bool = True,
        num_attempts: int | None = None,
        stop_on_first_successful_attempt: bool = True,
    ) -> np.ndarray | None:
        """Solve target end-effector pose IK and return a 7-DoF ``qpos``.

        Args:
            ee_position: Target position with shape ``(3,)`` in world / base frame.
            ee_quaternion: Target orientation with shape ``(4,)``.
            initial_joint_configuration: Optional initial guess for the solver.
            nullspace_reference: Optional nullspace reference joint configuration.
            quaternion_order: ``"wxyz"`` or ``"xyzw"``.
            linear_tol: Optional override for linear tolerance in meters.
            angular_tol: Optional override for angular tolerance in radians.
            max_steps: Optional override for per-attempt solver iterations.
            early_stop: Stop as soon as tolerances are met.
            num_attempts: Optional override for random restarts.
            stop_on_first_successful_attempt: Return on first valid solution.

        Returns:
            ``(7,)`` joint positions if a solution is found; otherwise ``None``.
        """
        position = self._as_position_array(ee_position, name="ee_position")
        quaternion_wxyz = self._as_quaternion_array(
            ee_quaternion,
            name="ee_quaternion",
            quaternion_order=quaternion_order,
        )
        initial_qpos = None
        if initial_joint_configuration is not None:
            initial_qpos = self._as_joint_array(
                initial_joint_configuration,
                name="initial_joint_configuration",
            )
        nullspace_qpos = None
        if nullspace_reference is not None:
            nullspace_qpos = self._as_joint_array(
                nullspace_reference,
                name="nullspace_reference",
            )

        target_pose = geometry.Pose(position=position, quaternion=quaternion_wxyz)
        solution = self._solver.solve(
            ref_pose=target_pose,
            random_state=self._random_state,
            linear_tol=self.linear_tol if linear_tol is None else float(linear_tol),
            angular_tol=self.angular_tol if angular_tol is None else float(angular_tol),
            max_steps=self.max_steps if max_steps is None else int(max_steps),
            early_stop=bool(early_stop),
            num_attempts=self.num_attempts if num_attempts is None else int(num_attempts),
            stop_on_first_successful_attempt=bool(stop_on_first_successful_attempt),
            initial_joint_configuration=None if initial_qpos is None else initial_qpos.tolist(),
            nullspace_reference=None if nullspace_qpos is None else nullspace_qpos.tolist(),
        )
        if solution is None:
            return None
        return np.asarray(solution, dtype=np.float64)

    def solve_pose(
        self,
        ee_position: np.ndarray,
        ee_quaternion: np.ndarray,
        initial_joint_configuration: np.ndarray | None = None,
        nullspace_reference: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray | None:
        """Alias for :meth:`solve` to make pose IK callsites explicit."""
        return self.solve(
            ee_position=ee_position,
            ee_quaternion=ee_quaternion,
            initial_joint_configuration=initial_joint_configuration,
            nullspace_reference=nullspace_reference,
            **kwargs,
        )

    def _as_position_array(self, value: np.ndarray, *, name: str) -> np.ndarray:
        position = np.asarray(value, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError(f"{name} must have shape (3,), got {position.shape}")
        return position

    def _as_quaternion_array(
        self,
        value: np.ndarray,
        *,
        name: str,
        quaternion_order: str,
    ) -> np.ndarray:
        quaternion = np.asarray(value, dtype=np.float64)
        if quaternion.shape != (4,):
            raise ValueError(f"{name} must have shape (4,), got {quaternion.shape}")

        if quaternion_order == "wxyz":
            quaternion_wxyz = quaternion
        elif quaternion_order == "xyzw":
            quaternion_wxyz = np.array(
                [quaternion[3], quaternion[0], quaternion[1], quaternion[2]],
                dtype=np.float64,
            )
        else:
            raise ValueError(
                f"Unsupported quaternion_order={quaternion_order!r}; use 'wxyz' or 'xyzw'"
            )

        norm = np.linalg.norm(quaternion_wxyz)
        if norm == 0.0:
            raise ValueError(f"{name} must be non-zero")
        return quaternion_wxyz / norm

    def _format_quaternion(
        self,
        quaternion_wxyz: np.ndarray,
        quaternion_order: str,
    ) -> np.ndarray:
        if quaternion_order == "wxyz":
            return quaternion_wxyz.copy()
        if quaternion_order == "xyzw":
            return np.array(
                [
                    quaternion_wxyz[1],
                    quaternion_wxyz[2],
                    quaternion_wxyz[3],
                    quaternion_wxyz[0],
                ],
                dtype=np.float64,
            )
        raise ValueError(
            f"Unsupported quaternion_order={quaternion_order!r}; use 'wxyz' or 'xyzw'"
        )

    def _as_joint_array(self, value: np.ndarray, *, name: str) -> np.ndarray:
        qpos = np.asarray(value, dtype=np.float64)
        if qpos.shape != (7,):
            raise ValueError(f"{name} must have shape (7,), got {qpos.shape}")
        return qpos
