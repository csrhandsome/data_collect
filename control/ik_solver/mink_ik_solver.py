"""Franka pose IK solver backed by mink."""

from __future__ import annotations

from pathlib import Path

import mink
import mujoco
import numpy as np


class MinkFrankaJointIKSolver:
    """Solve Franka wrist pose IK with mink differential IK.

    The public methods intentionally mirror ``FrankaJointIKSolver`` so callers can
    switch between the dm_control and mink implementations with minimal branching.
    """

    def __init__(
        self,
        site_name: str = "wrist_site",
        *,
        linear_tol: float = 1e-3,
        angular_tol: float = 1e-3,
        max_steps: int = 100,
        num_attempts: int = 5,
        random_seed: int = 0,
        dt: float = 0.05,
        solver: str = "daqp",
        position_cost: float = 1.0,
        orientation_cost: float = 0.5,
        posture_cost: float = 1e-3,
        damping: float = 1e-6,
        lm_damping: float = 1e-3,
        limit_margin: float = 1e-3,
    ) -> None:
        self.linear_tol = float(linear_tol)
        self.angular_tol = float(angular_tol)
        self.max_steps = int(max_steps)
        self.num_attempts = int(num_attempts)
        self.site_name = site_name
        self.dt = float(dt)
        self.solver = solver
        self.damping = float(damping)
        self._rng = np.random.RandomState(random_seed)

        model_path = (
            Path(__file__).resolve().parents[2] / "franka_mjcf" / "panda.xml"
        )
        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._configuration = mink.Configuration(self._model)
        self._site_id = mujoco.mj_name2id(
            self._model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )
        if self._site_id == -1:
            raise ValueError(f"Unknown Franka site: {site_name}")

        self._frame_task = mink.FrameTask(
            frame_name=site_name,
            frame_type="site",
            position_cost=float(position_cost),
            orientation_cost=float(orientation_cost),
            gain=1.0,
            lm_damping=float(lm_damping),
        )
        self._posture_task = mink.PostureTask(
            self._model,
            cost=float(posture_cost),
            gain=1.0,
            lm_damping=float(lm_damping),
        )
        self._limits = [
            mink.ConfigurationLimit(
                self._model,
                gain=0.95,
                min_distance_from_limits=float(limit_margin),
            )
        ]

    @property
    def joint_limits(self) -> np.ndarray:
        return self._model.jnt_range.copy()

    def forward_kinematics(
        self,
        qpos: np.ndarray,
        *,
        quaternion_order: str = "wxyz",
    ) -> tuple[np.ndarray, np.ndarray]:
        qpos_arr = self._as_joint_array(qpos, name="qpos")
        self._configuration.update(qpos_arr)
        position = np.asarray(self._configuration.data.site_xpos[self._site_id]).copy()
        rotation = np.asarray(self._configuration.data.site_xmat[self._site_id]).reshape(
            3,
            3,
        )
        quaternion_wxyz = mink.SO3.from_matrix(rotation).wxyz
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
        position = self._as_position_array(ee_position, name="ee_position")
        quaternion_wxyz = self._as_quaternion_array(
            ee_quaternion,
            name="ee_quaternion",
            quaternion_order=quaternion_order,
        )
        initial_qpos = (
            self._as_joint_array(
                initial_joint_configuration,
                name="initial_joint_configuration",
            )
            if initial_joint_configuration is not None
            else np.zeros(self._model.nq, dtype=np.float64)
        )
        nullspace_qpos = (
            self._as_joint_array(nullspace_reference, name="nullspace_reference")
            if nullspace_reference is not None
            else initial_qpos
        )

        target_pose = mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(quaternion_wxyz).normalize(),
            translation=position,
        )
        self._frame_task.set_target(target_pose)
        self._posture_task.set_target(nullspace_qpos)

        active_linear_tol = self.linear_tol if linear_tol is None else float(linear_tol)
        active_angular_tol = (
            self.angular_tol if angular_tol is None else float(angular_tol)
        )
        active_max_steps = self.max_steps if max_steps is None else int(max_steps)
        active_num_attempts = (
            self.num_attempts if num_attempts is None else int(num_attempts)
        )

        best_qpos = None
        best_cost = np.inf
        for attempt in range(max(1, active_num_attempts)):
            qpos = (
                initial_qpos.copy()
                if attempt == 0
                else self._sample_joint_configuration(nullspace_qpos)
            )
            self._configuration.update(qpos)

            for _ in range(max(1, active_max_steps)):
                try:
                    velocity = mink.solve_ik(
                        self._configuration,
                        [self._frame_task, self._posture_task],
                        dt=self.dt,
                        solver=self.solver,
                        damping=self.damping,
                        safety_break=False,
                        limits=self._limits,
                    )
                except mink.NoSolutionFound:
                    break

                qpos = self._configuration.integrate(velocity, self.dt)
                qpos = np.clip(qpos, self.joint_limits[:, 0], self.joint_limits[:, 1])
                self._configuration.update(qpos)
                linear_err, angular_err = self._pose_error(target_pose)
                if (
                    early_stop
                    and linear_err <= active_linear_tol
                    and angular_err <= active_angular_tol
                ):
                    break

            linear_err, angular_err = self._pose_error(target_pose)
            if linear_err <= active_linear_tol and angular_err <= active_angular_tol:
                cost = (
                    linear_err / max(active_linear_tol, 1e-12)
                    + angular_err / max(active_angular_tol, 1e-12)
                    + 0.01 * float(np.linalg.norm(qpos - nullspace_qpos))
                )
                if cost < best_cost:
                    best_cost = cost
                    best_qpos = qpos.copy()
                if stop_on_first_successful_attempt:
                    break

        return best_qpos

    def solve_pose(
        self,
        ee_position: np.ndarray,
        ee_quaternion: np.ndarray,
        initial_joint_configuration: np.ndarray | None = None,
        nullspace_reference: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray | None:
        return self.solve(
            ee_position=ee_position,
            ee_quaternion=ee_quaternion,
            initial_joint_configuration=initial_joint_configuration,
            nullspace_reference=nullspace_reference,
            **kwargs,
        )

    def _pose_error(self, target_pose: mink.SE3) -> tuple[float, float]:
        current_pose = self._configuration.get_transform_frame_to_world(
            self.site_name,
            "site",
        )
        error = target_pose.minus(current_pose)
        return float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))

    def _sample_joint_configuration(self, reference: np.ndarray) -> np.ndarray:
        lower = self.joint_limits[:, 0]
        upper = self.joint_limits[:, 1]
        noise = self._rng.normal(loc=0.0, scale=0.35, size=self._model.nq)
        local_sample = np.clip(reference + noise, lower, upper)
        if self._rng.rand() < 0.8:
            return local_sample
        return self._rng.uniform(lower, upper)

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
