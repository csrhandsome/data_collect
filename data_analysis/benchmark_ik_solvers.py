"""Benchmark dm_control and mink Franka pose IK solvers.

This is an offline benchmark: it never connects to the real robot. Targets are
generated from the Franka MJCF by forward kinematics, then both solvers try to
recover a joint configuration for the same target pose.

Example:
uv run -m data_analysis.benchmark_ik_solvers --samples 120 --attempts 3
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np

from control.ik_solver.dm_control_ik_solver import FrankaJointIKSolver
from control.ik_solver.mink_ik_solver import MinkFrankaJointIKSolver


HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)


from control.util.pose import quat_angle_xyzw as _quat_angle_error


def _sample_qpos(
    rng: np.random.RandomState,
    joint_limits: np.ndarray,
    mode: str,
    previous_qpos: np.ndarray,
) -> np.ndarray:
    lower = joint_limits[:, 0]
    upper = joint_limits[:, 1]
    if mode == "nominal":
        center = 0.5 * (lower + upper)
        half_range = 0.5 * (upper - lower)
        return rng.uniform(center - 0.55 * half_range, center + 0.55 * half_range)
    if mode == "local":
        noise = rng.normal(0.0, 0.12, size=7)
        return np.clip(previous_qpos + noise, lower, upper)
    if mode == "near_limits":
        qpos = rng.uniform(lower, upper)
        joints = rng.choice(7, size=2, replace=False)
        for joint in joints:
            span = upper[joint] - lower[joint]
            if rng.rand() < 0.5:
                qpos[joint] = lower[joint] + rng.uniform(0.005, 0.08) * span
            else:
                qpos[joint] = upper[joint] - rng.uniform(0.005, 0.08) * span
        return qpos
    raise ValueError(f"Unknown sample mode: {mode}")


def _solve_one(
    name: str,
    solver,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    initial_qpos: np.ndarray,
    attempts: int,
) -> dict[str, float | int | str]:
    start = time.perf_counter()
    solution = solver.solve_pose(
        target_pos,
        target_quat,
        initial_joint_configuration=initial_qpos,
        nullspace_reference=initial_qpos,
        early_stop=True,
        num_attempts=attempts,
        stop_on_first_successful_attempt=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    row: dict[str, float | int | str] = {
        "solver": name,
        "success": int(solution is not None),
        "elapsed_ms": elapsed_ms,
        "linear_error_m": float("nan"),
        "angular_error_rad": float("nan"),
        "joint_delta_norm": float("nan"),
        "max_joint_delta": float("nan"),
    }
    if solution is None:
        return row

    actual_pos, actual_quat = solver.forward_kinematics(solution)
    joint_delta = np.asarray(solution, dtype=np.float64) - initial_qpos
    row.update(
        {
            "linear_error_m": float(np.linalg.norm(actual_pos - target_pos)),
            "angular_error_rad": _quat_angle_error(actual_quat, target_quat),
            "joint_delta_norm": float(np.linalg.norm(joint_delta)),
            "max_joint_delta": float(np.max(np.abs(joint_delta))),
        }
    )
    return row


def _summarize(rows: list[dict[str, float | int | str]]) -> None:
    print("\nSummary")
    print("-" * 88)
    header = (
        f"{'case':<13} {'solver':<10} {'success':>9} {'time_ms':>10} "
        f"{'pos_mm':>10} {'ang_deg':>10} {'dq_norm':>10}"
    )
    print(header)
    print("-" * 88)

    cases = sorted({str(row["case"]) for row in rows})
    solvers = sorted({str(row["solver"]) for row in rows})
    for case in cases:
        for solver in solvers:
            subset = [
                row
                for row in rows
                if row["case"] == case and row["solver"] == solver
            ]
            success = [row for row in subset if int(row["success"]) == 1]
            success_rate = len(success) / max(len(subset), 1)
            time_ms = statistics.fmean(float(row["elapsed_ms"]) for row in subset)
            if success:
                pos_mm = statistics.fmean(
                    float(row["linear_error_m"]) * 1000.0 for row in success
                )
                ang_deg = statistics.fmean(
                    np.degrees(float(row["angular_error_rad"])) for row in success
                )
                dq_norm = statistics.fmean(float(row["joint_delta_norm"]) for row in success)
                print(
                    f"{case:<13} {solver:<10} {success_rate:>8.1%} "
                    f"{time_ms:>10.2f} {pos_mm:>10.3f} {ang_deg:>10.3f} {dq_norm:>10.3f}"
                )
            else:
                print(
                    f"{case:<13} {solver:<10} {success_rate:>8.1%} "
                    f"{time_ms:>10.2f} {'nan':>10} {'nan':>10} {'nan':>10}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Franka IK solvers.")
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument(
        "--cases",
        type=str,
        default="nominal,local,near_limits",
        help="Comma-separated cases to run: nominal,local,near_limits.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_analysis/quality_reports/ik_solver_benchmark.csv"),
    )
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be > 0")
    if args.attempts <= 0:
        raise ValueError("--attempts must be > 0")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")

    dm_solver = FrankaJointIKSolver(
        linear_tol=2e-3,
        angular_tol=5e-3,
        max_steps=args.max_steps,
        num_attempts=args.attempts,
    )
    mink_solver = MinkFrankaJointIKSolver(
        linear_tol=2e-3,
        angular_tol=5e-3,
        max_steps=args.max_steps,
        num_attempts=args.attempts,
    )
    solvers = {
        "dm_control": dm_solver,
        "mink": mink_solver,
    }

    rng = np.random.RandomState(args.seed)
    joint_limits = dm_solver.joint_limits
    valid_cases = {"nominal", "local", "near_limits"}
    cases = tuple(case.strip() for case in args.cases.split(",") if case.strip())
    if not cases:
        raise ValueError("--cases must include at least one case")
    unknown_cases = sorted(set(cases) - valid_cases)
    if unknown_cases:
        raise ValueError(f"Unknown --cases entries: {unknown_cases}")
    per_case = int(np.ceil(args.samples / len(cases)))
    rows: list[dict[str, float | int | str]] = []
    previous_qpos = HOME_QPOS.copy()

    for case in cases:
        for sample_idx in range(per_case):
            if len(rows) >= args.samples * len(solvers):
                break
            target_qpos = _sample_qpos(rng, joint_limits, case, previous_qpos)
            initial_qpos = np.clip(
                target_qpos + rng.normal(0.0, 0.18, size=7),
                joint_limits[:, 0],
                joint_limits[:, 1],
            )
            target_pos, target_quat = dm_solver.forward_kinematics(target_qpos)
            previous_qpos = target_qpos

            for solver_name, solver in solvers.items():
                row = _solve_one(
                    solver_name,
                    solver,
                    target_pos,
                    target_quat,
                    initial_qpos,
                    args.attempts,
                )
                row["case"] = case
                row["sample"] = sample_idx
                rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "sample",
        "solver",
        "success",
        "elapsed_ms",
        "linear_error_m",
        "angular_error_rad",
        "joint_delta_norm",
        "max_joint_delta",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _summarize(rows)
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
