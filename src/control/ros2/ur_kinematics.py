"""UR serial-chain FK and local IK using the driver's calibrated robot_description.

No nominal DH constants or mesh loading: fixed URDF origins preserve calibration
and the distinction between base_link, base and tool0.
"""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from control.util.pose import as_vector, position_quat_to_matrix


class URKinematics:
    def __init__(self, description, base_frame, tool_frame, joint_names):
        robot = ET.fromstring(description)
        by_child = {joint.find("child").get("link"): joint for joint in robot.findall("joint")}
        chain, seen, link = [], set(), tool_frame
        while link != base_frame:
            if link in seen or link not in by_child:
                raise ValueError(f"No URDF chain from {base_frame} to {tool_frame}")
            seen.add(link)
            joint = by_child[link]
            chain.append(joint)
            link = joint.find("parent").get("link")
        self.chain = []
        names, lower, upper, velocity = [], [], [], []
        for joint in reversed(chain):
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            transform = np.eye(4)
            transform[:3, 3] = as_vector(xyz, 3, "URDF origin")
            transform[:3, :3] = Rotation.from_euler("xyz", as_vector(rpy, 3, "URDF rpy")).as_matrix()
            axis = None
            if joint.get("type") != "fixed":
                if joint.get("type") != "revolute":
                    raise ValueError("UR chain requires bounded revolute joints")
                names.append(joint.get("name"))
                element = joint.find("axis")
                axis = as_vector(np.fromstring(element.get("xyz", "1 0 0")
                                              if element is not None else "1 0 0", sep=" "), 3, "URDF axis")
                if np.linalg.norm(axis) < 1e-9:
                    raise ValueError("URDF joint axis must be nonzero")
                axis = axis / np.linalg.norm(axis)
                limit = joint.find("limit")
                if limit is None:
                    raise ValueError("URDF joint limits are required")
                lower.append(float(limit.get("lower")))
                upper.append(float(limit.get("upper")))
                velocity.append(float(limit.get("velocity")))
            self.chain.append((transform, axis))
        if names != list(joint_names) or len(names) != 6:
            raise ValueError("Configured UR joint order must match the six-axis URDF chain")
        self.lower, self.upper, self.velocity = map(np.array, (lower, upper, velocity))
        if (not np.isfinite([self.lower, self.upper, self.velocity]).all()
                or np.any(self.lower >= self.upper) or np.any(self.velocity <= 0)):
            raise ValueError("URDF joint limits must be finite and ordered")

    def validate(self, joints):
        joints = as_vector(joints, 6, "UR joints")
        if np.any(joints < self.lower) or np.any(joints > self.upper):
            raise ValueError("UR target exceeds URDF joint limits")
        return joints

    def forward(self, joints, *, jacobian=False):
        joints = as_vector(joints, 6, "UR joints")
        pose, columns, index = np.eye(4), [], 0
        for origin, axis in self.chain:
            pose = pose @ origin
            if axis is not None:
                columns.append((pose[:3, 3].copy(), pose[:3, :3] @ axis))
                rotation = np.eye(4)
                rotation[:3, :3] = Rotation.from_rotvec(axis * joints[index]).as_matrix()
                pose = pose @ rotation
                index += 1
        if jacobian:
            return pose, np.column_stack([
                np.r_[np.cross(axis, pose[:3, 3] - origin), axis] for origin, axis in columns
            ])
        return pose

    def inverse(self, position, quaternion, seed):
        desired = position_quat_to_matrix(position, quaternion)
        joints = self.validate(seed).copy()
        for _ in range(60):
            pose, jacobian = self.forward(joints, jacobian=True)
            error = np.r_[desired[:3, 3] - pose[:3, 3],
                          Rotation.from_matrix(desired[:3, :3] @ pose[:3, :3].T).as_rotvec()]
            if np.linalg.norm(error[:3]) < 1e-5 and np.linalg.norm(error[3:]) < 1e-4:
                return joints
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-5 * np.eye(6), error)
            delta *= min(1.0, 0.1 / max(np.max(np.abs(delta)), 1e-12))
            joints = np.clip(joints + delta, self.lower, self.upper)
        raise ValueError("UR local IK did not converge; target is unreachable or near a singularity")
