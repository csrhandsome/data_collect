"""Shared vector validation for the alternative IK implementations."""
from control.util.pose import as_vector, normalize_quat_xyzw, quat_wxyz_to_xyzw, quat_xyzw_to_wxyz


class IKVectorInputs:
    def _as_position_array(self, value, *, name):
        return as_vector(value, 3, name)

    def _as_joint_array(self, value, *, name):
        return as_vector(value, 7, name)

    def _as_quaternion_array(self, value, *, name, quaternion_order):
        quaternion = as_vector(value, 4, name)
        if quaternion_order == 'wxyz':
            quaternion = quat_wxyz_to_xyzw(quaternion)
        elif quaternion_order != 'xyzw':
            raise ValueError("quaternion_order must be xyzw or wxyz")
        return quat_xyzw_to_wxyz(normalize_quat_xyzw(quaternion))

    def _format_quaternion(self, quaternion_wxyz, quaternion_order):
        if quaternion_order == 'wxyz':
            return as_vector(quaternion_wxyz, 4, 'quaternion')
        if quaternion_order == 'xyzw':
            return quat_wxyz_to_xyzw(quaternion_wxyz)
        raise ValueError("quaternion_order must be xyzw or wxyz")
