#pragma once

// Control law copied from the official Jazzy controller in the previous image:
// franka_example_controllers/src/fr3/cartesian_impedance_example_controller.cpp
// (franka_ros2 4acc05c). Two adaptations are required by libfranka 0.9 / FER:
// pose comes from FrankaRobotModel because that stack has no
// FrankaCartesianPoseInterface, and updateMotionTarget() is omitted because it
// overwrites ~/equilibrium_pose with a demo trajectory.

#include <array>
#include <tuple>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <controller_interface/controller_interface.hpp>
#include <franka_semantic_components/franka_robot_model.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_buffer.hpp>

namespace panda_cartesian_controller {

class CartesianPoseImpedanceController : public controller_interface::ControllerInterface {
 public:
  using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;

 private:
  static constexpr int num_joints{7};
  static constexpr int num_cartesian_dof{6};

  struct TargetPose {
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond orientation{1.0, 0.0, 0.0, 0.0};
  };
  struct CartesianGains {
    Eigen::Matrix<double, num_cartesian_dof, num_cartesian_dof> stiffness{
        Eigen::Matrix<double, num_cartesian_dof, num_cartesian_dof>::Zero()};
    Eigen::Matrix<double, num_cartesian_dof, num_cartesian_dof> damping{
        Eigen::Matrix<double, num_cartesian_dof, num_cartesian_dof>::Zero()};
  };

  static CartesianGains buildGains(const std::array<double, num_cartesian_dof>& k);
  void readJointState();
  std::tuple<Eigen::Quaterniond, Eigen::Vector3d> currentOrientationAndTranslation();
  void equilibriumPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  rcl_interfaces::msg::SetParametersResult onParameterUpdate(
      const std::vector<rclcpp::Parameter>& parameters);
  Eigen::Matrix<double, num_cartesian_dof, 1> computeError(
      const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation,
      const Eigen::Affine3d& transform) const;

  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;
  std::string arm_id_{"panda"};
  Eigen::Matrix<double, num_joints, 1> q_{Eigen::Matrix<double, num_joints, 1>::Zero()};
  Eigen::Matrix<double, num_joints, 1> dq_{Eigen::Matrix<double, num_joints, 1>::Zero()};
  double filter_params_{0.35};
  Eigen::Matrix<double, num_cartesian_dof, num_cartesian_dof> cartesian_stiffness_;
  Eigen::Matrix<double, num_cartesian_dof, num_cartesian_dof> cartesian_damping_;
  double nullspace_stiffness_{0.5};
  Eigen::Vector3d position_d_;
  Eigen::Quaterniond orientation_d_;
  Eigen::Matrix<double, num_joints, 1> q_d_nullspace_;
  realtime_tools::RealtimeBuffer<TargetPose> target_pose_buffer_;
  realtime_tools::RealtimeBuffer<CartesianGains> cartesian_gains_buffer_;
  realtime_tools::RealtimeBuffer<double> nullspace_stiffness_buffer_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_equilibrium_pose_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;
};

}  // namespace panda_cartesian_controller
