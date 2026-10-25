#include "panda_cartesian_controller/cartesian_pose_impedance_controller.hpp"

#include <algorithm>
#include <cassert>
#include <functional>

#include <franka/robot_state.h>

namespace panda_cartesian_controller {

CartesianPoseImpedanceController::CartesianGains CartesianPoseImpedanceController::buildGains(
    const std::array<double, num_cartesian_dof>& k) {
  CartesianGains g;
  for (int i = 0; i < num_cartesian_dof; ++i) {
    const double ki = std::max(0.0, k[i]);
    g.stiffness(i, i) = ki;
    g.damping(i, i) = 2.0 * std::sqrt(ki);  // critically-damped, same as upstream
  }
  return g;
}

CartesianPoseImpedanceController::CallbackReturn CartesianPoseImpedanceController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "panda");
    auto_declare<double>("nullspace_stiffness", 20.0);
    auto_declare<double>("translational_stiffness", 150.0);
    auto_declare<double>("rotational_stiffness", 10.0);
  } catch (const std::exception& exc) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", exc.what());
    return CallbackReturn::ERROR;
  }
  franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
      arm_id_ + "/robot_model", arm_id_ + "/robot_state");
  return CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
CartesianPoseImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; i++) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
CartesianPoseImpedanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  config.names.push_back(arm_id_ + "/robot_model");
  config.names.push_back(arm_id_ + "/robot_state");
  return config;
}

CartesianPoseImpedanceController::CallbackReturn CartesianPoseImpedanceController::on_configure(
    const rclcpp_lifecycle::State&) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  if (arm_id_ != "panda") {
    RCLCPP_ERROR(get_node()->get_logger(), "This Panda port only supports arm_id panda");
    return CallbackReturn::ERROR;
  }
  franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
      arm_id_ + "/robot_model", arm_id_ + "/robot_state");

  const double t_k = get_node()->get_parameter("translational_stiffness").as_double();
  const double r_k = get_node()->get_parameter("rotational_stiffness").as_double();
  const double n_k = get_node()->get_parameter("nullspace_stiffness").as_double();
  CartesianGains gains = buildGains({t_k, t_k, t_k, r_k, r_k, r_k});
  cartesian_stiffness_ = gains.stiffness;
  cartesian_damping_ = gains.damping;
  nullspace_stiffness_ = n_k;
  cartesian_gains_buffer_.initRT(gains);
  nullspace_stiffness_buffer_.initRT(n_k);
  target_pose_buffer_.initRT(TargetPose{});

  sub_equilibrium_pose_ = get_node()->create_subscription<geometry_msgs::msg::PoseStamped>(
      "~/equilibrium_pose", rclcpp::SystemDefaultsQoS(),
      std::bind(&CartesianPoseImpedanceController::equilibriumPoseCallback, this,
                std::placeholders::_1));
  param_cb_handle_ = get_node()->add_on_set_parameters_callback(std::bind(
      &CartesianPoseImpedanceController::onParameterUpdate, this, std::placeholders::_1));
  return CallbackReturn::SUCCESS;
}

CartesianPoseImpedanceController::CallbackReturn CartesianPoseImpedanceController::on_activate(
    const rclcpp_lifecycle::State&) {
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
  readJointState();
  Eigen::Quaterniond orientation_init;
  Eigen::Vector3d position_init;
  std::tie(orientation_init, position_init) = currentOrientationAndTranslation();
  orientation_init.normalize();
  position_d_ = position_init;
  orientation_d_ = orientation_init;
  q_d_nullspace_ = q_;
  TargetPose initial_target;
  initial_target.position = position_init;
  initial_target.orientation = orientation_init;
  target_pose_buffer_.initRT(initial_target);
  return CallbackReturn::SUCCESS;
}

CartesianPoseImpedanceController::CallbackReturn CartesianPoseImpedanceController::on_deactivate(
    const rclcpp_lifecycle::State&) {
  if (franka_robot_model_) {
    franka_robot_model_->release_interfaces();
  }
  return CallbackReturn::SUCCESS;
}

void CartesianPoseImpedanceController::readJointState() {
  for (int i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);
    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");
    q_(i) = position_interface.get_optional().value();
    dq_(i) = velocity_interface.get_optional().value();
  }
}

std::tuple<Eigen::Quaterniond, Eigen::Vector3d>
CartesianPoseImpedanceController::currentOrientationAndTranslation() {
  // Same column-major 4x4 extraction as FrankaCartesianPoseInterface.
  const std::array<double, 16> current_pose =
      franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector);
  Eigen::Matrix4d pose =
      Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::ColMajor>>(current_pose.data());
  return {Eigen::Quaterniond(pose.block<3, 3>(0, 0)), pose.block<3, 1>(0, 3)};
}

controller_interface::return_type CartesianPoseImpedanceController::update(
    const rclcpp::Time&, const rclcpp::Duration&) {
  readJointState();
  Eigen::Quaterniond orientation;
  Eigen::Vector3d position;
  std::tie(orientation, position) = currentOrientationAndTranslation();

  std::array<double, num_joints> coriolis_array = franka_robot_model_->getCoriolisForceVector();
  std::array<double, num_cartesian_dof * num_joints> jacobian_array =
      franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector);
  Eigen::Map<Eigen::Matrix<double, num_joints, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, num_cartesian_dof, num_joints>> jacobian(jacobian_array.data());

  Eigen::Affine3d transform = Eigen::Affine3d::Identity();
  transform.translation() = position;
  transform.rotate(orientation.toRotationMatrix());

  const TargetPose target = *target_pose_buffer_.readFromRT();
  RCLCPP_INFO_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 500,
                       "ee z measured %.4f desired %.4f target %.4f", position.z(), position_d_.z(),
                       target.position.z());
  const CartesianGains target_gains = *cartesian_gains_buffer_.readFromRT();
  const double target_nullspace_stiffness = *nullspace_stiffness_buffer_.readFromRT();
  const auto error = computeError(position, orientation, transform);

  Eigen::MatrixXd jacobian_transpose_pinv;
  constexpr double lambda = 0.2;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian.transpose(),
                                        Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType sing_vals = svd.singularValues();
  Eigen::MatrixXd singular = jacobian.transpose();
  singular.setZero();
  for (int i = 0; i < sing_vals.size(); i++) {
    singular(i, i) = sing_vals(i) / (sing_vals(i) * sing_vals(i) + lambda * lambda);
  }
  jacobian_transpose_pinv = svd.matrixV() * singular.transpose() * svd.matrixU().transpose();

  Eigen::Matrix<double, num_joints, 1> tau_task, tau_nullspace, tau_d;
  tau_task << jacobian.transpose() *
                  (-cartesian_stiffness_ * error - cartesian_damping_ * (jacobian * dq_));
  tau_nullspace << (Eigen::Matrix<double, num_joints, num_joints>::Identity() -
                    jacobian.transpose() * jacobian_transpose_pinv) *
                       (nullspace_stiffness_ * (q_d_nullspace_ - q_) -
                        2.0 * std::sqrt(nullspace_stiffness_) * dq_);
  tau_d << tau_task + tau_nullspace + coriolis;
  RCLCPP_INFO_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 500,
                       "error z %.4f stiffness zz %.1f tau %.3f", error(2), cartesian_stiffness_(2, 2),
                       tau_d.norm());

  for (int i = 0; i < num_joints; i++) {
    if (!command_interfaces_[i].set_value(tau_d[i])) {
      return controller_interface::return_type::ERROR;
    }
  }

  cartesian_stiffness_ =
      filter_params_ * target_gains.stiffness + (1.0 - filter_params_) * cartesian_stiffness_;
  cartesian_damping_ =
      filter_params_ * target_gains.damping + (1.0 - filter_params_) * cartesian_damping_;
  nullspace_stiffness_ =
      filter_params_ * target_nullspace_stiffness + (1.0 - filter_params_) * nullspace_stiffness_;
  position_d_ = filter_params_ * target.position + (1.0 - filter_params_) * position_d_;
  if (orientation_d_.coeffs().dot(target.orientation.coeffs()) < 0.0) {
    orientation_d_.coeffs() = -orientation_d_.coeffs();
  }
  orientation_d_ = orientation_d_.slerp(filter_params_, target.orientation);
  orientation_d_.normalize();
  return controller_interface::return_type::OK;
}

Eigen::Matrix<double, 6, 1> CartesianPoseImpedanceController::computeError(
    const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation,
    const Eigen::Affine3d& transform) const {
  Eigen::Matrix<double, num_cartesian_dof, 1> error;
  error.head(3) << position - position_d_;
  Eigen::Quaterniond orientation_corrected = orientation;
  if (orientation_d_.coeffs().dot(orientation_corrected.coeffs()) < 0.0) {
    orientation_corrected.coeffs() = -orientation_corrected.coeffs();
  }
  Eigen::Quaterniond error_quaternion(orientation_corrected.inverse() * orientation_d_);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  error.tail(3) << -transform.rotation() * error.tail(3);
  return error;
}

void CartesianPoseImpedanceController::equilibriumPoseCallback(
    const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  TargetPose target;
  target.position << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  target.orientation = Eigen::Quaterniond(msg->pose.orientation.w, msg->pose.orientation.x,
                                          msg->pose.orientation.y, msg->pose.orientation.z);
  if (target.orientation.coeffs().norm() < 1e-6) {
    RCLCPP_WARN(get_node()->get_logger(), "Discarding equilibrium pose with degenerate quaternion.");
    return;
  }
  target.orientation.normalize();
  target_pose_buffer_.writeFromNonRT(target);
}

rcl_interfaces::msg::SetParametersResult CartesianPoseImpedanceController::onParameterUpdate(
    const std::vector<rclcpp::Parameter>& parameters) {
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  for (const auto& parameter : parameters) {
    if (parameter.get_name() == "nullspace_stiffness") {
      const double value = parameter.as_double();
      if (!std::isfinite(value) || value < 0.0) {
        result.successful = false;
        result.reason = "nullspace_stiffness must be finite and non-negative";
        return result;
      }
      nullspace_stiffness_buffer_.writeFromNonRT(value);
    }
  }
  return result;
}

}  // namespace panda_cartesian_controller

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(panda_cartesian_controller::CartesianPoseImpedanceController,
                       controller_interface::ControllerInterface)
