// Isolated 1 kHz, zero-order-hold joint servo regression, without ROS or hardware.
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include <Eigen/Dense>

#include "panda_cartesian_controller/joint_impedance.hpp"

int main(int argc, char** argv) {
  using namespace panda_cartesian_controller;
  const bool old_damping = argc > 1 && std::string(argv[1]) == "old";
  const double scale = argc > 2 ? std::strtod(argv[2], nullptr) : 1.0;
  const double dt = argc > 3 ? std::strtod(argv[3], nullptr) : 0.001;
  // Coupled inertia at the configured start pose, obtained from the project's
  // Panda MJCF. Frozen at the endpoint to isolate sampled-loop stability from
  // nonlinear friction/contact effects; this is not a full hardware simulation.
  Eigen::Matrix<double, 7, 7> mass;
  mass <<
      .720477083, .066615773, .728251381, -.052331207, .026003476, .005321649, -.007010203,
      .066615773, 1.441209281, .014863950, -.663444158, -.011841350, -.080830858, .000849344,
      .728251381, .014863950, .919285200, -.013793719, .017090333, .010495149, -.006264054,
      -.052331207, -.663444158, -.013793719, .996736312, .019637909, .153338672, -.001369272,
      .026003476, -.011841350, .017090333, .019637909, .032208083, .000162567, .002653282,
      .005321649, -.080830858, .010495149, .153338672, .000162567, .053040872, -.001581999,
      -.007010203, .000849344, -.006264054, -.001369272, .002653282, -.001581999, .006682652;
  const Eigen::LDLT<Eigen::Matrix<double, 7, 7>> inertia(mass);
  JointImpedanceGains gains;
  if (old_damping) {
    for (int i = 0; i < 7; ++i) {
      gains.damping[i] = 2.0 * std::sqrt(gains.stiffness[i]);
    }
  }
  JointVector start;
  start << .5, .25, -.4, .5, .3, -.3, .7;
  start *= scale;
  const double distance = start.cwiseAbs().maxCoeff();
  const double duration = std::max({2.0, 1.875 * distance / .2, std::sqrt(5.774 * distance / .2)});
  JointVector position = start, velocity = JointVector::Zero(), reference = start, desired = start;
  JointVector previous_torque = JointVector::Zero();
  JointVector endpoint_min = JointVector::Constant(1e10);
  JointVector endpoint_max = JointVector::Constant(-1e10);
  double settled = 0.0, endpoint_peak_speed = 0.0, next_command = 0.0;
  double peak_torque_change = 0.0, now = 0.0;
  bool arrived = false;
  for (int tick = 0; tick * dt <= duration + 5.0; ++tick) {
    now = tick * dt;
    if (now + 1e-10 >= next_command) {
      const double f = std::min(now / duration, 1.0);
      desired = start * (1.0 - f * f * f * (10.0 - 15.0 * f + 6.0 * f * f));
      next_command += .01;  // Python commands at 100 Hz, controller at 1 kHz.
    }
    reference = .35 * desired + .65 * reference;
    const JointVector torque = jointImpedanceTorque(
        reference, position, velocity, JointVector::Zero(), previous_torque, dt, true, gains);
    peak_torque_change = std::max(peak_torque_change, (torque - previous_torque).cwiseAbs().maxCoeff());
    previous_torque = torque;
    velocity += inertia.solve(torque) * dt;
    position += velocity * dt;
    if (now >= duration) {
      endpoint_peak_speed = std::max(endpoint_peak_speed, velocity.cwiseAbs().maxCoeff());
      endpoint_min = endpoint_min.cwiseMin(position);
      endpoint_max = endpoint_max.cwiseMax(position);
      settled = position.cwiseAbs().maxCoeff() <= .02 && velocity.cwiseAbs().maxCoeff() <= .005
                    ? settled + dt : 0.0;
      if (settled >= .2) {
        arrived = true;
        break;
      }
    }
  }
  std::cout << "{\"arrived\":" << (arrived ? "true" : "false")
            << ",\"elapsed_s\":" << now
            << ",\"endpoint_peak_speed\":" << endpoint_peak_speed
            << ",\"endpoint_peak_to_peak\":" << (endpoint_max - endpoint_min).cwiseAbs().maxCoeff()
            << ",\"end_position_error\":" << position.cwiseAbs().maxCoeff()
            << ",\"end_velocity\":" << velocity.cwiseAbs().maxCoeff()
            << ",\"peak_torque_change\":" << peak_torque_change << "}\n";
}
