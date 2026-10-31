#pragma once

#include <algorithm>
#include <array>

#include <Eigen/Core>

namespace panda_cartesian_controller {

using JointVector = Eigen::Matrix<double, 7, 1>;

struct JointImpedanceGains {
  // Paired gains from the pinned FER Panda move_to_start example. D = 2*sqrt(K)
  // assumes unit inertia; it is not critical damping for this coupled arm and
  // destabilizes the sampled feedback loop on its low-inertia wrist axes.
  std::array<double, 7> stiffness{600, 600, 600, 600, 250, 150, 50};
  std::array<double, 7> damping{30, 30, 30, 30, 10, 10, 5};
};

inline JointVector jointImpedanceTorque(
    const JointVector& reference, const JointVector& position, const JointVector& velocity,
    const JointVector& coriolis, const JointVector& previous_torque, double period_s,
    bool limit_rate, const JointImpedanceGains& gains = JointImpedanceGains{}) {
  JointVector torque;
  const double delta = 1000.0 * std::clamp(period_s, 0.0, 0.001);
  for (int i = 0; i < 7; ++i) {
    torque[i] = gains.stiffness[i] * (reference[i] - position[i]) -
                gains.damping[i] * velocity[i] + coriolis[i];
    if (limit_rate) {
      torque[i] = std::clamp(torque[i], previous_torque[i] - delta, previous_torque[i] + delta);
    }
  }
  return torque;
}

}  // namespace panda_cartesian_controller
