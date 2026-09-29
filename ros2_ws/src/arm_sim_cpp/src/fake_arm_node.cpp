// Publishes a sine-wave JointState for the 7 Panda joints on /joint_states.
// Placeholder for the MuJoCo bridge node.

#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

using namespace std::chrono_literals;

class FakeArmNode : public rclcpp::Node
{
public:
  FakeArmNode()
  : Node("fake_arm_node")
  {
    rate_hz_ = declare_parameter<double>("rate_hz", 50.0);
    amplitude_ = declare_parameter<double>("amplitude", 0.5);

    publisher_ = create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);

    const auto period = std::chrono::duration<double>(1.0 / rate_hz_);
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      [this]() { on_timer(); });

    start_time_ = now();
    RCLCPP_INFO(get_logger(), "fake_arm_node publishing /joint_states at %.1f Hz", rate_hz_);
  }

private:
  void on_timer()
  {
    const double t = (now() - start_time_).seconds();

    sensor_msgs::msg::JointState msg;
    msg.header.stamp = now();
    msg.name = joint_names_;
    msg.position.resize(joint_names_.size());
    msg.velocity.resize(joint_names_.size());

    for (size_t i = 0; i < joint_names_.size(); ++i) {
      const double phase = static_cast<double>(i) * 0.5;
      msg.position[i] = amplitude_ * std::sin(t + phase);
      msg.velocity[i] = amplitude_ * std::cos(t + phase);
    }

    publisher_->publish(msg);
  }

  const std::vector<std::string> joint_names_ = {
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7"};

  double rate_hz_{50.0};
  double amplitude_{0.5};
  rclcpp::Time start_time_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FakeArmNode>());
  rclcpp::shutdown();
  return 0;
}
