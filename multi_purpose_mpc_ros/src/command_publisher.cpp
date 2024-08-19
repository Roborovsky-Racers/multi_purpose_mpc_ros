#include "multi_purpose_mpc_ros/command_publisher.hpp"

namespace roborovsky::multi_purpose_mpc_ros
{

// Public methods

CommandPublisher::CommandPublisher(rclcpp::Node::SharedPtr node)
: node_(node)
{

  command_publisher_ = node_->create_publisher<AckermannControlCommand>("/control/command/control_cmd", 10);

  command_subscriber_ = node_->create_subscription<AckermannControlBoostCommand>(
    "ackermann_control_command",
    10,
    std::bind(&CommandPublisher::commandCallback, this, std::placeholders::_1)
  );
}

void CommandPublisher::run()
{
  rclcpp::Rate high_rate(500);
  rclcpp::Rate low_rate(20);

  while (rclcpp::ok())
  {
    command_publisher_->publish(command_.command);
    if (command_.boost_mode)
    {
      high_rate.sleep();
    }
    else
    {
      low_rate.sleep();
    }
  }

}

// Private methods

void CommandPublisher::commandCallback(const AckermannControlBoostCommand::SharedPtr msg)
{
  command_ = *msg;
}



} // namespace roborovsky::multi_purpose_mpc_ros
