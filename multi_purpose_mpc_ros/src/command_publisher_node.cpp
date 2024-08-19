#include "multi_purpose_mpc_ros/command_publisher.hpp"
#include <rclcpp/executors.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("command_publisher");
  auto command_publisher_node = std::make_shared<roborovsky::multi_purpose_mpc_ros::CommandPublisher>(node);

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return EXIT_SUCCESS;
}
