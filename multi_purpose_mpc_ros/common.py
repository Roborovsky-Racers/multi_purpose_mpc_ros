import rclpy
import argparse

def parse_args_without_ros(argv):
    args_without_ros = rclpy.utilities.remove_ros_args(argv) # type: ignore
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True,
                        help="Path to the config.yaml file")
    return parser.parse_args(args_without_ros[1:])