#!/usr/bin/env python3
import rclpy
from rclpy.signals import SignalHandlerOptions
from rclpy.node import Node
from std_msgs.msg import Float32

import numpy as np
import osqp
from scipy import sparse
import matplotlib.pyplot as plt


class MPCSimulation(Node):
    def __init__(self):
        super().__init__("mpc_simulation")
        self._pub = self.create_publisher(Float32, "mpc_test", 10)
        self._timer = self.create_timer(1, self.publish)

    def publish(self):
        msg = Float32()
        msg.data = 1.0
        self.get_logger().info("Publishing: '%f'" % msg.data)
        self._pub.publish(msg)


def main(args=None):
    # rclpy.init(args=args)
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    try:
        node = MPCSimulation()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown requested")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    # Runs a talker node when this script is run directly (not through an entrypoint)
    main()
