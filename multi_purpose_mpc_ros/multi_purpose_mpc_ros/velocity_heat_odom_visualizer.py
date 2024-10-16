#!/usr/bin/env python3

import numpy as np
import copy

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from std_msgs.msg import ColorRGBA


class VelocityHeatOdomVisualizer(Node):
    # MAX_VELOCITY = 8.3 # m/s (30 km/h)
    MAX_VELOCITY = 6.94 # m/s (25 km/h)
    MIN_VELOCITY = 2.77 # m/s (10 km/h)
    PUB_RATE = 4.0 # Hz
    MARKER_BUFFER_SIZE = 130

    def __init__(self) -> None:
        super().__init__("velocity_heat_odom_visualizer")

        self._markers = MarkerArray()
        self._setup_pub_sub()
        self._last_publish_time = self.get_clock().now()
        self._marker_id = 0

    def _setup_pub_sub(self) -> None:
        latching_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._marker_pub = self.create_publisher(
            MarkerArray, "/vel_heat_odom_marker", latching_qos)

        self._odom_sub = self.create_subscription(
            Odometry, "/localization/kinematic_state", self._odom_callback, 1)

    def _odom_callback(self, msg: Odometry) -> None:
        self._publish_velocity_heat_odom(msg)

    def _publish_velocity_heat_odom(self, msg: Odometry) -> None:
        if (self.get_clock().now() - self._last_publish_time).nanoseconds / 1e9 < 1.0 / self.PUB_RATE:
            return
        self._last_publish_time = self.get_clock().now()
        marker_id = self._marker_id % self.MARKER_BUFFER_SIZE
        self._marker_id += 1

        velocity_norm = np.linalg.norm([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
        normalized_velocity = max(0.0, min((velocity_norm - self.MIN_VELOCITY) / (self.MAX_VELOCITY - self.MIN_VELOCITY), 1.0))
        r = normalized_velocity
        g = 0.2
        b = 1.0 - normalized_velocity

        m = Marker()
        m.header.frame_id = "map"
        m.ns = f"vel_heat_odom_{marker_id}"
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.pose = copy.deepcopy(msg.pose.pose)
        m.pose.position.z = 100.0
        m.pose.orientation = msg.pose.pose.orientation
        m.scale.x = 2.0
        m.scale.y = 0.6
        m.scale.z = 1.0
        m.color = ColorRGBA(r=r, g=g, b=b, a=1.0)

        text = Marker()
        text.header.frame_id = "map"
        text.ns = f"vel_heat_odom_text_{marker_id}"
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose = copy.deepcopy(msg.pose.pose)
        text.pose.position.x += 1.0
        text.pose.position.y += 1.0
        text.pose.position.z = 101.0
        text.scale.z = 1.0
        text.text = f"{velocity_norm * 3.6:.2f}"
        text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)

        self._markers.markers.append(m)
        self._markers.markers.append(text)
        if len(self._markers.markers) > self.MARKER_BUFFER_SIZE * 2:
            self._markers.markers.pop(0)
            self._markers.markers.pop(0)
        self._marker_pub.publish(self._markers)
