#!/usr/bin/env python3

import yaml
from typing import NamedTuple
import copy

# ROS 2
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

# Multi_Purpose_MPC
from multi_purpose_mpc_ros.core.map import Map
from multi_purpose_mpc_ros.core.reference_path import ReferencePath
from multi_purpose_mpc_ros.core.utils import load_waypoints, load_ref_path

# Project
from multi_purpose_mpc_ros.common import convert_to_namedtuple, file_exists


class ReferencePathVisualizer(Node):
    PKG_PATH: str = get_package_share_directory('multi_purpose_mpc_ros') + "/"

    def __init__(self, config_path: str) -> None:
        super().__init__("reference_path_visualizer")

        # Load configuration
        self._cfg = self._load_config(config_path)
        self._initialize()
        self._setup_publisher()

    def _load_config(self, config_path: str) -> NamedTuple:
        with open(config_path, "r") as f:
            cfg: NamedTuple = convert_to_namedtuple(yaml.safe_load(f))

        # Check if the files exist
        mandatory_files = [cfg.map.yaml_path, cfg.waypoints.csv_path]
        for file_path in mandatory_files:
            file_exists(self.in_pkg_share(file_path))
        return cfg

    def _initialize(self) -> None:
        def create_map() -> Map:
            return Map(self.in_pkg_share(self._cfg.map.yaml_path))

        def create_ref_path(map: Map) -> ReferencePath:
            cfg_ref_path = self._cfg.reference_path

            is_ref_path_given = cfg_ref_path.csv_path != ""
            if is_ref_path_given:
                self.get_logger().info("Using given reference path")
                wp_x, wp_y, _, _ = load_ref_path(self.in_pkg_share(self._cfg.reference_path.csv_path))
                return ReferencePath(
                    map,
                    wp_x,
                    wp_y,
                    cfg_ref_path.resolution,
                    cfg_ref_path.smoothing_distance,
                    cfg_ref_path.max_width,
                    cfg_ref_path.circular)
            else:
                self.get_logger().info("Using waypoints to create reference path")
                wp_x, wp_y = load_waypoints(self.in_pkg_share(self._cfg.waypoints.csv_path))
                return ReferencePath(
                    map,
                    wp_x,
                    wp_y,
                    cfg_ref_path.resolution,
                    cfg_ref_path.smoothing_distance,
                    cfg_ref_path.max_width,
                    cfg_ref_path.circular)

        self._map = create_map()
        self._reference_path = create_ref_path(self._map)

    def _setup_publisher(self) -> None:
        latching_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._ref_path_pub = self.create_publisher(
            MarkerArray, "/mpc/ref_path", latching_qos)

        # Timer to periodically publish the reference path
        self._timer = self.create_timer(1.0, self._publish_ref_path)

    def _publish_ref_path(self):
        self._publish_ref_path_marker(self._reference_path)

    def _publish_ref_path_marker(self, ref_path: ReferencePath):
        ref_path_marker_array = MarkerArray()
        m_base = Marker()
        m_base.header.frame_id = "map"
        m_base.ns = "ref_path"
        m_base.type = Marker.LINE_STRIP
        m_base.action = Marker.ADD
        m_base.pose.position.z = 0.0
        m_base.scale.x = 0.2
        m_base.color.a = 0.7
        m_base.color.r = 0.0
        m_base.color.g = 0.0
        m_base.color.b = 1.0
        spheres = Marker()
        spheres.header.frame_id = "map"
        spheres.ns = "ref_path_point"
        spheres.type = Marker.SPHERE_LIST
        spheres.action = Marker.ADD
        radius = 0.2
        spheres.scale.x = radius
        spheres.scale.y = radius
        spheres.scale.z = radius
        spheres.color.a = 0.7
        spheres.color.r = 1.0
        spheres.color.g = 1.0
        spheres.color.b = 0.0
        for i in range(len(ref_path.waypoints) - 1):
            m = copy.deepcopy(m_base)
            m.id = i
            start = Point()
            start.x = ref_path.waypoints[i].x
            start.y = ref_path.waypoints[i].y
            end = Point()
            end.x = ref_path.waypoints[i + 1].x
            end.y = ref_path.waypoints[i + 1].y
            m.points.append(start)
            m.points.append(end)
            ref_path_marker_array.markers.append(m)
            p = Point()
            p.x = ref_path.waypoints[i].x
            p.y = ref_path.waypoints[i].y
            p.z = 0.0
            spheres.points.append(p)
        ref_path_marker_array.markers.append(spheres)
        self._ref_path_pub.publish(ref_path_marker_array)

    @classmethod
    def in_pkg_share(cls, file_path: str) -> str:
        return cls.PKG_PATH + file_path


def main(args=None):
    rclpy.init(args=args)
    config_path = "config/your_config_file.yaml"  # Replace with your actual config file path
    node = ReferencePathVisualizer(config_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()