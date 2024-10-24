#!/usr/bin/env python3

from typing import Dict, List, Tuple, Optional, NamedTuple

import copy
import numpy as np
import yaml
from collections import OrderedDict

# ROS 2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

# Multi_Purpose_MPC
from multi_purpose_mpc_ros.core.reference_path import ReferencePath
from multi_purpose_mpc_ros.tools.reference_path_generator import ReferencePathGenerator
from multi_purpose_mpc_ros.common import convert_to_namedtuple, file_exists

RED = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
YELLOW = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
CYAN = ColorRGBA(r=0.0, g=156.0 / 255.0, b=209.0 / 255.0, a=1.0)
WHITE = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)

class ReferenceVelocityConfigulator(Node):
    MIN_VELOCITY = 20.0 # km/h
    MAX_VELOCITY = 30.0 # km/h

    def __init__(
            self,
            ref_path_config_path: str,
            ref_vel_config_path: str) -> None:
        super().__init__("reference_velocity_configulator") # type: ignore
        self._reference_path = ReferencePathGenerator.get_reference_path(ref_path_config_path)
        self._cfg: Dict = self._load_config(ref_vel_config_path)
        self._setup_publisher()
        self._register_params()
        self._timer = self.create_timer(2.0, self._timer_callback)

    @classmethod
    def _load_config(cls, config_path: str) -> Dict:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg # type: ignore

    def _register_params(self):
        def declatre_parameters():
            for wp_name, ref_vel in self._cfg.items():
                self.declare_parameter(f"{wp_name}", ref_vel)

        def param_cb(parameters):
            for param in parameters:
                self._cfg[param.name] = param.value
            self._update_marker()
            return SetParametersResult(successful=True)

        declatre_parameters()
        self.add_on_set_parameters_callback(param_cb)

    def _setup_publisher(self) -> None:
        latching_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._ref_vel_marker_pub = self.create_publisher(
            MarkerArray, "/ref_vel_marker", latching_qos)

    def _timer_callback(self) -> None:
        self._update_marker()

    def _update_marker(self):

        def add_ref_path_marker(markers: MarkerArray, section_ref_vel_map: Dict) -> None:
            line_base = Marker()
            line_base.header.frame_id = "map"
            line_base.ns = "ref_path"
            line_base.type = Marker.LINE_STRIP
            line_base.action = Marker.ADD
            line_base.pose.position.z = 0.0
            line_base.scale.x = 0.2

            ref_path = self._reference_path
            current_section_idx = -1
            section_ref_vel_list = list(section_ref_vel_map.items())
            _, current_ref_vel = section_ref_vel_list[current_section_idx]

            for i in range(len(ref_path.waypoints) - 1):
                if (current_section_idx < len(section_ref_vel_list)-1) and \
                    (i >= section_ref_vel_list[current_section_idx + 1][0]):
                    current_section_idx += 1
                    _, current_ref_vel = section_ref_vel_list[current_section_idx]
                line = copy.deepcopy(line_base)
                line.id = i
                line.color = self.create_vel_heat_color(current_ref_vel)
                start = Point()
                start.x = ref_path.waypoints[i].x
                start.y = ref_path.waypoints[i].y
                end = Point()
                end.x = ref_path.waypoints[i + 1].x
                end.y = ref_path.waypoints[i + 1].y
                line.points.append(start) # type: ignore
                line.points.append(end) # type: ignore
                markers.markers.append(line) # type: ignore

        def add_section_markers(markers: MarkerArray, section_ref_vel_map: Dict) -> None:
            spheres = Marker()
            spheres.header.frame_id = "map"
            spheres.ns = "section_start_point"
            spheres.type = Marker.SPHERE_LIST
            spheres.action = Marker.ADD
            radius = 1.0
            spheres.scale = Vector3(x=radius, y=radius, z=radius)
            spheres.color = YELLOW

            text_base = Marker()
            text_base.header.frame_id = "map"
            text_base.type = Marker.TEXT_VIEW_FACING
            text_base.action = Marker.ADD
            text_base.pose.position.z = 0.0
            text_base.scale.z = 1.8

            for wp_idx, ref_vel in section_ref_vel_map.items():
                p = Point()
                p.x = self._reference_path.waypoints[wp_idx].x
                p.y = self._reference_path.waypoints[wp_idx].y
                p.z = 10.
                spheres.points.append(p) #type: ignore

                text = copy.deepcopy(text_base)
                text.ns = f"ref_vel_{wp_idx}"
                text.pose.position = copy.deepcopy(p)
                text.pose.position.x += 2.0
                text.pose.position.y += 2.0
                text.text = f"wp{wp_idx}:\n{ref_vel:.2f} kmph"
                text.color = self.create_vel_heat_color(ref_vel)
                markers.markers.append(text) # type: ignore

            markers.markers.append(spheres) # type: ignore

        section_ref_vel_map = OrderedDict()
        for wp_name, ref_vel in sorted(self._cfg.items(), key=lambda x: int(x[0].split("_")[-1])):
            idx = wp_name.split("_")[-1]
            section_ref_vel_map[int(idx)] = ref_vel

        ref_vel_marker_array = MarkerArray()
        add_ref_path_marker(ref_vel_marker_array, section_ref_vel_map)
        add_section_markers(ref_vel_marker_array, section_ref_vel_map)
        self._ref_vel_marker_pub.publish(ref_vel_marker_array)

    @classmethod
    def create_vel_heat_color(cls, vel_norm: float, alpha: float = 1.0) -> ColorRGBA:
        normalized_velocity = max(
            0.0,
            min(
                (vel_norm - cls.MIN_VELOCITY) / (cls.MAX_VELOCITY - cls.MIN_VELOCITY),
                1.0)
            )
        r = normalized_velocity
        g = 0.2
        b = 1.0 - normalized_velocity
        return ColorRGBA(r=r, g=g, b=b, a=alpha)

def main(args=None):
    rclpy.init(args=args)
    config_path = "config/ref_vel.yaml"
    node = ReferencePathGenerator(config_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()