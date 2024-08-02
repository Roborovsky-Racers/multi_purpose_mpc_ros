#!/usr/bin/env python3

import yaml
from typing import List, Tuple, Optional, NamedTuple
import dataclasses
from scipy import sparse
from scipy.sparse import dia_matrix
import numpy as np

# ROS 2
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rclpy.clock import Clock, ClockType

from nav_msgs.msg import Odometry

# autoware
from autoware_auto_control_msgs.msg import AckermannControlCommand, AckermannLateralCommand, LongitudinalCommand

# Multi_Purpose_MPC
from multi_purpose_mpc_ros.core.map import Map, Obstacle
from multi_purpose_mpc_ros.core.reference_path import ReferencePath
from multi_purpose_mpc_ros.core.spatial_bicycle_models import BicycleModel
from multi_purpose_mpc_ros.core.MPC import MPC
from multi_purpose_mpc_ros.core.utils import load_waypoints, kmh_to_m_per_sec

# Project
from multi_purpose_mpc_ros.common import convert_to_namedtuple, file_exists

def array_to_ackermann_control_command(u: np.ndarray) -> AckermannControlCommand:
    msg = AckermannControlCommand()
    # TODO: Implement the conversion
    return msg

@dataclasses.dataclass
class MPCConfig:
    N: int
    Q: dia_matrix
    R: dia_matrix
    QN: dia_matrix
    v_max: float
    a_min: float
    a_max: float
    ay_max: float
    delta_max: float
    control_rate: float


class MPCController(Node):

    PKG_PATH: str = get_package_share_directory('multi_purpose_mpc_ros') + "/"

    def __init__(self, config_path: str) -> None:
        super().__init__("mpc_controller") # type: ignore

        self._cfg = self._load_config(config_path)
        self._odom: Optional[Odometry] = None
        self._initialize()
        self._setup_pub_sub()

    def _load_config(self, config_path: str) -> NamedTuple:
        with open(config_path, "r") as f:
            cfg: NamedTuple = convert_to_namedtuple(yaml.safe_load(f)) # type: ignore

        # Check if the files exist
        mandatory_files = [cfg.map.yaml_path, cfg.waypoints.csv_path] # type: ignore
        for file_path in mandatory_files:
            file_exists(self.in_pkg_share(file_path))
        return cfg

    def _initialize(self) -> None:
        def create_map() -> Map:
            return Map(self.in_pkg_share(self._cfg.map.yaml_path)) # type: ignore

        def create_ref_path(map: Map) -> ReferencePath:
            wp_x, wp_y = load_waypoints(self.in_pkg_share(self._cfg.waypoints.csv_path)) # type: ignore

            cfg_ref_path = self._cfg.reference_path # type: ignore
            return ReferencePath(
                map,
                wp_x,
                wp_y,
                cfg_ref_path.resolution,
                cfg_ref_path.smoothing_distance,
                cfg_ref_path.max_width,
                cfg_ref_path.circular)

        def create_obstacles() -> Optional[List[Obstacle]]:
            use_csv_obstacles = self._cfg.obstacles.csv_path != "" # type: ignore
            if use_csv_obstacles:
                obstacles_file_path = self.in_pkg_share(self._cfg.obstacles.csv_path) # type: ignore
                obs_x, obs_y = load_waypoints(obstacles_file_path)
                obstacles = []
                for cx, cy in zip(obs_x, obs_y):
                    obstacles.append(Obstacle(cx=cx, cy=cy, radius=self._cfg.obstacles.radius)) # type: ignore
                return obstacles
            else:
                return None
        def create_car(ref_path: ReferencePath) -> BicycleModel:
            cfg_model = self._cfg.bicycle_model # type: ignore
            return BicycleModel(
                ref_path,
                cfg_model.length,
                cfg_model.width,
                cfg_model.Ts)

        def create_mpc(car: BicycleModel) -> Tuple[MPCConfig, MPC]:
            cfg_mpc = self._cfg.mpc # type: ignore
            mpc_cfg = MPCConfig(
                cfg_mpc.N,
                sparse.diags(cfg_mpc.Q),
                sparse.diags(cfg_mpc.R),
                sparse.diags(cfg_mpc.QN),
                kmh_to_m_per_sec(cfg_mpc.v_max),
                cfg_mpc.a_min,
                cfg_mpc.a_max,
                cfg_mpc.ay_max,
                np.deg2rad(cfg_mpc.delta_max_deg),
                cfg_mpc.control_rate)

            state_constraints = {
                "xmin": np.array([-np.inf, -np.inf, -np.inf]),
                "xmax": np.array([np.inf, np.inf, np.inf])}
            input_constraints = {
                "umin": np.array([0.0, -np.tan(mpc_cfg.delta_max) / car.length]),
                "umax": np.array([mpc_cfg.v_max, np.tan(mpc_cfg.delta_max) / car.length])}
            mpc = MPC(
                car,
                mpc_cfg.N,
                mpc_cfg.Q,
                mpc_cfg.R,
                mpc_cfg.QN,
                state_constraints,
                input_constraints,
                mpc_cfg.ay_max)
            return mpc_cfg, mpc

        def compute_speed_profile(car: BicycleModel, mpc_config: MPCConfig) -> None:
            speed_profile_constraints = {
                "a_min": mpc_config.a_min, "a_max": mpc_config.a_max,
                "v_min": 0.0, "v_max": mpc_config.v_max, "ay_max": mpc_config.ay_max}
            car.reference_path.compute_speed_profile(speed_profile_constraints)

        self._map = create_map()
        self._reference_path = create_ref_path(self._map)
        self._obstacles = create_obstacles()
        self._car = create_car(self._reference_path)
        self._mpc_cfg, self._mpc = create_mpc(self._car)
        compute_speed_profile(self._car, self._mpc_cfg)

    def _setup_pub_sub(self) -> None:
        self._command_pub = self.create_publisher(
            AckermannControlCommand, "/control/command/control_cmd", 1)
        self._odom_sub = self.create_subscription(
            Odometry, "/localization/kinematic_state", self._odom_callback, 1)

    def _odom_callback(self, msg):
        self._odom = msg

    def _wait_until_odom_received(self, timeout: float = 30.) -> None:
        t_start = Clock(clock_type=ClockType.ROS_TIME).now()
        rate = self.create_rate(30)
        while self._odom is None:
            now = Clock(clock_type=ClockType.ROS_TIME).now()
            if (now - t_start).nanoseconds < timeout * 1e9:
                raise TimeoutError("Timeout while waiting for odometry message")
            rate.sleep()

    def run(self) -> None:
        self._wait_until_odom_received()
        control_rate = self.create_rate(self._mpc_cfg.control_rate)

        while rclpy.ok():
            u: np.ndarray = self._mpc.get_control()
            self._car.drive(u)
            self._command_pub.publish(array_to_ackermann_control_command(u))

            control_rate.sleep()

    @classmethod
    def in_pkg_share(cls, file_path: str) -> str:
        return cls.PKG_PATH + file_path
