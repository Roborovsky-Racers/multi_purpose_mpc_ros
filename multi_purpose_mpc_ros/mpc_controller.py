#!/usr/bin/env python3

import os
import yaml
from collections import namedtuple
from typing import List, Tuple, Optional, NamedTuple, Dict, Union
import dataclasses
from scipy import sparse
from scipy.sparse import dia_matrix
import numpy as np

# ROS 2
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from nav_msgs.msg import Odometry

# autoware
from autoware_auto_control_msgs.msg import AckermannControlCommand

# Multi_Purpose_MPC
from multi_purpose_mpc_ros.multi_purpose_mpc.src.map import Map, Obstacle
from multi_purpose_mpc_ros.multi_purpose_mpc.src.reference_path import ReferencePath
from multi_purpose_mpc_ros.multi_purpose_mpc.src.spatial_bicycle_models import BicycleModel
from multi_purpose_mpc_ros.multi_purpose_mpc.src.MPC import MPC
from multi_purpose_mpc_ros.multi_purpose_mpc.src.utils import load_waypoints

# 再帰的に dict を namedtuple に変換する関数
def convert_to_namedtuple(
        data: Union[Dict, List, NamedTuple, float, str, bool],
        tuple_name="Config"
        ) -> Union[Dict, List, NamedTuple, float, str, bool]:
    if isinstance(data, dict):
        fields = {key: convert_to_namedtuple(value, key) for key, value in data.items()}
        return namedtuple(tuple_name, fields.keys())(**fields)
    elif isinstance(data, list):
        return [convert_to_namedtuple(item, tuple_name) for item in data]
    else:
        return data

def file_exists(file_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found: " + file_path)


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


class MPCController:

    PKG_PATH: str = get_package_share_directory('multi_purpose_mpc_ros') + "/"

    def __init__(self, node: Node, config_path: str) -> None:

        self._node = node
        self._cfg = self._load_config(config_path)
        self._initialize()
        self._setup_pub_sub()


    def _load_config(self, config_path: str) -> NamedTuple:
        with open(config_path, "r") as f:
            cfg: NamedTuple = convert_to_namedtuple(yaml.safe_load(f)) # type: ignore

        # Check if the files exist
        mandatory_files = [cfg.map.yaml_path, cfg.waypoints.csv_path]
        for file_path in mandatory_files:
            file_exists(self.in_pkg_share(file_path))
        return cfg

    def _initialize(self) -> None:
        def create_ref_path() -> ReferencePath:
            map = Map(self.in_pkg_share(self._cfg.map.yaml_path))
            wp_x, wp_y = load_waypoints(self.in_pkg_share(self._cfg.waypoints.csv_path))

            cfg_ref_path = self._cfg.reference_path
            return ReferencePath(
                map,
                wp_x,
                wp_y,
                cfg_ref_path.resolution,
                cfg_ref_path.smoothing_distance,
                cfg_ref_path.max_width,
                cfg_ref_path.circular)

        def create_obstacles() -> Optional[List[Obstacle]]:
            use_csv_obstacles = self._cfg.obstacles.csv_path != ""
            if use_csv_obstacles:
                obstacles_file_path = self.in_pkg_share(self._cfg.obstacles.csv_path)
                obs_x, obs_y = load_waypoints(obstacles_file_path)
                obstacles = []
                for cx, cy in zip(obs_x, obs_y):
                    obstacles.append(Obstacle(cx=cx, cy=cy, radius=self._cfg.obstacles.radius))
                return obstacles
            else:
                return None
        def create_car(ref_path: ReferencePath) -> BicycleModel:
            cfg_model = self._cfg.bicycle_model
            return BicycleModel(
                ref_path,
                cfg_model.length,
                cfg_model.width,
                cfg_model.Ts)

        def create_mpc(car: BicycleModel) -> Tuple[MPCConfig, MPC]:
            cfg_mpc = self._cfg.mpc
            mpc_cfg = MPCConfig(
                cfg_mpc.N,
                sparse.diags(cfg_mpc.Q),
                sparse.diags(cfg_mpc.R),
                sparse.diags(cfg_mpc.QN),
                cfg_mpc.v_max,
                cfg_mpc.a_min,
                cfg_mpc.a_max,
                cfg_mpc.ay_max,
                np.deg2rad(cfg_mpc.delta_max_deg))

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

        self._reference_path = create_ref_path()
        self._obstacles = create_obstacles()
        self._car = create_car(self._reference_path)
        self._mpc_cfg, self._mpc = create_mpc(self._car)
        compute_speed_profile(self._car, self._mpc_cfg)

    def _setup_pub_sub(self) -> None:
        self._pub = self._node.create_publisher(
            AckermannControlCommand, "/control/command/control_cmd", 1)
        self._sub = self._node.create_subscription(
            Odometry, "/localization/kinematic_state", self._odom_callback, 1)

    def _odom_callback(self, msg):
        print("received odom: ", msg)

    @classmethod
    def in_pkg_share(cls, file_path: str) -> str:
        return cls.PKG_PATH + file_path
