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
from rclpy.parameter import Parameter

from std_msgs.msg import Bool, Float64MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Pose2D

# autoware
from autoware_auto_control_msgs.msg import AckermannControlCommand

# Multi_Purpose_MPC
from multi_purpose_mpc_ros.core.map import Map, Obstacle
from multi_purpose_mpc_ros.core.reference_path import ReferencePath
from multi_purpose_mpc_ros.core.spatial_bicycle_models import BicycleModel
from multi_purpose_mpc_ros.core.MPC import MPC
from multi_purpose_mpc_ros.core.utils import load_waypoints, kmh_to_m_per_sec

# Project
from multi_purpose_mpc_ros.common import convert_to_namedtuple, file_exists
from multi_purpose_mpc_ros.simulation_logger import SimulationLogger

def array_to_ackermann_control_command(stamp, u: np.ndarray, a_max: float) -> AckermannControlCommand:
    msg = AckermannControlCommand()
    msg.stamp = stamp
    msg.lateral.stamp = stamp
    msg.lateral.steering_tire_angle = u[1]
    msg.lateral.steering_tire_rotation_rate = 2.0
    msg.longitudinal.stamp = stamp
    msg.longitudinal.speed = u[0]
    msg.longitudinal.acceleration = a_max
    return msg

def yaw_from_quaternion(q: Quaternion):
    sqx = q.x * q.x
    sqy = q.y * q.y
    sqz = q.z * q.z
    sqw = q.w * q.w

    # Cases derived from https://orbitalstation.wordpress.com/tag/quaternion/
    sarg = -2 * (q.x*q.z - q.w*q.y) / (sqx + sqy + sqz + sqw) # normalization added from urdfom_headers

    if sarg <= -0.99999:
        yaw = -2. * np.arctan2(q.y, q.x)
    elif sarg >= 0.99999:
        yaw = 2. * np.arctan2(q.y, q.x)
    else:
        yaw = np.arctan2(2. * (q.x*q.y + q.w*q.z), sqw + sqx - sqy - sqz)

    return yaw

def odom_to_pose_2d(odom: Odometry) -> Pose2D:
    pose = Pose2D()
    pose.x = odom.pose.pose.position.x
    pose.y = odom.pose.pose.position.y
    pose.theta = yaw_from_quaternion(odom.pose.pose.orientation)

    return pose

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
        self._enable_control = False
        self._initialize()
        self._setup_pub_sub()

        # set use_sim_time parameter
        param = Parameter("use_sim_time", Parameter.Type.BOOL, True)
        self.set_parameters([param])

        # wait for clock received
        rclpy.spin_once(self, timeout_sec=1)

    def destroy(self) -> None:
        self._timer.destroy() # type: ignore
        self._command_pub.shutdown() # type: ignore
        self._odom_sub.shutdown() # type: ignore
        self._obstacles_sub.shutdown() # type: ignore
        self._group.destroy() # type: ignore
        super().destroy_node()

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

        # Obstacles
        self._use_obstacles_topic = self._obstacles is None
        self._obstacles_updated = False
        self._last_obstacles_msgs_raw = None

    def _setup_pub_sub(self) -> None:
        # Publishers
        self._command_pub = self.create_publisher(
            AckermannControlCommand, "/control/command/control_cmd", 1)

        # Subscribers
        self._odom_sub = self.create_subscription(
            Odometry, "/localization/kinematic_state", self._odom_callback, 1)
        self._obstacles_sub = self.create_subscription(
            Float64MultiArray, "/aichallenge/objects", self._obstacles_callback, 1)
        self._control_mode_request_sub = self.create_subscription(
            Bool, "control/control_mode_request_topic", self._control_mode_request_callback, 1)

    def _odom_callback(self, msg: Odometry) -> None:
        self._odom = msg

    def _obstacles_callback(self, msg: Float64MultiArray) -> None:
        if not self._use_obstacles_topic:
            return

        self._obstacles_updated = self._last_obstacles_msgs_raw == msg.data
        if self._obstacles_updated:
            self._last_obstacles_msgs_raw = msg.data
            for i in range(0, len(msg.data), 4):
                x = msg.data[i]
                y = msg.data[i + 1]
                self._obstacles.append(Obstacle(cx=x, cy=y, radius=self._cfg.obstacles.radius)) # type: ignore

    def _control_mode_request_callback(self, msg):
        if msg.data:
            self.get_logger().info("Control mode request received")
            self._enable_control = True

    def _wait_until_odom_received(self, timeout: float = 30.) -> None:
        t_start = self.get_clock().now()
        self.get_logger().info(f"t_start: {t_start}")
        rate = self.create_rate(30)
        while self._odom is None:
            now = self.get_clock().now()
            if (now - t_start).nanoseconds > timeout * 1e9:
                raise TimeoutError("Timeout while waiting for odometry message")
            rate.sleep()

    def _wait_until_control_mode_request_received(self, timeout: float = 240.) -> None:
        t_start = self.get_clock().now()
        rate = self.create_rate(30)
        while not self._enable_control:
            now = self.get_clock().now()
            if (now - t_start).nanoseconds > timeout * 1e9:
                raise TimeoutError("Timeout while waiting for control mode request message")
            rate.sleep()

    def run(self) -> None:
        self._wait_until_odom_received()
        self._wait_until_control_mode_request_received()
        control_rate = self.create_rate(self._mpc_cfg.control_rate)

        pose = odom_to_pose_2d(self._odom) # type: ignore
        self._car.update_states(pose.x, pose.y, pose.theta)
        self._car.update_reference_path(self._car.reference_path)

        sim_logger = SimulationLogger(
            self.get_logger(),
            self._car.temporal_state.x, self._car.temporal_state.y, True, False, True, 50)

        self.get_logger().info(f"START!")

        loop = 0
        lap_times = []
        next_lap_start = False
        MAX_LAPS=6

        t_start = self.get_clock().now()
        while rclpy.ok() and (not sim_logger.stop_requested()) and len(lap_times) < MAX_LAPS:
            loop += 1
            now = self.get_clock().now()
            t = (now - t_start).nanoseconds / 1e9

            if self._obstacles_updated:
                self._obstacles_updated = False
                self.get_logger().info("Obstacles updated")
                self._map.reset_map()
                self._map.add_obstacles(self._obstacles)


            pose = odom_to_pose_2d(self._odom) # type: ignore
            self._car.update_states(pose.x, pose.y, pose.theta)
            # print(f"car x: {self._car.temporal_state.x}, y: {self._car.temporal_state.y}, psi: {self._car.temporal_state.psi}")
            # print(f"mpc x: {self._mpc.model.temporal_state.x}, y: {self._mpc.model.temporal_state.y}, psi: {self._mpc.model.temporal_state.psi}")

            u: np.ndarray = self._mpc.get_control()
            # self.get_logger().info(f"u: {u}")

            if len(u) == 0:
                self.get_logger().error("No control signal")
                control_rate.sleep()

            self._car.drive(u)
            self._command_pub.publish(array_to_ackermann_control_command(now.to_msg(), u, self._cfg.mpc.a_max)) # type: ignore

            # Log states
            sim_logger.log(self._car, u, t)

            sim_logger.plot_animation(t, 0, lap_times, u, self._mpc, self._car)

            # Push next obstacle
            if loop % 50 == 0:
                # obstacle_manager.push_next_obstacle_random()

                # update reference path
                # 現在は、同じコースを何周もするだけ
                # (コースの途中で reference_path を更新してもOK)
                self._car.update_reference_path(self._car.reference_path)

            # Check if a lap has been completed
            if (next_lap_start and self._car.s >= self._car.reference_path.length or next_lap_start and self._car.s < self._car.reference_path.length / 20.0):
                if len(lap_times) > 0:
                    lap_time = t - sum(lap_times)
                else:
                    lap_time = t
                lap_times.append(lap_time)
                next_lap_start = False

                self.get_logger().info(f'Lap {len(lap_times)} completed! Lap time: {lap_time} s')

            # LAPインクリメント直後にゴール付近WPを最近傍WPとして認識してしまうと、 s>=lengthとなり
            # 次の周回がすぐに終了したと判定されてしまう場合があるため、
            # 誤判定防止のために少しだけ余計に走行した後に次の周回が開始したと判定する
            if not next_lap_start and (self._car.reference_path.length / 10.0 < self._car.s and self._car.s < self._car.reference_path.length / 10.0 * 2.0):
                next_lap_start = True
                self.get_logger().info(f'Next lap start!')

            control_rate.sleep()

        # show results
        sim_logger.show_results(lap_times, self._car)

    @classmethod
    def in_pkg_share(cls, file_path: str) -> str:
        return cls.PKG_PATH + file_path
