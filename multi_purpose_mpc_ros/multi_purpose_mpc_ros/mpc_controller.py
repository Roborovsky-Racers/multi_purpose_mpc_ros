#!/usr/bin/env python3

import yaml
from typing import List, Tuple, Optional, NamedTuple
import dataclasses
from scipy import sparse
from scipy.sparse import dia_matrix
import numpy as np
import time
import copy

# ROS 2
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rclpy.parameter import Parameter
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

from std_msgs.msg import Bool, Float64MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Pose2D, Point

# autoware
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_planning_msgs.msg import Trajectory

# Multi_Purpose_MPC
from multi_purpose_mpc_ros.core.map import Map, Obstacle
from multi_purpose_mpc_ros.core.reference_path import ReferencePath
from multi_purpose_mpc_ros.core.spatial_bicycle_models import BicycleModel
from multi_purpose_mpc_ros.core.MPC import MPC
from multi_purpose_mpc_ros.core.utils import load_waypoints, kmh_to_m_per_sec, load_ref_path

# Project
from multi_purpose_mpc_ros.common import convert_to_namedtuple, file_exists
from multi_purpose_mpc_ros.simulation_logger import SimulationLogger
from multi_purpose_mpc_ros.obstacle_manager import ObstacleManager

def array_to_ackermann_control_command(stamp, u: np.ndarray, acc: float) -> AckermannControlCommand:
    msg = AckermannControlCommand()
    msg.stamp = stamp
    msg.lateral.stamp = stamp
    msg.lateral.steering_tire_angle = u[1]
    msg.lateral.steering_tire_rotation_rate = 2.0
    msg.longitudinal.stamp = stamp
    msg.longitudinal.speed = 0.0  #u[0]
    msg.longitudinal.acceleration = acc
    # msg.longitudinal.acceleration = -acc  # hack!!!!
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
    USE_BUG_ACC = True
    BUG_VEL = 40.0 # km/h
    BUG_ACC = 400.0

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
        self._mpc_pred_pub.shutdown() # type: ignore
        self._ref_path_pub.shutdown() # type: ignore
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

    def _create_reference_path_from_autoware_trajectory(self, trajectory: Trajectory) -> ReferencePath:
        wp_x = [0] * len(trajectory.points)
        wp_y = [0] * len(trajectory.points)
        for i, p in enumerate(trajectory.points):
            wp_x[i] = p.pose.position.x
            wp_y[i] = p.pose.position.y

        cfg_ref_path = self._cfg.reference_path # type: ignore
        reference_path = ReferencePath(
            self._map,
            wp_x,
            wp_y,
            cfg_ref_path.resolution,
            cfg_ref_path.smoothing_distance,
            cfg_ref_path.max_width,
            cfg_ref_path.circular)

        mpc_config = self._mpc_cfg
        speed_profile_constraints = {
            "a_min": mpc_config.a_min, "a_max": mpc_config.a_max,
            "v_min": 0.0, "v_max": mpc_config.v_max, "ay_max": mpc_config.ay_max}

        if not reference_path.compute_speed_profile(speed_profile_constraints):
            return None

        return reference_path

    def _initialize(self) -> None:
        def create_map() -> Map:
            return Map(self.in_pkg_share(self._cfg.map.yaml_path)) # type: ignore

        def create_ref_path(map: Map) -> ReferencePath:
            cfg_ref_path = self._cfg.reference_path # type: ignore

            is_ref_path_given = cfg_ref_path.csv_path != "" # type: ignore
            if is_ref_path_given:
                print("Using given reference path")
                wp_x, wp_y, _, _ = load_ref_path(self.in_pkg_share(self._cfg.reference_path.csv_path)) # type: ignore
                return ReferencePath(
                    map,
                    wp_x,
                    wp_y,
                    cfg_ref_path.resolution,
                    cfg_ref_path.smoothing_distance,
                    cfg_ref_path.max_width,
                    cfg_ref_path.circular)

            else:
                print("Using waypoints to create reference path")
                wp_x, wp_y = load_waypoints(self.in_pkg_share(self._cfg.waypoints.csv_path)) # type: ignore

                return ReferencePath(
                    map,
                    wp_x,
                    wp_y,
                    cfg_ref_path.resolution,
                    cfg_ref_path.smoothing_distance,
                    cfg_ref_path.max_width,
                    cfg_ref_path.circular)


        def create_obstacles() -> List[Obstacle]:
            use_csv_obstacles = self._cfg.obstacles.csv_path != "" # type: ignore
            if use_csv_obstacles:
                obstacles_file_path = self.in_pkg_share(self._cfg.obstacles.csv_path) # type: ignore
                obs_x, obs_y = load_waypoints(obstacles_file_path)
                obstacles = []
                for cx, cy in zip(obs_x, obs_y):
                    obstacles.append(Obstacle(cx=cx, cy=cy, radius=self._cfg.obstacles.radius)) # type: ignore
                self._obstacle_manager = ObstacleManager(self._map, obstacles)
                return obstacles
            else:
                return []

        def create_car(ref_path: ReferencePath) -> BicycleModel:
            cfg_model = self._cfg.bicycle_model # type: ignore
            return BicycleModel(
                ref_path,
                cfg_model.length,
                cfg_model.width,
                1.0 / self._cfg.mpc.control_rate) # type: ignore

        def create_mpc(car: BicycleModel) -> Tuple[MPCConfig, MPC]:
            cfg_mpc = self._cfg.mpc # type: ignore
            mpc_cfg = MPCConfig(
                cfg_mpc.N,
                sparse.diags(cfg_mpc.Q),
                sparse.diags(cfg_mpc.R),
                sparse.diags(cfg_mpc.QN),
                kmh_to_m_per_sec(self.BUG_VEL if self.USE_BUG_ACC else cfg_mpc.v_max),
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

        self._trajectory: Optional[Trajectory] = None

        # Obstacles
        self._use_obstacles_topic = self._obstacles == []
        self._obstacles_updated = False
        self._last_obstacles_msgs_raw = None

    def _setup_pub_sub(self) -> None:
        # Publishers
        self._command_pub = self.create_publisher(
            AckermannControlCommand, "/control/command/control_cmd", 1)
        self._mpc_pred_pub = self.create_publisher(
            MarkerArray, "/mpc/prediction", 1)
        latching_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._ref_path_pub = self.create_publisher(
            MarkerArray, "/mpc/ref_path", latching_qos)

        # Subscribers
        self._odom_sub = self.create_subscription(
            Odometry, "/localization/kinematic_state", self._odom_callback, 1)
        self._obstacles_sub = self.create_subscription(
            Float64MultiArray, "/aichallenge/objects", self._obstacles_callback, 1)
        self._control_mode_request_sub = self.create_subscription(
            Bool, "control/control_mode_request_topic", self._control_mode_request_callback, 1)
        self._trajectory_sub = self.create_subscription(
            Trajectory, "planning/scenario_planning/trajectory", self._trajectory_callback, 1)

    def _odom_callback(self, msg: Odometry) -> None:
        self._odom = msg

    def _obstacles_callback(self, msg: Float64MultiArray) -> None:
        if not self._use_obstacles_topic:
            return

        obstacles_updated = (self._last_obstacles_msgs_raw != msg.data) and (len(msg.data) > 0)
        if obstacles_updated:
            self._last_obstacles_msgs_raw = msg.data
            self._obstacles = []
            for i in range(0, len(msg.data), 4):
                x = msg.data[i]
                y = msg.data[i + 1]
                self._obstacles.append(Obstacle(cx=x, cy=y, radius=self._cfg.obstacles.radius)) # type: ignore
            # NOTE: This flag should be set to True only after the obstacles are updated
            self._obstacles_updated = True

    def _control_mode_request_callback(self, msg):
        if msg.data:
            self.get_logger().info("Control mode request received")
            self._enable_control = True

    def _trajectory_callback(self, msg):
        self._trajectory = msg

    def _wait_until_odom_received(self, timeout: float = 30.) -> None:
        t_start = self.get_clock().now()
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

    def _publish_mpc_pred_marker(self, x_pred, y_pred):
        pred_marker_array = MarkerArray()
        m_base = Marker()
        m_base.header.frame_id = "map"
        m_base.ns = "mpc_pred"
        m_base.type = Marker.SPHERE
        m_base.action = Marker.ADD
        m_base.pose.position.z = 0.0
        m_base.scale.x = 0.3
        m_base.scale.y = 0.3
        m_base.scale.z = 0.3
        m_base.color.a = 1.0
        m_base.color.r = 0.0
        m_base.color.g = 1.0
        m_base.color.b = 0.0
        for i in range(len(x_pred)):
            m = copy.deepcopy(m_base)
            m.id = i
            m.pose.position.x = x_pred[i]
            m.pose.position.y = y_pred[i]
            pred_marker_array.markers.append(m) # type: ignore
        self._mpc_pred_pub.publish(pred_marker_array)

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
        for i in range(len(ref_path.waypoints) - 1):
            m = copy.deepcopy(m_base)
            m.id = i
            start = Point()
            start.x = ref_path.waypoints[i].x
            start.y = ref_path.waypoints[i].y
            end = Point()
            end.x = ref_path.waypoints[i + 1].x
            end.y = ref_path.waypoints[i + 1].y
            m.points.append(start) # type: ignore
            m.points.append(end) # type: ignore
            ref_path_marker_array.markers.append(m) # type: ignore
        self._ref_path_pub.publish(ref_path_marker_array)

    def run(self) -> None:
        SHOW_PLOT_ANIMATION = False
        PLOT_RESULTS = False
        ANIMATION_INTERVAL = 20

        self._wait_until_odom_received()
        self._wait_until_control_mode_request_received()
        control_rate = self.create_rate(self._mpc_cfg.control_rate)

        pose = odom_to_pose_2d(self._odom) # type: ignore
        self._car.update_states(pose.x, pose.y, pose.theta)

        self._car.update_reference_path(self._car.reference_path)

        sim_logger = SimulationLogger(
            self.get_logger(),
            self._car.temporal_state.x, self._car.temporal_state.y, self._cfg.sim_logger.animation_enabled, SHOW_PLOT_ANIMATION, PLOT_RESULTS, ANIMATION_INTERVAL) # type: ignore

        self.get_logger().info(f"START!")

        loop = 0
        lap_times = []
        next_lap_start = False

        kp = 100.0
        last_u = np.array([0.0, 0.0])

        # for i in range(10):
        #     self._obstacle_manager.push_next_obstacle()

        self._publish_ref_path_marker(self._car.reference_path)

        t_start = self.get_clock().now()
        last_t = t_start
        while rclpy.ok() and (not sim_logger.stop_requested()):# and len(lap_times) < MAX_LAPS:
            self.get_logger().info("loop")
            # TODO:ここでbug accのpubをする or 別ノード
            control_rate.sleep()

            if self._cfg.reference_path.update_by_topic and self._trajectory is None: # type: ignore
                continue

            if loop % 100 == 0:
                # update obstacles
                if not self._use_obstacles_topic:
                    # self._obstacle_manager.push_next_obstacle()
                    self._obstacles = self._obstacle_manager.current_obstacles
                    self._obstacles_updated = True

                # update reference path
                if self._cfg.reference_path.update_by_topic: # type: ignore
                    new_referece_path = self._create_reference_path_from_autoware_trajectory(self._trajectory)
                    if new_referece_path is not None:
                        self._car.reference_path = new_referece_path
                        self._car.update_reference_path(self._car.reference_path)

                def plot_reference_path(car):
                    import matplotlib.pyplot as plt
                    import sys
                    fig, ax = plt.subplots(1, 1)
                    car.reference_path.show(ax)
                    plt.show()
                    sys.exit(1)
                # plot_reference_path(self._car)

            loop += 1

            now = self.get_clock().now()
            t = (now - t_start).nanoseconds / 1e9
            dt = (now - last_t).nanoseconds / 1e9
            last_t = now

            if self._obstacles_updated:
                self._obstacles_updated = False
                # self.get_logger().info("Obstacles updated")
                self._map.reset_map()
                self._map.add_obstacles(self._obstacles)


            pose = odom_to_pose_2d(self._odom) # type: ignore
            v = self._odom.twist.twist.linear.x

            self._car.update_states(pose.x, pose.y, pose.theta)
            # print(f"car x: {self._car.temporal_state.x}, y: {self._car.temporal_state.y}, psi: {self._car.temporal_state.psi}")
            # print(f"mpc x: {self._mpc.model.temporal_state.x}, y: {self._mpc.model.temporal_state.y}, psi: {self._mpc.model.temporal_state.psi}")

            u: np.ndarray = self._mpc.get_control()
            # self.get_logger().info(f"u: {u}")

            if len(u) == 0:
                self.get_logger().error("No control signal", throttle_duration_sec=1)
                continue

            acc = 0.
            bug_acc_enabled = False
            if self.USE_BUG_ACC:

                def deg2rad(deg):
                    return deg * np.pi / 180.0

                if abs(v) > kmh_to_m_per_sec(40.0):
                    bug_acc_enabled = False
                    acc = self._mpc_cfg.a_min
                elif abs(v) > kmh_to_m_per_sec(37.0) or abs(u[1]) > deg2rad(12.0):
                    bug_acc_enabled = False
                    acc = self._mpc_cfg.a_max
                else:
                    bug_acc_enabled = True
                    # acc = 400.0
                    acc = 430.0
            else:
                acc =  kp * (u[0] - v)
                # print(f"v: {v}, u[0]: {u[0]}, acc: {acc}")
                acc = np.clip(acc, self._mpc_cfg.a_min, self._mpc_cfg.a_max)
            # u[0] = np.clip(last_u[0] + acc * dt, 0.0, self._mpc_cfg.v_max)
            u[0] = v
            last_u[0] = u[0]
            last_u[1] = u[1]


            self._car.drive(u)
            if self.USE_BUG_ACC and bug_acc_enabled:
                sleep_duration = (1.0/self._mpc_cfg.control_rate)/160.
                cmd = array_to_ackermann_control_command(now.to_msg(), u, acc)
                # for _ in range(1 + (15 if abs(v) < kmh_to_m_per_sec(37.0) else 0) + (10 if abs(v) < kmh_to_m_per_sec(32.0) else 0)):
                for _ in range(10 + (20 if abs(v) < kmh_to_m_per_sec(39.0) else 0) + (20 if abs(v) < kmh_to_m_per_sec(32.0) else 0)):
                    now_stamp = self.get_clock().now().to_msg()
                    cmd.stamp = now_stamp
                    cmd.lateral.stamp = now_stamp
                    cmd.longitudinal.stamp = now_stamp
                    self._command_pub.publish(cmd) #ignore
                    time.sleep(sleep_duration)
            else:
                self._command_pub.publish(array_to_ackermann_control_command(now.to_msg(), u, acc)) #ignore

            # Log states
            sim_logger.log(self._car, u, t)
            sim_logger.plot_animation(t, loop, lap_times, u, self._mpc, self._car)


            # 約 0.25 秒ごとに予測結果を表示
            if loop % (self._mpc_cfg.control_rate // 4) == 0:
                self._publish_mpc_pred_marker(self._mpc.current_prediction[0], self._mpc.current_prediction[1]) # type: ignore

            # Check if a lap has been completed
            if (next_lap_start and self._car.s >= self._car.reference_path.length or next_lap_start and self._car.s < self._car.reference_path.length / 20.0):
                if len(lap_times) > 0:
                    lap_time = t - sum(lap_times)
                else:
                    lap_time = t
                lap_times.append(lap_time)
                next_lap_start = False

                # self.get_logger().info(f'Lap {len(lap_times)} completed! Lap time: {lap_time} s')

            # LAPインクリメント直後にゴール付近WPを最近傍WPとして認識してしまうと、 s>=lengthとなり
            # 次の周回がすぐに終了したと判定されてしまう場合があるため、
            # 誤判定防止のために少しだけ余計に走行した後に次の周回が開始したと判定する
            if not next_lap_start and (self._car.reference_path.length / 10.0 < self._car.s and self._car.s < self._car.reference_path.length / 10.0 * 2.0):
                next_lap_start = True
                # self.get_logger().info(f'Next lap start!')

        # show results
        sim_logger.show_results(lap_times, self._car)

    @classmethod
    def in_pkg_share(cls, file_path: str) -> str:
        return cls.PKG_PATH + file_path
