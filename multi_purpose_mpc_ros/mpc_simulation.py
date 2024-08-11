#!/usr/bin/env python3

import sys
from typing import List, Optional
import copy
import matplotlib.pyplot as plt

# ROS 2
import rclpy

# Project
from multi_purpose_mpc_ros.core.map import Map, Obstacle
from multi_purpose_mpc_ros.core.MPC import MPC
from multi_purpose_mpc_ros.core.spatial_bicycle_models import BicycleModel
from multi_purpose_mpc_ros.mpc_controller import MPCController
from multi_purpose_mpc_ros.simulation_logger import SimulationLogger
from multi_purpose_mpc_ros.obstacle_manager import ObstacleManager


class MPCSimulation:
    def __init__(self, controller: MPCController):
        self._controller = controller

    def run(self):
        SHOW_SIM_ANIMATION = True
        SHOW_PLOT_ANIMATION = True
        PLOT_RESULTS = True
        ANIMATION_INTERVAL = 20
        PRINT_INTERVAL = 10
        MAX_LAPS = 6

        mpc: MPC = self._controller._mpc
        map: Map = self._controller._map
        car: BicycleModel = mpc.model

        def plot_reference_path(car):
            fig, ax = plt.subplots(1, 1)
            car.reference_path.show(ax)
            plt.show()
            sys.exit(1)
        plot_reference_path(car)

        obstacles: List[Obstacle] = copy.deepcopy(self._controller._obstacles)
        obstacle_manager = ObstacleManager(map, obstacles)

        logger = self._controller.get_logger()
        sim_logger = SimulationLogger(
            logger,
            car.temporal_state.x, car.temporal_state.y, SHOW_SIM_ANIMATION, SHOW_PLOT_ANIMATION, PLOT_RESULTS, ANIMATION_INTERVAL)

        t = 0.0
        loop = 0
        lap_times = []
        next_lap_start = False

        while rclpy.ok() and (not sim_logger.stop_requested()) and len(lap_times) < MAX_LAPS:
            if PRINT_INTERVAL != 0 and loop % PRINT_INTERVAL == 0:
                logger.info(f"t = {t}, s = {car.s}, x = {car.temporal_state.x}, y = {car.temporal_state.y}")
            loop += 1

            u = mpc.get_control()

            # Get control signals
            u = mpc.get_control()

            # Simulate car
            car.drive(u)

            # Increment simulation time
            t += car.Ts

            # Log states
            sim_logger.log(car, u, t)

            # Plot animation
            sim_logger.plot_animation(t, loop, lap_times, u, mpc, car)
            import time

            # Push next obstacle
            if loop % 50 == 0:
                obstacle_manager.push_next_obstacle_random()

                # update reference path
                # 現在は、同じコースを何周もするだけ
                # (コースの途中で reference_path を更新してもOK)
                car.update_reference_path(car.reference_path)

            # Check if a lap has been completed
            if (next_lap_start and car.s >= car.reference_path.length or next_lap_start and car.s < car.reference_path.length / 20.0):
                if len(lap_times) > 0:
                    lap_time = t - sum(lap_times)
                else:
                    lap_time = t
                lap_times.append(lap_time)
                next_lap_start = False

                logger.info(f'Lap {len(lap_times)} completed! Lap time: {lap_time} s')

            # LAPインクリメント直後にゴール付近WPを最近傍WPとして認識してしまうと、 s>=lengthとなり
            # 次の周回がすぐに終了したと判定されてしまう場合があるため、
            # 誤判定防止のために少しだけ余計に走行した後に次の周回が開始したと判定する
            if not next_lap_start and (car.reference_path.length / 10.0 < car.s and car.s < car.reference_path.length / 10.0 * 2.0):
                next_lap_start = True
                logger.info(f'Next lap start!')

        # show results
        sim_logger.show_results(lap_times, car)
