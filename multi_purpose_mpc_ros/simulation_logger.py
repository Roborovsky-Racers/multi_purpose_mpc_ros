from rclpy.impl.rcutils_logger import RcutilsLogger
import numpy as np
import matplotlib.pyplot as plt
from multi_purpose_mpc_ros.core.MPC import MPC
from multi_purpose_mpc_ros.core.utils import format_time, m_per_sec_to_kmh

class SimulationLogger:
    def __init__(
            self,
            logger: RcutilsLogger,
            init_x: float,
            init_y: float,
            show_sim_animation: bool,
            show_plot_animation: bool,
            plot_results: bool,
            animation_interval: bool):

        self._logger = logger
        self._show_sim_animation = show_sim_animation
        self._show_plot_animation = show_plot_animation
        self._plot_results = plot_results
        self._animation_interval = animation_interval

        self._stop_requested = False

        self.axes = None

        num_cols = 0
        if self._show_sim_animation:
            num_cols += 1
        if self._show_plot_animation:
            num_cols += 2

        if num_cols > 0:
            self.fig, self.axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))
            if num_cols == 1:
                self.axes = [self.axes]

        if self._show_sim_animation or self._show_plot_animation:
            self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Logging containers
        self.x_log = [init_x]
        self.y_log = [init_y]
        self.v_log = [0.0]
        self.t_log = [0.0]
        self.delta_log = [0.0]

    def stop_requested(self):
        return self._stop_requested

    def on_key(self, event):
        if event.key == 'q':
            self._stop_requested = True

    def log(self, car, u, t):
        # Log car state
        self.x_log.append(car.temporal_state.x)
        self.y_log.append(car.temporal_state.y)
        self.v_log.append(m_per_sec_to_kmh(u[0]))
        self.delta_log.append(np.degrees(u[1]))
        self.t_log.append(t)

    def plot_animation(self, t, loop, lap_times, u, mpc: MPC, car):
        idx = 0

        if loop % self._animation_interval == 0:
            if self._show_sim_animation:
                # Plot path and drivable area
                car.reference_path.show(self.axes[idx])

                # Plot car
                car.show(self.axes[idx])

                # Plot MPC prediction
                mpc.show_prediction(self.axes[idx])

                # Plot passed path
                self.axes[idx].plot(self.x_log[0], self.y_log[0], 'b*')
                self.axes[idx].plot(self.x_log, self.y_log)

                lap_time = lap_times[-1] if len(lap_times) > 0 else 0

                # Set figure title
                self.axes[idx].set_title(f'MPC Simulation: v(t): {m_per_sec_to_kmh(u[0]):.2f} km/s,\ndelta(t): {np.degrees(u[1]):.2f} deg, Duration: {format_time(t)},\nLap: {len(lap_times)}, Lap time {format_time(lap_time)}', fontsize=10)
                self.axes[idx].axis('off')
                idx += 1

            if self._show_plot_animation:
                self.axes[idx].cla()
                self.axes[idx].plot(self.t_log, self.v_log)
                self.axes[idx].set_xlabel('Time [s]')
                self.axes[idx].set_ylabel('Speed [km/h]')
                idx += 1

                self.axes[idx].cla()
                self.axes[idx].plot(self.t_log, self.delta_log)
                self.axes[idx].set_xlabel('Time [s]')
                self.axes[idx].set_ylabel('Steering Angle [deg]')
                idx += 1

            if idx > 0:
                plt.tight_layout()
                plt.pause(0.001)


    def show_results(self, lap_times, car):
        total_time = sum(lap_times)
        ave_lap_time = total_time / len(lap_times) if len(lap_times) > 0 else 0
        fastest_lap_time = min(lap_times) if len(lap_times) > 0 else 0

        self._logger.info("#########################################")
        self._logger.info("Simulation finished!")
        self._logger.info(f"       Total laps: {len(lap_times)}")
        self._logger.info(f"       Total time: {format_time(total_time)} s")
        self._logger.info(f" Average Lap time: {format_time(ave_lap_time)} s")
        self._logger.info(f" Fastest Lap time: {format_time(fastest_lap_time)} s")
        self._logger.info("-----------------------------------------")
        for i, lap_time in enumerate(lap_times):
            self._logger.info(f"       Lap {i+1} time: {format_time(lap_time)} s")
        self._logger.info("#########################################")

        if self._plot_results:
            if self.stop_requested or not self._show_sim_animation or not self._show_plot_animation:
                plt.close()
                self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5.5))
            for ax in self.axes:
                ax.cla()

            idx = 0
            car.reference_path.show(self.axes[idx])
            car.show(self.axes[idx])
            self.axes[idx].plot(self.x_log[0], self.y_log[0], 'b*')
            self.axes[idx].plot(self.x_log, self.y_log)
            self.axes[idx].set_title(f'Simulation result: Lap: {len(lap_times)}, Total time: {format_time(sum(lap_times))}', fontsize=10)

            self.axes[idx].axis('off')
            idx += 1

            # plot results
            self.axes[idx].plot(self.t_log, self.v_log)
            self.axes[idx].set_xlabel('Time [s]')
            self.axes[idx].set_ylabel('Speed [km/h]')
            idx += 1

            self.axes[idx].plot(self.t_log, self.delta_log)
            self.axes[idx].set_xlabel('Time [s]')
            self.axes[idx].set_ylabel('Steering Angle [deg]')
            idx += 1

            plt.tight_layout()
            plt.show()