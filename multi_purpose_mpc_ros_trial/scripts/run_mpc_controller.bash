#!/bin/bash
source $(ros2 pkg prefix multi_purpose_mpc_ros_trial)/.venv/bin/activate
python3 $(ros2 pkg prefix multi_purpose_mpc_ros_trial)/lib/multi_purpose_mpc_ros_trial/mpc_controller $@
