#!/bin/bash
source $(ros2 pkg prefix multi_purpose_mpc_ros)/.venv/bin/activate
cd $(ros2 pkg prefix --share multi_purpose_mpc_ros)/Multi-Purpose-MPC/src
python3 simulation.py