#!/bin/bash
source $(ros2 pkg prefix multi_purpose_mpc)/.venv/bin/activate
python3 $(ros2 pkg prefix multi_purpose_mpc)/lib/multi_purpose_mpc/simulation
