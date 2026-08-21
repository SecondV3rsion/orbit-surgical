# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym
import os

from . import ik_rel_env_cfg

##
# Register Gym environments.
##


##
# Inverse Kinematics - Absolute Pose Control
##



##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Handover-Needle-Dual-MOPS-BC-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.NeedleHandoverEnvCfg,
        #"robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)
