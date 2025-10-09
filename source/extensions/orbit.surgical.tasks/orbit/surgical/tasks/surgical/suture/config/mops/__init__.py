# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym
import os

from . import agents, ik_abs_env_cfg, ik_rel_env_cfg, joint_pos_env_cfg
from . import ik_rel_mimic_env_cfg, ik_rel_mimic_env

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Suture-MOPS-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.NeedleSutureEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftNeedlePPORunnerCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Suture-MOPS-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg.NeedleSutureEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftNeedlePPORunnerCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Suture-MOPS-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.NeedleSutureEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftNeedlePPORunnerCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

## Inverse Kinematics - Relative Pose Control with Mimic
##

#TO DO