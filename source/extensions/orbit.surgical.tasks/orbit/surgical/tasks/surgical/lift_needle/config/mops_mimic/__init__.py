# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym
import os

from . import agents, ik_rel_env_cfg, ik_abs_env_cfg
from . import ik_rel_mimic_env_cfg, ik_abs_mimic_env_cfg

##
# Register Gym environments.
##


##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Lift-Needle-MOPS-BC-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg.NeedleLiftEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Needle-MOPS-Mimic-IK-Abs-v0",
    entry_point="orbit.surgical.tasks.surgical.lift_needle.config.mops_mimic.ik_abs_mimic_env:NeedleLiftMimicEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_mimic_env_cfg.NeedleLiftMimicEnvCfg,
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Lift-Needle-MOPS-BC-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.NeedleLiftEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_validation.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Needle-MOPS-Mimic-IK-Rel-v0",
    entry_point="orbit.surgical.tasks.surgical.lift_needle.config.mops_mimic.ik_rel_mimic_env:NeedleLiftMimicEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_mimic_env_cfg.NeedleLiftMimicEnvCfg,
    },
    disable_env_checker=True,
)
