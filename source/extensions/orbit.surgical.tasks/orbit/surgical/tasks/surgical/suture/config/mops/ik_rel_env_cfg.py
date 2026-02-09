# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from orbit.surgical.assets.mops import MOPS_HIGH_PD_CFG  # isort: skip


@configclass
class NeedleSutureEnvCfg(joint_pos_env_cfg.NeedleSutureEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set MOPS as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = MOPS_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state = MOPS_HIGH_PD_CFG.InitialStateCfg(
            pos=(0.0, -0.3, -0.2),  # initial position of the robot base
            rot=MOPS_HIGH_PD_CFG.init_state.rot,  # initial orientation of the robot bas
            joint_pos=MOPS_HIGH_PD_CFG.init_state.joint_pos,  # initial joint positions
        )
        # Set actions for the specific robot type (MOPS)
        self.actions.body_joint_pos_1 = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "kuka_A1",
                "kuka_A2",
                "kuka_A3",
                "kuka_A4",
                "kuka_A5",
                "kuka_A6",
                "kuka_A7",
                "tool_roll",
                "tool_pitch",
                "tool_yaw0",
            ],
            body_name="tool_tcp0",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
        )

        # Set MOPS as robot 2
        self.scene.robot_2 = MOPS_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot2")
        self.scene.robot_2.init_state = MOPS_HIGH_PD_CFG.InitialStateCfg(
            pos=(0.0, 0.3, -0.2),  # initial position of the robot base
            rot=MOPS_HIGH_PD_CFG.init_state.rot,  # initial orientation of the robot bas
            joint_pos=MOPS_HIGH_PD_CFG.init_state.joint_pos,  # initial joint positions
        )
        # Set actions for the specific robot type (MOPS)
        self.actions.body_joint_pos_2 = DifferentialInverseKinematicsActionCfg(
            asset_name="robot_2",
            joint_names=[
                "kuka_A1",
                "kuka_A2",
                "kuka_A3",
                "kuka_A4",
                "kuka_A5",
                "kuka_A6",
                "kuka_A7",
                "tool_roll",
                "tool_pitch",
                "tool_yaw0",
            ],
            body_name="tool_tcp0",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
        )

