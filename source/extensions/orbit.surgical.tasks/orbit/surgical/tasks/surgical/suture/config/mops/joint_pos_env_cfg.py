# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

from orbit.surgical.tasks.surgical.suture import mdp
from orbit.surgical.tasks.surgical.suture.suture_env_cfg import SutureEnvCfg 

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from orbit.surgical.assets.mops import MOPS_CFG  # isort: skip


@configclass
class NeedleSutureEnvCfg(SutureEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set MOPS as robot
        self.scene.robot_1 = MOPS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot_1.init_state = MOPS_CFG.InitialStateCfg(
            pos=(0.0, 0.0, -0.2),  # initial position of the robot base
            rot=MOPS_CFG.init_state.rot,  # initial orientation of the robot bas
            joint_pos=MOPS_CFG.init_state.joint_pos,  # initial joint positions
        )

        # Set actions for the specific robot type (MOPS)
        self.actions.body_joint_pos = mdp.JointPositionActionCfg(
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
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.finger_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["tool_yaw1", "tool_yaw2"],
            open_command_expr={"tool_yaw1": 0.6, "tool_yaw2": 0.6},
            close_command_expr={"tool_yaw1": 0.08, "tool_yaw2": 0.08},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "tool_tcp0"

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.02, 0.02, 0.02)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/kuka_link_0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/tool_tcp0",
                    name="end_effector",
                ),
            ],
        )