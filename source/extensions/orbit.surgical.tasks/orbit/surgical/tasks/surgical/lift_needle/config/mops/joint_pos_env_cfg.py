# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass

from orbit.surgical.tasks.surgical.lift_needle import mdp
from orbit.surgical.tasks.surgical.lift_needle.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from orbit.surgical.assets.mops import MOPS_CFG  # isort: skip


@configclass
class NeedleLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set MOPS as robot
        self.scene.robot = MOPS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (MOPS)
        self.actions.body_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "kuka_joint_1",
                "kuka_joint_2",
                "kuka_joint_3",
                "kuka_joint_4",
                "kuka_joint_5",
                "kuka_joint_6",
                "kuka_joint_7",
                "lnd_tool_roll_joint",
                "lnd_tool_pitch_joint",
                "lnd_tool_yaw_joint",
            ],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.finger_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["lnd_tool_gripper.*_joint"],
            open_command_expr={"lnd_tool_gripper1_joint": -0.5, "lnd_tool_gripper2_joint": 0.5},
            close_command_expr={"lnd_tool_gripper1_joint": -0.09, "lnd_tool_gripper2_joint": 0.09},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "lnd_tool_tip_link"

        # Set Suture Needle as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.4, 0.015), rot=(1, 0, 0, 0)),
            spawn=UsdFileCfg(
                usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Surgical_needle/needle_sdf.usd",
                scale=(0.4, 0.4, 0.4),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=8,
                    max_angular_velocity=200,
                    max_linear_velocity=200,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
            ),
        )

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
                    prim_path="{ENV_REGEX_NS}/Robot/lnd_tool_tip_link",
                    name="end_effector",
                ),
            ],
        )


@configclass
class NeedleLiftEnvCfg_PLAY(NeedleLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
