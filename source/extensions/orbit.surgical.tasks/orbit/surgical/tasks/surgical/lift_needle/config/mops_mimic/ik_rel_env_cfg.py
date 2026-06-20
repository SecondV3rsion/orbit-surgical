# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

from orbit.surgical.tasks.surgical.lift_needle import mdp
from orbit.surgical.tasks.surgical.lift_needle.lift_env_mimic_cfg import LiftEnvCfg

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from orbit.surgical.assets.mops import MOPS_CFG  # isort: skip


@configclass
class NeedleLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set MOPS as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = MOPS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state = MOPS_CFG.InitialStateCfg(
            pos=(0.0, 0.0, -0.208),  # initial position of the robot base
            rot=MOPS_CFG.init_state.rot,  # initial orientation of the robot bas
            joint_pos=MOPS_CFG.init_state.joint_pos,  # initial joint positions
        )
        # Set actions for the specific robot type (MOPS)
        self.actions.body_joint_pos = DifferentialInverseKinematicsActionCfg(
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
            ],
            body_name="tool_tcp0",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=1.0,
            clip={"kuka_A(1|2|3|4|5|6)": (-0.01, 0.01)},
        )
        self.actions.finger_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["tool_yaw1", "tool_yaw2"],
            open_command_expr={"tool_yaw1": 0.6, "tool_yaw2": 0.6},
            close_command_expr={"tool_yaw1": 0.08, "tool_yaw2": 0.08},
        )

        # Set the body name for the end effector
        self.commands.object_pose.body_name = "tool_tcp0"

        # Set Suture Needle as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(0.7071068, 0, 0, 0.7071068)),
            spawn=UsdFileCfg(
                usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Surgical_needle/needle.usd",
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
                    prim_path="{ENV_REGEX_NS}/Robot/tool_tcp0",
                    name="end_effector",
                ),
            ],
        )

        self.scene.finger_1_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/kuka_link_0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/tool_tcp1",
                    name="finger_1",
                ),
            ],
        )

        self.scene.finger_2_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/kuka_link_0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/tool_tcp2",
                    name="finger_2",
                ),
            ],
        )