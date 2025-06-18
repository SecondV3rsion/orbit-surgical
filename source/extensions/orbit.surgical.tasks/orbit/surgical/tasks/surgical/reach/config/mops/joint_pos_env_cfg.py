from __future__ import annotations

import math

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

import orbit.surgical.tasks.surgical.reach.mdp as mdp
from orbit.surgical.tasks.surgical.reach.reach_env_cfg import ReachEnvCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass
import numpy as np

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from orbit.surgical.assets.mops import MOPS_CFG  # isort: skip


##
# Environment configuration
##

DEFAULT_ROT_TCP = [-np.pi, np.pi/2, 0] # roll, pitch, yaw

@configclass
class MOPSReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # simulation settings
        self.viewer.eye = (4, -0.6, 0.3)

        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Table/table.usd",
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, -0.457), rot=(0.7071068, 0, 0, 0.7071068)),
        )

        # switch robot to MOPS
        self.scene.robot = MOPS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["tool_tcp0"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["tool_tcp0"]
        #override terminations
        self.terminations.reached_goal.params["asset_cfg"].body_names = ["tool_tcp0"]
        # override observations
        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
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
            scale=1,
            use_default_offset=True,
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose = mdp.UniformPoseCommandCfg(
            asset_name="robot",
            body_name="tool_tcp0",
            resampling_time_range=(4.0, 4.0),
            debug_vis=True,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(0.45, 0.55),
                pos_y=(-0.1, 0.1),
                pos_z=(0.2, 0.3),
                roll=(DEFAULT_ROT_TCP[0] - np.pi/4, DEFAULT_ROT_TCP[0] + np.pi/4),
                pitch=(DEFAULT_ROT_TCP[1] - np.pi/4, DEFAULT_ROT_TCP[1] + np.pi/4),
                yaw=(DEFAULT_ROT_TCP[2] - np.pi/4, DEFAULT_ROT_TCP[2] + np.pi/4),
            ),
        )

        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (0.5, 1.5),
                "velocity_range": (0.0, 0.0),
            },
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
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


@configclass
class MOPSReachEnvCfg_PLAY(MOPSReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
