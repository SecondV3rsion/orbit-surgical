import numpy as np
from dataclasses import MISSING

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

from . import mdp

##
# Scene definition
##

DEFAULT_ROT_TCP = [-np.pi, np.pi/2, 0] # roll, pitch, yaw

@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and an object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot, and end-effector frames.
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    finger_1_frame: FrameTransformerCfg = MISSING
    finger_2_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.7, 0.0, -0.457), rot=(0.7071068, 0, 0, 0.7071068)),
        spawn=UsdFileCfg(usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Table/table.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.95)),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.65, 0.75),
            pos_y=(-0.05, 0.05),
            pos_z=(0.3, 0.35),
            roll=(DEFAULT_ROT_TCP[0], DEFAULT_ROT_TCP[0]),
            pitch=(DEFAULT_ROT_TCP[1], DEFAULT_ROT_TCP[1]),
            yaw=(DEFAULT_ROT_TCP[2], DEFAULT_ROT_TCP[2]),
        ),
    )

    object_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    body_joint_pos: mdp.JointPositionActionCfg = MISSING
    finger_joint_pos: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands_pos, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        eef_pos_r = ObsTerm(func=mdp.ee_frame_pos_r)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_state, params={"finger1_name": "tool_yaw1",
                                                            "finger2_name": "tool_yaw2", 
                                                            "robot_cfg": SceneEntityCfg("robot"),
                                                            "open_value": 0.6,
                                                            "close_value": 0.08,
                                                            })


        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.65, 0.75), "y": (-0.03, 0.03), "z": (0.015, 0.015)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    #REACH----------------------------
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.05}, weight=0.2)

    orientation_error = RewTerm(
        func=mdp.orientation_command_error,
        params={"command_name": "object_pose", "asset_cfg": SceneEntityCfg("robot", body_names="tool_tcp0")},
        weight=-0.05,
    )
    #-----------------------------------

    #GRASP---------------------------

    # grasping_object = RewTerm(
    # func=mdp.object_grasping, 
    # params={"robot_cfg": SceneEntityCfg("robot"),
    #         "ee_frame_cfg": SceneEntityCfg("ee_frame"), 
    #         "object_cfg": SceneEntityCfg("object"),
    #         "diff_threshold": 0.02,
    #         "gripper_threshold": 0.1
    #         },  
    # weight=0.4)
    
    object_grasped = RewTerm(
        func=mdp.object_grasped, 
        params={"robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"), 
                "object_cfg": SceneEntityCfg("object"),
                "diff_threshold": 0.01,
                "gripper_threshold": 0.4
                },  
        weight=2.0)
    
    #-----------------------------------

    # LIFT ---------------------------

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.02}, weight=5.0)
    
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.2, "minimal_height": 0.02, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.02, "command_name": "object_pose"},
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )                

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # Increase penalty for action rate gradually

    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
    # )  

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.02, "asset_cfg": SceneEntityCfg("object")}
    ) 

    success = DoneTerm(
        func=mdp.object_reached_goal,
        params={"threshold": 0.02}
    )

##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=512, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 1.0 / 200.0
        self.viewer.eye = (1.6, 0.0, 0.3)
        self.viewer.lookat = (0.1, 0.0, 0.04)
