# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.assets import Articulation
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def generated_commands_pos(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    command = env.command_manager.get_command(command_name)
    return command[:, :3]

def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat

def ee_frame_pos_r(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_source[:, 0, :]

def ee_frame_quat_r(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_source[:, 0, :] 

    return ee_frame_quat

def gripper_pos(
    env: ManagerBasedRLEnv,
    finger1_name: str = "tool_yaw1",
    finger2_name: str = "tool_yaw2", 
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    
    robot: Articulation = env.scene[robot_cfg.name]

    finger1_idx = robot.joint_names.index(finger1_name)
    finger2_idx = robot.joint_names.index(finger2_name)

    finger_joint_1 = robot.data.joint_pos[:, finger1_idx].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot.data.joint_pos[:, finger2_idx].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)

def gripper_state(
    env: ManagerBasedRLEnv,
    finger1_name: str = "tool_yaw1",
    finger2_name: str = "tool_yaw2",
    open_value: float = 0.6,
    close_value: float = 0.08,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    finger1_idx = robot.joint_names.index(finger1_name)
    finger2_idx = robot.joint_names.index(finger2_name)

    # Average both fingers
    finger_avg = (robot.data.joint_pos[:, finger1_idx] + 
                  robot.data.joint_pos[:, finger2_idx]) / 2.0  # (N,)

    # Midpoint between open (0.6) and closed (0.08)
    threshold = (open_value + close_value) / 2.0  # 0.34

    result = torch.where(finger_avg > threshold,
                         torch.ones_like(finger_avg),   # open  → 1.0
                         torch.zeros_like(finger_avg))  # closed → 0.0

    return result.unsqueeze(1)

def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b
