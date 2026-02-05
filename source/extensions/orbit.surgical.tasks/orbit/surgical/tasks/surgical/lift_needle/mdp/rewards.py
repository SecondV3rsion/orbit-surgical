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
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_grasping(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.01,
    gripper_open_val: torch.tensor = torch.tensor([0.6]),
    gripper_threshold: float = 0.1,
    finger1_name: str = "tool_yaw1",
    finger2_name: str = "tool_yaw2",
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    finger1_idx = robot.joint_names.index(finger1_name)
    finger2_idx = robot.joint_names.index(finger2_name)
    joint_pos = robot.data.joint_pos

    grasped = torch.logical_and(
        pose_diff < diff_threshold,
        torch.abs(joint_pos[:, finger1_idx] - gripper_open_val.to(env.device)) > gripper_threshold,
    )
    grasped = torch.logical_and(
        grasped, torch.abs(joint_pos[:, finger2_idx] - gripper_open_val.to(env.device)) > gripper_threshold
    )

    return grasped

def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.01,
    gripper_open_val: torch.tensor = torch.tensor([0.6]),
    gripper_threshold: float = 0.1,
    finger1_name: str = "tool_yaw1",
    finger2_name: str = "tool_yaw2",
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    finger1_idx = robot.joint_names.index(finger1_name)
    finger2_idx = robot.joint_names.index(finger2_name)
    joint_pos = robot.data.joint_pos

    grasped = torch.logical_and(
        joint_pos[:, finger1_idx] < gripper_threshold,
        joint_pos[:, finger2_idx] < gripper_threshold,
    )

    grasped = torch.logical_and(
        pose_diff < diff_threshold,
        grasped,
    )

    # in_ee = between_fingers(
    #     env,
    #     ee_frame_cfg=ee_frame_cfg,
    #     finger1_frame_cfg=SceneEntityCfg("finger_1_frame"),
    #     finger2_frame_cfg=SceneEntityCfg("finger_2_frame"),
    #     object_cfg=object_cfg,
    #     std=diff_threshold,
    #     z_offset=0.002,
    #     x_offset=0.02,
    # )

    #grasped = grasped * in_ee

    #grasped = torch.logical_and(in_ee, grasped)

    return grasped

def gripper_state_by_distance(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    std_far: float = 0.02,     # ~2 cm
    std_near: float = 0.008,   # ~8 mm
    gripper_threshold: float = 0.1,
    finger1_name: str = "tool_yaw1",
    finger2_name: str = "tool_yaw2",
) -> torch.Tensor:

    # --- scene ---
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # --- distance ---
    obj_pos = object.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    d = torch.norm(obj_pos - ee_pos, dim=1)

    # --- gripper state ---
    f1 = robot.joint_names.index(finger1_name)
    f2 = robot.joint_names.index(finger2_name)
    joint_pos = robot.data.joint_pos

    gripper_open = (
        (joint_pos[:, f1] > gripper_threshold) +
        (joint_pos[:, f2] > gripper_threshold)
    ) / 2.0

    #gripper_closed = 1.0 - gripper_open

    # --- FAR: reward open ---
    far_term = gripper_open * torch.tanh(d / std_far)

    # --- NEAR: reward closed ---
    #near_term = gripper_closed * (1 - torch.tanh(d / std_near))

    return far_term # + near_term


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

def object_is_lifted_tanh(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object higher using tanh-kernel."""
    object: RigidObject = env.scene[object_cfg.name]

    return torch.tanh(object.data.root_pos_w[:, 2]/minimal_height)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_open_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    finger1_name: str = "tool_yaw1",
    finger2_name: str = "tool_yaw2",
    gripper_threshold: float = 0.2,
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    
    robot: Articulation = env.scene[robot_cfg.name]

    finger1_idx = robot.joint_names.index(finger1_name)
    finger2_idx = robot.joint_names.index(finger2_name)
    joint_pos = robot.data.joint_pos
    open_grasper = torch.logical_and(
        joint_pos[:, finger1_idx] > gripper_threshold, joint_pos[:, finger2_idx] > gripper_threshold
    )

    return torch.where(open_grasper, 1 - torch.tanh(object_ee_distance / std), 0.0)

def object_open_z_distance(
    env: ManagerBasedRLEnv,
    std: float,
    z_offset: float = 0.02,
    gripper_threshold: float = 0.2,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    finger1_name: str = "tool_yaw1",
    finger2_name: str = "tool_yaw2",
) -> torch.Tensor:
    """
    Reward XY reaching:
    - above object + z_offset → only if gripper is open
    - below object + z_offset → regardless of gripper state
    """

    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    reach_reward = 1.0 - torch.tanh(object_ee_distance / std)

    # Height condition
    above_obj = ee_w[:, 2] > (cube_pos_w[:, 2] + z_offset)

    # Gripper state
    finger1_idx = robot.joint_names.index(finger1_name)
    finger2_idx = robot.joint_names.index(finger2_name)
    joint_pos = robot.data.joint_pos

    gripper_open = torch.logical_and(
        joint_pos[:, finger1_idx] > gripper_threshold,
        joint_pos[:, finger2_idx] > gripper_threshold,
    )

    # Mask logic
    valid = torch.logical_or(
        ~above_obj,          # below z threshold → always valid
        gripper_open         # above → only if open
    )

    return torch.where(valid, reach_reward, torch.zeros_like(reach_reward))

def object_ee_xy_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos_w = object.data.root_pos_w[..., :2]  # (num_envs, 2)
    ee_w = ee_frame.data.target_pos_w[..., 0, :2]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_xy_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_xy_distance / std)

def ee_height_above_object(
        env: ManagerBasedRLEnv,
        target_height: float=0.0025,
        std: float=0.002,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:

    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_z = object.data.root_pos_w[:, 2]
    ee_z = ee_frame.data.target_pos_w[..., 0, 2]
    dz = ee_z - object_z - target_height
    return 1 - torch.tanh(torch.abs(dz) / std)



def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))

def between_fingers(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    finger1_frame_cfg: SceneEntityCfg,
    finger2_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    std: float = 0.01,
    z_offset: float = 0.02,
    x_offset: float = 0.02,
) -> torch.Tensor:
    """Reward = 1 if object is between fingers in X and Y, and close in Z to EE."""

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    finger1_frame: FrameTransformer = env.scene[finger1_frame_cfg.name]
    finger2_frame: FrameTransformer = env.scene[finger2_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # Positions: shape (num_envs, 3)
    object_pos = obj.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    f1__y_pos = finger1_frame.data.target_pos_w[:, 0, 1] # y position
    f2_y_pos = finger2_frame.data.target_pos_w[:, 0, 1] # y position

    # ------ x BETWEEN FINGERS ------
    cond_x = (obj.data.root_pos_w[:, 0] + x_offset > ee_pos[:, 0]) & (obj.data.root_pos_w[:, 0] - x_offset < ee_pos[:, 0])
    # ------ Y BETWEEN FINGERS ------
    cond_y = (obj.data.root_pos_w[:, 1] > f1__y_pos) & (obj.data.root_pos_w[:, 1] < f2_y_pos)
    # ------ Z CLOSE TO EE ------
    z_distance = torch.abs(ee_pos[:, 2]- obj.data.root_pos_w[:, 2])
    cond_z =  z_distance < z_offset
    
    # ------ DISTANCE ------
    # Distance of the end-effector to the object: (num_envs,)
    pose_diff = torch.linalg.vector_norm(object_pos - ee_pos, dim=1)
    # Combine all three conditions
    grasp_success = (cond_x & cond_y & cond_z) * (1 - torch.tanh(pose_diff / std))

    return grasp_success