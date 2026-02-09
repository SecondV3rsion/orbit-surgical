# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="mops_high_pd", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
import numpy as np

##
# Pre-defined configs
##
from orbit.surgical.assets.mops import MOPS_CFG, MOPS_HIGH_PD_CFG  # isort:skip
from orbit.surgical.assets.iiwa14 import IIWA14_CFG, IIWA14_HIGH_PD_CFG  # isort:skip
from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

history = {"target": [], "actual": []}

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.7, 0.0, -0.457), rot=(0.7071068, 0, 0, 0.7071068)),
        spawn=UsdFileCfg(usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Table/table.usd"),
    )

    object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.0), rot=(0.7071068, 0, 0, 0.7071068)),
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

    # articulation
    if args_cli.robot == "mops":
        robot = MOPS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "mops_high_pd":
        robot = MOPS_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "iiwa14":
        robot = IIWA14_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "iiwa14_high_pd":
        robot = IIWA14_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: mops, mops_high_pd, iiwa14, iiwa14_high_pd")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    object = scene["object"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm x y z and orientation as quaternion w x y z
    object_goal = torch.tensor([0.6, 0, 0.3, 0.0, 0.707, 0.0, -0.707], device=sim.device)
    object_pose_w = object.data.default_root_state.clone()[:,0:7]
    # object_pose_w[:,2] += 0.002
    object_pose_w[:,3:7] = object_goal[3:7]

    root_state = object.data.root_state_w.clone()

    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = object_goal

    # Specify robot-specific parameters
    if args_cli.robot == "mops" or args_cli.robot == "mops_high_pd":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["kuka_A(1|2|3|4|5|6|7)", "tool_roll", "tool_pitch", "tool_yaw1", "tool_yaw2"], body_names=["tool_tcp0"])
    elif args_cli.robot == "iiwa14" or args_cli.robot == "iiwa14_high_pd":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["kuka_A(1|2|3|4|5|6|7)"], body_names=["kuka_link_7"])
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: mops, mops_high_pd, iiwa14, iiwa14_high_pd")

    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    state = 0
    # Simulation loop
    count = 0
    while simulation_app.is_running():
        # reset
        if state == 0:
            # reset time
            count = 0
            # reset object
            # write root state to simulation
            object.write_root_pose_to_sim(root_state[:, :7])
            object.write_root_velocity_to_sim(root_state[:, 7:])
            object.reset()
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset actions
            ik_commands[:] = object_pose_w
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            # switch state
            state = 1
        elif state == 1:
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            ee_object_diff = ee_pos_b - ik_commands[:, 0:3]

            if torch.norm(ee_object_diff) < 0.01:
                state = 2
        elif state == 2:
            # hold position
            joint_pos_des = robot.data.joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # close grasper
            joint_pos_des[:, -2:] = 0.08

            # reset actions
            ik_commands[:] = object_goal
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            
            if robot.data.joint_pos[0, -2] < 0.1:
                state = 3
        elif state == 3:
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            ee_goal_diff = ee_pos_b - object_goal[0:3]
            if torch.norm(ee_goal_diff) < 0.03:
                state = 0

        history["target"].append(joint_pos_des.clone())
        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        history["actual"].append(robot.data.joint_pos.clone())
        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

        count += 1
        # if count % 170 == 0:
        #     #plot_joint_errors(history)
        #     plot_joint_positions(history)

def plot_joint_positions(joint_positions1, joint_positions2=None, joint_positions3=None):
    import matplotlib.pyplot as plt
    import torch

    def plot_one_robot(joint_positions, robot_name="Robot"):
        target = torch.stack(joint_positions["target"]).detach().cpu()   # (T, 1, n_joints)
        actual = torch.stack(joint_positions["actual"]).detach().cpu()
        
        T = target.shape[0]
        num_joints = target.shape[2]

        timesteps = range(T)

        plt.figure(figsize=(8, num_joints * 2))

        for j in range(num_joints):
            plt.subplot(num_joints, 1, j + 1)
            plt.plot(timesteps, target[:, 0, j], label=f"Target joint {j}", color="red")
            plt.plot(timesteps, actual[:, 0, j], label=f"Actual joint {j}", color="black")
            plt.legend(loc='upper right')
            plt.grid(True)

        plt.suptitle(f"{robot_name}: Joint Positions Over Time", fontsize=14)
        plt.xlabel("Time Step")
        plt.ylabel("Joint Position (radians)")
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for title

    # ---- Plot each robot ----
    plot_one_robot(joint_positions1, "Robot 1")
    if joint_positions2 is not None:
        plot_one_robot(joint_positions2, "Robot 2")
    if joint_positions3 is not None:
        plot_one_robot(joint_positions3, "Robot 3")

    plt.show()

def plot_joint_errors(joint_positions1, joint_positions2=None, joint_positions3=None):
    import matplotlib.pyplot as plt
    import torch

    robots = [joint_positions1, joint_positions2, joint_positions3]
    robots = [r for r in robots if r is not None]  # keep only valid

    if not robots:
        print("No robot data provided")
        return

    # Assume all robots have the same number of joints and timesteps
    num_joints = robots[0]["target"][0].shape[1]
    T = len(robots[0]["target"])
    timesteps = range(T)

    plt.figure(figsize=(10, num_joints * 2))

    # For each joint, create a subplot
    for j in range(num_joints):
        plt.subplot(num_joints, 1, j + 1)
        for i, joint_positions in enumerate(robots):
            target = torch.stack(joint_positions["target"]).detach().cpu()   # (T,1,N)
            actual = torch.stack(joint_positions["actual"]).detach().cpu()   # (T,1,N)
            error = target[:, 0, j] - actual[:, 0, j]                        # (T,)
            plt.plot(timesteps, error, label=f"Robot {i+1}")
            plt.grid(True)
            plt.legend(loc='upper right')

    plt.suptitle(f"Joint Errors", fontsize=12)
    plt.xlabel("Time Step")
    plt.ylabel("Error (rad)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for title
    plt.show()         

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
