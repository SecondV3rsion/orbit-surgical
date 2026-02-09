# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a mops robot and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from pprint import pprint

##
# Pre-defined configs
##
from orbit.surgical.assets.iiwa14 import IIWA14_CFG, IIWA14_HIGH_PD_CFG  # isort: skip


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2"
    # Each group will have a robot in it
    origins = [[0.0, 0.0, 0.6], [0.0, 0.6, 0.6]]  # list of origins for each robot
    # Origin 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # Origin 2
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    # Articulation
    iiwa14_cfg = IIWA14_CFG.copy()
    iiwa14_cfg.prim_path = "/World/Origin1/Robot"
    iiwa14_1 = Articulation(cfg=iiwa14_cfg)

    iiwa14_high_pd_cfg = IIWA14_HIGH_PD_CFG.copy()
    iiwa14_high_pd_cfg.prim_path = "/World/Origin2/Robot"
    iiwa14_2 = Articulation(cfg=iiwa14_high_pd_cfg)

    # return the scene information
    scene_entities = {"iiwa14_1": iiwa14_1, "iiwa14_2": iiwa14_2}
    return scene_entities, origins

# Create dictionaries to hold joint position history

robot1_jp_history = {"target": [], "actual": []}
robot2_jp_history = {"target": [], "actual": []}

history = [
    {"target": [], "actual": []} for _ in range(2)
]


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.

    robots = [entities["iiwa14_1"], entities["iiwa14_2"]]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world

            for i, robot in enumerate(robots):
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[i]

                # write to sim
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                # set joint positions
                joint_pos_default, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos_default, joint_vel)

                # clear internal buffers
                robot.reset()
            print("[INFO]: Resetting robot state...")
        
        sin_wave = 0.1 * torch.sin(torch.tensor(count * sim_dt))

        for i, robot in enumerate(robots):
            joint_pos = robot.data.default_joint_pos.clone()
            # set target joint positions
            joint_pos[0, :] += sin_wave
            robot.set_joint_position_target(joint_pos)

            # store a copy of the joint positions
            history[i]["target"].append(joint_pos.clone())
            # -- write data to sim
            robot.write_data_to_sim()

        # Perform step
        sim.step()
        # Increment counter
        count += 1

        # read actual joint positions
        for i, robot in enumerate(robots):
            history[i]["actual"].append(robot.data.joint_pos.clone())
            
        # Update buffers
        for robot in robots:
            robot.update(sim_dt)

        # every 499 steps: plot the history
        if count % 499 == 0 and count > 0:
            plot_joint_positions(history[0], history[1])
            plot_joint_errors(history[0], history[1])


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
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([4.0, 0.3, 2.0], [0.0, 0.3, 1.0])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()