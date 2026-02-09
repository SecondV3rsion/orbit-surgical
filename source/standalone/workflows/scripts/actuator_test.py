# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import orbit.surgical.tasks  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import matplotlib.pyplot as plt

history_obs1 = []
history_obs2 = []
history_obs3 = []
history_actions1 = []
history_actions2 = []
history_actions3 = []
sin_history = []

robot_base_pos1=torch.tensor([0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0],device='cuda')
robot_base_pos2=torch.tensor([0.0, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0],device='cuda')
robot_base_pos3=torch.tensor([0.0, 1.2, 0.6, 0.0, 0.0, 0.0, 0.0],device='cuda')

def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    step = 1
    obs, _ = env.get_observations()
    first_obs = obs.clone()
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            if args_cli.task == "Isaac-Actuator-test-MOPS-v0":
                history_obs1.append(obs[0, :12].clone())
                history_obs2.append(obs[0, 12:24].clone())
                history_obs3.append(obs[0, 24:36].clone())

                #print(f"DEBUG: Aplying action JOINT")
                actions = first_obs + 0.04 * torch.sin(torch.tensor(step / 10.0)) # joint position

                history_actions1.append(actions[0, :12].clone())
                history_actions2.append(actions[0, 12:24].clone())
                history_actions3.append(actions[0, 24:36].clone())

            elif args_cli.task == "Isaac-Actuator-test-MOPS-IK-Abs-v0":
                history_obs1.append(obs[0, :7].clone())
                history_obs2.append(obs[0, 7:14].clone())
                history_obs3.append(obs[0, 14:21].clone())

                actions = first_obs
                actions[0,0:3] = torch.tensor([0.6,0.0,0.4]) #+ 0.1 * torch.sin(torch.tensor(step / 10.0))
                actions[0,7:10] = torch.tensor([0.6,0.0,0.4]) #+ 0.1 * torch.sin(torch.tensor(step / 10.0))
                actions[0,14:17] = torch.tensor([0.6,0.0,0.4]) #+ 0.1 * torch.sin(torch.tensor(step / 10.0))

                # z axis sinus wave
                actions[0,2] = 0.2 * torch.sin(torch.tensor(step / 10.0))
                actions[0,9] = + 0.2 * torch.sin(torch.tensor(step / 10.0))
                actions[0,16] = + 0.2 * torch.sin(torch.tensor(step / 10.0))



                #print(f"DEBUG: Aplying action ABS IK")
                
                history_actions1.append(actions[0, :7].clone() + robot_base_pos1)
                history_actions2.append(actions[0, 7:14].clone() + robot_base_pos2)
                history_actions3.append(actions[0, 14:21].clone() + robot_base_pos3)



            elif args_cli.task == "Isaac-Actuator-test-MOPS-IK-Rel-v0":
                history_obs1.append(obs[0, :7].clone())
                history_obs2.append(obs[0, 7:14].clone())
                history_obs3.append(obs[0, 14:21].clone())

                # calculate difference between current ee_pos and initial ee_pos
                ee_pos1 = obs[0, :3].clone()
                ee_pos2 = obs[0, 7:10].clone()
                ee_pos3 = obs[0, 14:17].clone()
                delta_ee1 = ee_pos1 - first_obs[0, :3] 
                delta_ee2 = ee_pos2 - first_obs[0, 7:10] 
                delta_ee3 = ee_pos3 - first_obs[0, 14:17]

                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

                # Relative actions for each robot
                actions[0,0:3]  = delta_ee1
                actions[0,7:10]  = delta_ee2
                actions[0,12:15] = delta_ee3
                
                if step < 50:
                    actions[0,2]  = 0.01
                    actions[0,9]  = 0.01
                    actions[0,16] = 0.01
                elif step < 100:
                    actions[0,2]  = -0.01
                    actions[0,9]  = -0.01
                    actions[0,16] = -0.01
                else:
                    step = 0  # reset step counter



                history_actions1.append(actions[0, :7].clone())
                history_actions2.append(actions[0, 7:14].clone())
                history_actions3.append(actions[0, 14:21].clone())
        
            
            # apply actions
            obs, _, _, _ = env.step(actions)

            if step % 200 == 0:
                history_actions = [history_actions1, history_actions2, history_actions3]
                history_obs = [history_obs1, history_obs2, history_obs3]
                #plot_actions_obs(history_actions, history_obs)
                # plt.plot(sin_history)
                # plt.title("Sin Wave Action Over Time")
                # plt.xlabel("Time Step")
                # plt.ylabel("Action Value")
                # plt.grid(True)
                # plt.tight_layout()
                # plt.show()
            step += 1

    # close the simulator
    env.close()

def plot_actions_obs(data1, data2):
    """
    Create 3 subplots (one per robot):
    - Actions plotted with black lines
    - Observations plotted with red dashed lines
    """

    # Convert lists of tensors ➝ stacked CPU tensors (T, N)
    acts = [torch.stack(data1[i]).detach().cpu()[:, :7] for i in range(3)]
    obs  = [torch.stack(data2[i]).detach().cpu()[:, :7] for i in range(3)]

    timesteps = range(acts[0].shape[0])
    num_joints = acts[0].shape[1]

    fig, axs = plt.subplots(3, 1, figsize=(8, 14), sharex=True)

    robot_names = ["Robot 1", "Robot 2", "Robot 3"]

    for idx in range(3):
        for j in range(num_joints):
            axs[idx].plot(
                timesteps, acts[idx][:, j],
                color="black", linewidth=1,
                label="Actions" if j == 0 else ""
            )
            axs[idx].plot(
                timesteps, obs[idx][:, j],
                color="red", linestyle="--", linewidth=1,
                label="Obs" if j == 0 else ""
            )

        axs[idx].set_title(f"{robot_names[idx]}")
        axs[idx].grid(True)
        axs[idx].legend(loc="upper right")

    axs[-1].set_xlabel("Time Step")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
