import argparse
import time
import torch
import psutil
import GPUtil

from isaaclab.app import AppLauncher
import cli_args  # for RSL-RL config parsing

# CLI Args
parser = argparse.ArgumentParser(description="Benchmark max environments for IsaacLab task.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable fabric.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import after simulator launched
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import orbit.surgical.tasks  # noqa
import isaaclab_tasks  # noqa


def get_gpu_usage():
    gpus = GPUtil.getGPUs()
    if not gpus:
        return 0.0, 0.0
    return gpus[0].memoryUsed, gpus[0].memoryTotal


def get_cpu_usage():
    mem = psutil.virtual_memory()
    return mem.used / 1e9, mem.total / 1e9


def try_env(task_name: str, num_envs: int, sim_time: float = 5.0):
    try:
        env_cfg = parse_env_cfg(task_name, num_envs=num_envs, use_fabric=not args_cli.disable_fabric)
        env = gym.make(task_name, cfg=env_cfg)
        obs, _ = env.reset()

        device = env.unwrapped.device

        start = time.time()
        frames = 0
        while time.time() - start < sim_time:
            actions = torch.from_numpy(env.action_space.sample()).to(device)
            obs, _, _, _, _ = env.step(actions)
            frames += 1


        total = time.time() - start
        fps = frames / total

        gpu_used, gpu_total = get_gpu_usage()
        cpu_used, cpu_total = get_cpu_usage()

        print(f"✅ {num_envs} envs at {fps:.1f} FPS | GPU: {gpu_used:.1f}/{gpu_total:.1f} GB | CPU: {cpu_used:.1f}/{cpu_total:.1f} GB")
        env.close()
        return True
    except Exception as e:
        print(f"❌ {num_envs} envs failed: {e}")
        return False

def binary_search_max_envs(task_name: str, initial_upper: int = 2048):
    lower = 1
    upper = initial_upper
    best_working = 0

    # Step 1: Expand upward until failure
    while True:
        print(f"\n🚀 Expanding: Trying {upper} environments...")
        success = try_env(task_name, num_envs=upper)
        if success:
            best_working = upper
            lower = upper + 1
            upper *= 2  # double upper bound
        else:
            break  # we hit the failure point

    # Step 2: Binary search between last working and failing
    while lower <= upper:
        mid = (lower + upper) // 2
        print(f"\n🔎 Binary Search: Trying {mid} environments...")
        success = try_env(task_name, num_envs=mid)
        if success:
            best_working = mid
            lower = mid + 1
        else:
            upper = mid - 1

    return best_working

def main():
    print(f"🔧 Benchmarking task: {args_cli.task}")
    max_envs = binary_search_max_envs(args_cli.task, initial_upper=2048)
    print(f"\n✅ Max stable environments: {max_envs}")

if __name__ == "__main__":
    main()
    simulation_app.close()
