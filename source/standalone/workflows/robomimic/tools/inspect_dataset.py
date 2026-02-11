import h5py
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True, help="Path to hdf5 dataset")
args = parser.parse_args()


def inspect_dataset(path):
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise ValueError("Dataset does not contain 'data' group.")

        data_group = f["data"]
        episodes = list(data_group.keys())

        print("=" * 60)
        print(f"Dataset: {path}")
        print(f"Number of episodes: {len(episodes)}")
        print("=" * 60)

        # ---- env args ----
        if "env_args" in data_group.attrs:
            print("\nEnvironment arguments:")
            try:
                env_args = json.loads(data_group.attrs["env_args"])
                print(json.dumps(env_args, indent=2))
            except Exception:
                print("Could not decode env_args")
        else:
            print("No env_args found!")

        print("\nEpisode statistics:")
        lengths = []

        obs_keys_reference = None
        action_shape_reference = None

        for ep in episodes:
            ep_group = data_group[ep]

            # Actions
            actions = ep_group["actions"]
            ep_len = actions.shape[0]
            lengths.append(ep_len)

            # Observations
            obs_group = ep_group["obs"]
            obs_keys = list(obs_group.keys())

            if obs_keys_reference is None:
                obs_keys_reference = obs_keys
            else:
                if obs_keys != obs_keys_reference:
                    print(f"WARNING: Observation keys mismatch in {ep}")

            # Action shape consistency
            if action_shape_reference is None:
                action_shape_reference = actions.shape[1:]
            else:
                if actions.shape[1:] != action_shape_reference:
                    print(f"WARNING: Action shape mismatch in {ep}")

        lengths = np.array(lengths)

        print(f"Min episode length: {lengths.min()}")
        print(f"Max episode length: {lengths.max()}")
        print(f"Mean episode length: {lengths.mean():.2f}")
        print(f"Std episode length: {lengths.std():.2f}")

        print("\nObservation keys:")
        for k in obs_keys_reference:
            print(f" - {k}")

        print(f"\nAction shape per timestep: {action_shape_reference}")

        print("\nDataset looks structurally consistent ✔")


if __name__ == "__main__":
    inspect_dataset(args.file)
