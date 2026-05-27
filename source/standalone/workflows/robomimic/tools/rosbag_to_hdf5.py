#!/usr/bin/env python3
"""
rosbag_to_hdf5.py
-----------------
Convert a ROS2 bag (recorded with pedal_recorder or similar) into an HDF5
dataset that matches the Isaac Lab / Robomimic schema expected by BC training.

Expected HDF5 output structure
-------------------------------
data/
  demo_0/
    actions          (T, action_dim)          – commanded EEF delta pose + gripper
    obs/
      eef_pos        (T, 3)
      eef_quat       (T, 4)
      joint_pos      (T, N)
      joint_vel      (T, N)
      gripper_pos    (T, 2)
      actions        (T, action_dim)          – obs copy of actions (Robomimic convention)
  demo_1/
    ...


"""

import argparse
import os
import sys
from pathlib import Path
import json

import sqlite3
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import get_typestore, Stores
except ImportError:
    sys.exit(
        "rosbags is not installed.\n"
        "Install with:  pip install rosbags"
    )

# ---------------------------------------------------------------------------
# ── CONFIGURATION ──────────────────────────────────────────────────────────
# 
# Edit these to match what your robot publishes.
# ---------------------------------------------------------------------------

# ── Isaac Lab sim parameters (must match your env cfg) ─────────────────────
ISAAC_SCALE = 0.1   # scale= in DifferentialInverseKinematicsActionCfg

# ── Gripper constants ──────────────────────────────────────────────────────────
GRIPPER_OPEN_VALUE   = 0.6    # grasper_angle value that means "open"
GRIPPER_CLOSED_VALUE = 0.08   # grasper_angle value that means "closed"
GRIPPER_THRESHOLD    = (GRIPPER_OPEN_VALUE + GRIPPER_CLOSED_VALUE) / 2 

TOPIC_MAP: dict[str, str] = {
    # Observations  (mops_msgs/ObjectLiftObs)
    "/a/mops_state": "mops_msgs/msg/ObjectLiftObs",

    # Action commands    (mops_msgs/ToolEndEffectorState)
    "/a/servo_joint_ik": "mops_msgs/msg/ToolEndEffectorState",
}

# ---------------------------------------------------------------------------
# Bag reading
# ---------------------------------------------------------------------------
def read_bag(bag_path: str) -> dict[str, list[tuple[float, object]]]:
    """
    Read all messages from the bag for the topics we care about.

    Args:
        bag_path: Path to the rosbag2 directory (contains metadata.yaml + *.db3).

    Returns:
        {topic: [(timestamp_sec, msg), ...]}
        Timestamps are in seconds (float). Messages are fully deserialised
        ROS message objects – ready to be flattened into HDF5 datasets.
    """
    bag_path = Path(bag_path)

    # rosbag2 stores data in a .db3 file inside the bag directory
    db_files = sorted(bag_path.glob("*.db3"))
    if not db_files:
        raise FileNotFoundError(f"No .db3 file found in {bag_path}")

    # Pre-build deserialiser for every topic we care about
    type_map: dict[str, type] = {
        topic: get_message(msg_type)
        for topic, msg_type in TOPIC_MAP.items()
    }

    data: dict[str, list[tuple[float, object]]] = {t: [] for t in TOPIC_MAP}

    for db_file in db_files:
        con = sqlite3.connect(str(db_file))
        cur = con.cursor()

        # Build topic_name -> msg_type_name from the bag's own registry
        cur.execute("SELECT id, name, type FROM topics")
        topic_rows = {name: (tid, typ) for tid, name, typ in cur.fetchall()}

        # Only query topics that exist in this db file AND in TOPIC_MAP
        relevant = {
            name: topic_rows[name][0]          # name -> topic_id
            for name in TOPIC_MAP
            if name in topic_rows
        }
        if not relevant:
            con.close()
            continue

        placeholders = ",".join("?" * len(relevant))
        cur.execute(
            f"SELECT topic_id, timestamp, data FROM messages "
            f"WHERE topic_id IN ({placeholders}) ORDER BY timestamp",
            list(relevant.values()),
        )

        # Reverse map: topic_id -> topic_name  (for the lookup below)
        id_to_name = {tid: name for name, tid in relevant.items()}

        for topic_id, timestamp_ns, raw in cur.fetchall():
            topic_name = id_to_name[topic_id]
            msg = deserialize_message(raw, type_map[topic_name])
            timestamp_sec = timestamp_ns * 1e-9
            data[topic_name].append((timestamp_sec, msg))

        con.close()

    return data

# ---------------------------------------------------------------------------
# Action processing
# ---------------------------------------------------------------------------

def build_action_array_rel(actions: np.ndarray) -> np.ndarray:
    """
    Convert absolute EEF pose actions to relative (delta) actions.

    Input:
        actions : (T, 8) float32  —  [x, y, z, qx, qy, qz, qw, grasper_angle]
                  as stored in msg.action

    Output:
        np.ndarray (T, 7) float32  —  [dx, dy, dz, droll, dpitch, dyaw, gripper]
        Matches IsaacLab's DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True)
        gripper: +1 = open, -1 = close
    """
    T = len(actions)
    out = np.zeros((T, 7), dtype=np.float32)

    pos  = actions[:, 0:3]           # (T, 3)
    quat = actions[:, 3:7]           # (T, 4)  x,y,z,w
    grip = actions[:, 7]             # (T,)

    rots = R.from_quat(quat)         # scipy expects x,y,z,w — matches ROS

    for i in range(T):
        if i == 0:
            # First step: no previous state, zero delta
            out[i, 0:3] = 0.0
            out[i, 3:6] = 0.0
        else:
            # Position delta
            out[i, 0:3] = (pos[i] - pos[i - 1]).astype(np.float32)

            # Rotation delta: R_rel = R_cur * R_prev^{-1}
            delta_rot    = rots[i] * rots[i - 1].inv()
            out[i, 3:6]  = delta_rot.as_euler('xyz').astype(np.float32)

        out[i, 6] = _gripper_binary(grip[i])

    return out

def build_action_array_abs(actions: np.ndarray) -> np.ndarray:
    """
    Pass through absolute EEF pose actions for IsaacLab's DifferentialIKController.
    Input:
        actions : (T, 8) float32  —  [x, y, z, qx, qy, qz, qw, grasper_angle]
                  as stored in msg.action
    Output:
        np.ndarray (T, 8) float32  —  [x, y, z, qw, qx, qy, qz, gripper]
        Matches IsaacLab's DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False)
        gripper: +1 = open, -1 = close
    """
    T = len(actions)
    out = np.zeros((T, 8), dtype=np.float32)

    out[:, 0:3] = actions[:, 0:3]          # x, y, z  — pass through directly
    out[:, 3]   = actions[:, 6]            # qw  (was at index 6)
    out[:, 4:7] = actions[:, 3:6]          # qx, qy, qz  (were at indices 3,4,5)
    out[:, 7]   = np.array([_gripper_binary(g) for g in actions[:, 7]], dtype=np.float32)

    return out

def build_action_array_joint(actions: np.ndarray, joint_pos_rel: np.ndarray) -> np.ndarray:
    """
    Build joint position actions for IsaacLab's JointPositionActionCfg +
    BinaryJointPositionActionCfg.

    Input:
        actions       : (T, 8)  — [x, y, z, qx, qy, qz, qw, gripper_angle]
        joint_pos_rel : (T, 11) — [kuka_A1..A7, tool_roll, tool_pitch, tool_yaw0, tool_yaw1, tool_yaw2]
                                   9 body joints + 2 finger joints

    Output:
        np.ndarray (T, 10) — [kuka_A1..A7, tool_roll, tool_pitch, tool_yaw | gripper_binary]
                              9 body joints for JointPositionActionCfg
                              1 binary value for BinaryJointPositionActionCfg
    """
    T = len(actions)
    N = joint_pos_rel.shape[1] - 3          # 12 - 3 = 9 body joints

    out = np.zeros((T, N + 1), dtype=np.float32)
    out[:, :N] = joint_pos_rel[:, :N]       # first 9 cols only, excludes tool_yaw1/2
    out[:, N]  = np.array([_gripper_binary(g) for g in actions[:, 7]], dtype=np.float32)

    return out

# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def write_hdf5(
    data: dict[str, list[tuple[float, object]]],
    output_path: str,
    env_name: str = "mops_lift",
    obs_topic: str = "/a/mops_state",
    robot_root_pose: np.ndarray | None = None,      # (1,7) [x,y,z, qx,qy,qz,qw] of robot base in world;
) -> None:
    """
    Write a single-demo HDF5 file matching IsaacLab / robomimic structure exactly.

    Structure:
        data/
            (attr) total
            (attr) env_args
            demo_0/
                actions                         (T, 7)
                initial_state/
                    articulation/robot/
                        joint_position          (1, 12)
                        joint_velocity          (1, 12)
                        root_pose               (1, 7)
                        root_velocity           (1, 6)
                    rigid_object/object/
                        root_pose               (1, 7)
                        root_velocity           (1, 6)
                obs/
                    actions                     (T, 7)
                    eef_pos                     (T, 3)
                    eef_quat                    (T, 4)
                    gripper_pos                 (T, 2)
                    joint_pos                   (T, 12)
                    joint_vel                   (T, 12)
                    object_position             (T, 3)
                    target_object_position      (T, 3)
                states/
                    articulation/robot/
                        joint_position          (T, 12)
                        joint_velocity          (T, 12)
                        root_pose               (T, 7)
                        root_velocity           (T, 6)
                    rigid_object/object/
                        root_pose               (T, 7)
                        root_velocity           (T, 6)              
    """
    observations = data[obs_topic]
    T = len(observations)
    if T == 0:
        raise ValueError(f"No messages found on topic {obs_topic!r}")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    hdf5_file = output_path / "demo.hdf5"

    # ── Unpack observations ────────────────────────────────────────────────
    ros_actions = np.zeros((T, 8), dtype=np.float32)
    actions = np.zeros((T, 7), dtype=np.float32)
    joint_pos_abs   = np.zeros((T, 12), dtype=np.float32)
    joint_vel_abs   = np.zeros((T, 12), dtype=np.float32)
    joint_pos_rel   = np.zeros((T, 12), dtype=np.float32)
    joint_vel_rel   = np.zeros((T, 12), dtype=np.float32)
    eef_pos     = np.zeros((T,  3), dtype=np.float32)
    eef_quat    = np.zeros((T,  4), dtype=np.float32)
    object_pos  = np.zeros((T,  3), dtype=np.float32)
    gripper_pos = np.zeros((T,  2), dtype=np.float32)

    valid_indices = []

    for i, (ts, msg) in enumerate(observations):
        action = msg.action
        
        # Skip if action contains any NaN
        if np.any(np.isnan(action)):
            continue
        
        ros_actions[i]    = action
        joint_pos_abs[i]  = msg.q_pos_abs
        joint_vel_abs[i]  = msg.q_vel_abs
        joint_pos_rel[i]  = msg.q_pos_rel
        joint_vel_rel[i]  = msg.q_vel_rel
        eef_pos[i]        = msg.eef_pos
        eef_quat[i]       = msg.eef_quat
        object_pos[i]     = msg.object_pos
        gripper_pos[i]    = msg.gripper_pos
        
        valid_indices.append(i)

    # Trim arrays to only valid rows
    ros_actions   = ros_actions[valid_indices]
    joint_pos_abs = joint_pos_abs[valid_indices]
    joint_vel_abs = joint_vel_abs[valid_indices]
    joint_pos_rel = joint_pos_rel[valid_indices]
    joint_vel_rel = joint_vel_rel[valid_indices]
    eef_pos       = eef_pos[valid_indices]
    eef_quat      = eef_quat[valid_indices]
    object_pos    = object_pos[valid_indices]
    gripper_pos   = gripper_pos[valid_indices]

    # target = fixed goal = last observed object position
    target_object_position = np.tile(object_pos[-1], (T, 1))  # (T, 3)

    # ── initial_state values (first timestep) ─────────────────────────────
    # Robot base in world frame — pass in if known, otherwise identity
    if robot_root_pose is None:
        robot_root_pose = np.array([[0.0, 0.0, -0.2, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # (1,7) TO DO: UPDTE DEFAULT TO MATCH YOUR ROBOT

    robot_joint_pos_init = np.array([[0.0, 0.0, 0.0, -1.27, 0.0, 0.31, 0.0, 0.01, 0.01, 0.01, 0.6, 0.6]], dtype=np.float32)  # (1, 12)
    
    # Object pose at t=0: xyz from obs, but with fixed neutral orientation (since we don't observe object orientation in the MOPS state)    
    obj_root_pose_init = np.array(
        [[object_pos[0, 0], object_pos[0, 1], object_pos[0, 2], 0.7071068, 0.0, 0.0, 0.7071068]],
        dtype=np.float32,
    )

    env_args = json.dumps({
        "env_name":   env_name,
        "env_type":   2,
        "env_kwargs": {},
    })

    # ── Write ──────────────────────────────────────────────────────────────
    with h5py.File(hdf5_file, "w") as f:

        data_grp = f.create_group("data")
        data_grp.attrs["total"]    = T
        data_grp.attrs["env_args"] = env_args

        demo = data_grp.create_group("demo_0")
        demo.attrs["num_samples"] = T

        # actions (top-level)
        #actions = build_action_array_rel(ros_actions)  # Rel actions
        #actions = build_action_array_abs(ros_actions)  # Abs actions
        actions = build_action_array_joint(ros_actions, joint_pos_rel)   # Joint actions
        print(f"Writing actions with shape {actions.shape} to HDF5...")
        demo.create_dataset("actions",    data=actions)

        # ── initial_state ──────────────────────────────────────────────────
        init = demo.create_group("initial_state")

        init_robot = init.create_group("articulation/robot")
        init_robot.create_dataset("joint_position", data=robot_joint_pos_init)                      # (1, 12)
        init_robot.create_dataset("joint_velocity", data=np.zeros((1, 12), dtype=np.float32))       # (1, 12)
        init_robot.create_dataset("root_pose",      data=robot_root_pose)                           # (1, 7)
        init_robot.create_dataset("root_velocity",  data=np.zeros((1, 6), dtype=np.float32))        # (1, 6)

        init_obj = init.create_group("rigid_object/object")
        init_obj.create_dataset("root_pose",     data=obj_root_pose_init)                           # (1, 7)
        init_obj.create_dataset("root_velocity", data=np.zeros((1, 6), dtype=np.float32))           # (1, 6)

        # ── obs ────────────────────────────────────────────────────────────
        obs_grp = demo.create_group("obs")
        obs_grp.create_dataset("actions",                data=actions)                      # (T, 7)
        obs_grp.create_dataset("eef_pos",                data=eef_pos)                      # (T, 3)
        obs_grp.create_dataset("eef_quat",               data=eef_quat)                     # (T, 4)
        obs_grp.create_dataset("gripper_pos",            data=gripper_pos)                  # (T, 2)
        obs_grp.create_dataset("joint_pos",              data=joint_pos_rel)                    # (T, 12)
        obs_grp.create_dataset("joint_vel",              data=joint_vel_rel)                    # (T, 12)
        obs_grp.create_dataset("object_position",        data=object_pos)                   # (T, 3)
        obs_grp.create_dataset("target_object_position", data=target_object_position)       # (T, 3)

        # ── states (per-timestep) ──────────────────────────────────────────
        states = demo.create_group("states")

        states_robot = states.create_group("articulation/robot")
        states_robot.create_dataset("joint_position", data=joint_pos_abs)                           # (T, 12)
        states_robot.create_dataset("joint_velocity", data=joint_vel_abs)                           # (T, 12)
        states_robot.create_dataset("root_pose",      data=np.tile(robot_root_pose, (T, 1)))    # (T, 7)
        states_robot.create_dataset("root_velocity",  data=np.zeros((T, 6), dtype=np.float32))  # (T, 6)

        states_obj = states.create_group("rigid_object/object")
        states_obj.create_dataset("root_pose",     data=np.tile(obj_root_pose_init, (T, 1)))    # (T, 7)
        states_obj.create_dataset("root_velocity", data=np.zeros((T, 6), dtype=np.float32))     # (T, 6)

    print(f"Wrote {T} timesteps → {hdf5_file}")

# ── Helpers ────────────────────────────────────────────────────────────────────

def _quat_to_euler(x: float, y: float, z: float, w: float) -> np.ndarray:
    """ROS quaternion (x,y,z,w) → Euler angles (roll, pitch, yaw) in radians."""
    return R.from_quat([x, y, z, w]).as_euler("xyz")   # intrinsic XYZ = roll-pitch-yaw


def _gripper_binary(grasper_angle: float) -> float:
    """Convert continuous grasper_angle to binary action: +1 = open, -1 = close."""
    return 1.0 if grasper_angle >= GRIPPER_THRESHOLD else -1.0


def _nearest_action_idx(action_timestamps: np.ndarray, query_ts: float) -> int:
    """Return the index of the action whose timestamp is closest to query_ts."""
    return int(np.argmin(np.abs(action_timestamps - query_ts)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert a ROS2 bag to Isaac Lab / Robomimic HDF5 format."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the rosbag2 directory (the folder that contains metadata.yaml)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path for the output .hdf5 file",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="Isaac-Lift-Needle-MOPS-IK-Rel-v0",
        help="Environment name written into HDF5 metadata (default: Isaac-Lift-Needle-MOPS-IK-Rel-v0)",
    )
    args = parser.parse_args()

    # ── Validate input ─────────────────────────────────────────────────────
    if not Path(args.input_path).exists():
        sys.exit(f"Input path does not exist: {args.input_path}")

    # ── Read bag ───────────────────────────────────────────────────────────
    print(f"Reading bag: {args.input_path}")
    data = read_bag(args.input_path)

    write_hdf5(data, args.output_path, env_name=args.env_name)


if __name__ == "__main__":
    main()
