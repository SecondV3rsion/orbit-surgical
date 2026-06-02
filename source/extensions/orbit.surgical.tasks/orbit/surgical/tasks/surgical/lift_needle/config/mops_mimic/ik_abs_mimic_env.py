import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLMimicEnv
import isaaclab.utils.math as PoseUtils


class NeedleLiftMimicEnv(ManagerBasedRLMimicEnv):
    """
    Isaac Lab Mimic environment wrapper class for Needle Lift env.
    Provides access to subtask signals (grasp, object_lifted, goal_reached).
    """

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Args:
            eef_name: Name of the end effector.
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
        """
        if env_ids is None:
            env_ids = slice(None)

        # Retrieve end effector pose from the observation buffer
        robot_base_pos = self.scene["robot"].data.root_pos_w[env_ids]  # (N, 3)
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids] - robot_base_pos
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        # Quaternion format is w,x,y,z
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))
    
    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """
        Converts absolute target EEF pose into absolute action format.
        Mimics IsaacLab relative action implementation structure.
        """
        # target position and rotation
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        # get gripper action
        (gripper_action,) = gripper_action_dict.values()

        # add noise to action
        pose_action = torch.cat([target_pos, PoseUtils.quat_from_matrix(target_rot)], dim=0)
        if action_noise_dict is not None:
            noise = action_noise_dict["mops"] * torch.randn_like(pose_action)
            pose_action += noise
        
        return torch.cat([pose_action, gripper_action], dim=0).unsqueeze(0)
    
    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert action to target pose."""
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        target_pos = action[:, :3]
        target_quat = action[:, 3:7]
        target_rot = PoseUtils.matrix_from_quat(target_quat)

        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()

        return {eef_name: target_poses}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts the gripper actuation part from a sequence of env actions (compatible with env.step).

        Args:
            actions: environment actions. The shape is (num_envs, num steps in a demo, action_dim).

        Returns:
            A dictionary of torch.Tensor gripper actions. Key to each dict is an eef_name.
        """
        # last dimension is gripper action
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}

    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Gets a dictionary of termination signal flags for each subtask in a task.
        The flag is 1 when the subtask has been completed and 0 otherwise.

        This is used for automatic subtask term signal annotation during dataset
        annotation. Can be skipped if you plan to annotate manually.
        """
        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]

        signals["grasp"] = subtask_terms["grasp"][env_ids]
        signals["object_lifted"] = subtask_terms["object_lifted"][env_ids]
        signals["goal_reached"] = subtask_terms["goal_reached"][env_ids]

        return signals
