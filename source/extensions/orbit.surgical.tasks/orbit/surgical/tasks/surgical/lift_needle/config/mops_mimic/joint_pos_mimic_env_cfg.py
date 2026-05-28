from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from . import ik_abs_env_cfg

@configclass
class NeedleLiftMimicEnvCfg(ik_abs_env_cfg.NeedleLiftEnvCfg, MimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Override datagen config
        self.datagen_config.name = "needle_lift"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # Define subtasks
        subtask_configs = []

        subtask_configs.append(
            SubTaskConfig(
                object_ref="object",  
                subtask_term_signal="grasp",  
                subtask_term_offset_range=(5, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.02,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp needle",
                next_subtask_description="Lift needle",
            )
        )

        subtask_configs.append(
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal="object_lifted",
                subtask_term_offset_range=(5, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.02,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Lift needle",
                next_subtask_description="Move needle to goal",
            )
        )

        subtask_configs.append(
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal="goal_reached",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.02,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place needle at goal",
            )
        )

        # Assign to the config
        self.subtask_configs["needle_lift"] = subtask_configs

