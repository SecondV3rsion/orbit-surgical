"""Configuration for the MOPS robot.

The following configurations are available:

* :obj:`MOPS_CFG`: Kuka robot arm + Lnd Xi tool
* :obj:`MOPS_HIGH_PD_CFG`: Kuka robot arm + Lnd Xi tool with stiffer PD control

"""

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.math import quat_from_euler_xyz
import torch

##
# Configuration
##

# Given real-world position
world_to_base_pos = [0, 0, 0]

# Given real-world RPY (roll, pitch, yaw) in radians
world_to_base_rpy = torch.tensor([0, 0, 0]) 

# Convert RPY to quaternion (expects separate roll, pitch, yaw)
world_to_base_quat = quat_from_euler_xyz(world_to_base_rpy[0], world_to_base_rpy[1], world_to_base_rpy[2])


initial_joint_positions = {
    "kuka_A1": 0.0,
    "kuka_A2": 0.0,
    "kuka_A3": 0.0,
    "kuka_A4": -1.27,
    "kuka_A5": 0.0,
    "kuka_A6": 0.31,
    "kuka_A7": 0.0,
    "tool_roll": 0.01,
    "tool_pitch": 0.01,
    "tool_yaw0": 0.01,
    "tool_yaw1": -0.09,
    "tool_yaw2": 0.09,
}


MOPS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Robots/MOPS/mopsV2/mopsV2.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=initial_joint_positions,
        pos=world_to_base_pos,
        rot=world_to_base_quat.tolist(),
    ),
    actuators={
        "kuka": ImplicitActuatorCfg(
            joint_names_expr=[
                "kuka_A1",
                "kuka_A2",
                "kuka_A3",
                "kuka_A4",
                "kuka_A5",
                "kuka_A6",
                "kuka_A7",
            ],
            effort_limit_sim=400.0,
            velocity_limit_sim=2.175,
            stiffness=400.0,
            damping=40.0,
        ),
        "tool": ImplicitActuatorCfg(
            joint_names_expr=[
                "tool_roll",
                "tool_pitch",
                "tool_yaw0",
                "tool_yaw1",
                "tool_yaw2",
            ],
            effort_limit_sim=40.0,
            velocity_limit_sim=1,
            stiffness=500,
            damping=0.1,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of MOPS robot arm."""


MOPS_HIGH_PD_CFG = MOPS_CFG.copy()
MOPS_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
MOPS_HIGH_PD_CFG.actuators["kuka"].stiffness = 600.0
MOPS_HIGH_PD_CFG.actuators["kuka"].damping = 80.0
"""Configuration of MOPS robot arm with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
