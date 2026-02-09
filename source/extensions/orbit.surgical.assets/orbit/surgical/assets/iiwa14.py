"""Configuration for the MOPS robot.

The following configurations are available:

* :obj:`IIWA14_CFG`: Kuka iiwa14
* :obj:`IIWA14_HIGH_PD_CFG`: Kuka iiwa14 with stiffer PD control

"""

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
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
}

IIWA14_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Robots/iiwa14/iiwa14.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=32, 
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=initial_joint_positions,
        pos=world_to_base_pos,
        rot=world_to_base_quat.tolist(),
    ),
    actuators={
        "kuka": ImplicitActuatorCfg(
            joint_names_expr=[
                "kuka_A(1|2|3|4|5|6|7)",
            ],
            effort_limit_sim=300.0,
            stiffness={
                "kuka_A(1|2|3|4)": 300.0,
                "kuka_A5": 100.0,
                "kuka_A6": 50.0,
                "kuka_A7": 25.0,
            },
            damping={
                "kuka_A(1|2|3|4)": 45.0,
                "kuka_A5": 20.0,
                "kuka_A6": 15.0,
                "kuka_A7": 15.0,
            },
            friction=1.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of IIWA14 robot arm."""

IIWA14_HIGH_PD_CFG = IIWA14_CFG.copy()
IIWA14_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = False
IIWA14_HIGH_PD_CFG.actuators={
        "kuka": ImplicitActuatorCfg(
            joint_names_expr=[
                "kuka_A(1|2|3|4|5|6|7)",
            ],
            effort_limit_sim=300.0,
            stiffness={
                "kuka_A(1|2|3|4)": 900.0,
                "kuka_A5": 300.0,
                "kuka_A6": 150.0,
                "kuka_A7": 75.0,
            },
            damping={
                "kuka_A(1|2|3|4)": 45.0,
                "kuka_A5": 20.0,
                "kuka_A6": 15.0,
                "kuka_A7": 15.0,
            },
            friction=1.0,
        ),
    }
"""Configuration of IIWA14 robot arm with stiffer PD control."""