"""Configuration for the MOPS robot.

The following configurations are available:

* :obj:`MOPS_CFG`: Kuka robot arm + Lnd Xi tool
* :obj:`MOPS_HIGH_PD_CFG`: Kuka robot arm + Lnd Xi tool with stiffer PD control

"""

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

initial_joint_positions = {
    "kuka_joint_1": -0.4,
    "kuka_joint_2": 1.2,
    "kuka_joint_3": 0.5,
    "kuka_joint_4": -1.2,
    "kuka_joint_5": 1.4,
    "kuka_joint_6": 0.0,
    "kuka_joint_7": 0.0,
    "lnd_tool_roll_joint": 0.01,
    "lnd_tool_pitch_joint": 0.01,
    "lnd_tool_yaw_joint": 0.01,
    "lnd_tool_gripper1_joint": -0.09,
    "lnd_tool_gripper2_joint": 0.09,
}


MOPS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Robots/MOPS/mops.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=20, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=initial_joint_positions,
        pos=(0.0, -0.0765, 0.67),
        rot=(0.0883866, -0.1406036, -0.5248117, 0.8348599),
    ),
    actuators={
        "kuka": ImplicitActuatorCfg(
            joint_names_expr=[
                "kuka_joint_1",
                "kuka_joint_2",
                "kuka_joint_3",
                "kuka_joint_4",
                "kuka_joint_5",
                "kuka_joint_6",
                "kuka_joint_7",
            ],
            effort_limit=50.0,
            velocity_limit=2.0, 
            stiffness=800.0,  
            damping=100.0, 
        ),
        "lnd": ImplicitActuatorCfg(
            joint_names_expr=[
                "lnd_tool_roll_joint",
                "lnd_tool_pitch_joint",
                "lnd_tool_yaw_joint",
            ],
            effort_limit=12.0,
            velocity_limit=1.0,
            stiffness=300.0,
            damping=40.0,
        ),
        "lnd_gripper": ImplicitActuatorCfg(
            joint_names_expr=["lnd_tool_gripper.*"],
            effort_limit=1.0,
            velocity_limit=0.5, 
            stiffness=500,  
            damping=0.1,  
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of MOPS robot arm."""


MOPS_HIGH_PD_CFG = MOPS_CFG.copy()
MOPS_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
MOPS_HIGH_PD_CFG.actuators["kuka"].stiffness = 1000.0
MOPS_HIGH_PD_CFG.actuators["kuka"].damping = 40.0
MOPS_HIGH_PD_CFG.actuators["lnd"].stiffness = 800.0
MOPS_HIGH_PD_CFG.actuators["lnd"].damping = 40.0
"""Configuration of MOPS robot arm with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
