"""ANYbotics ANYmal C constants."""

# Idx  | Name                 | qpos_addr | qpos_count | qpos_indices
# -----+----------------------+-----------+------------+-------------------------
# 0    | root                 | 0         | 7          | [0-6]
# 1    | FL_HIP_JOINT         | 7         | 1          | [7]
# 2    | FL_THIGH_JOINT       | 8         | 1          | [8]
# 3    | FL_CALF_JOINT        | 9         | 1          | [9]
# 4    | FR_HIP_JOINT         | 10        | 1          | [10]
# 5    | FR_THIGH_JOINT       | 11        | 1          | [11]
# 6    | FR_CALF_JOINT        | 12        | 1          | [12]
# 7    | RL_HIP_JOINT         | 13        | 1          | [13]
# 8    | RL_THIGH_JOINT       | 14        | 1          | [14]
# 9    | RL_CALF_JOINT        | 15        | 1          | [15]
# 10   | RR_HIP_JOINT         | 16        | 1          | [16]
# 11   | RR_THIGH_JOINT       | 17        | 1          | [17]
# 12   | RR_CALF_JOINT        | 18        | 1          | [18]


from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

_HERE = Path(__file__).parent

HJE_C_XML: Path = _HERE / "xmls" / "hje.xml"
assert HJE_C_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, HJE_C_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(HJE_C_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

EFFORT_LIMIT_1 = 100.0 # 
EFFORT_LIMIT_2 = 150.0

# Random small armature since we don't know the real value.
ARMATURE_1 = 0.005 # Rotor inertia J, 1.48 [kgcm²]
ARMATURE_2 = 0.005 # Rotor inertia J, 1.48 [kgcm²]

# PD gains derived from armature, targeting 10 Hz natural frequency.
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS = ARMATURE * NATURAL_FREQ**2
DAMPING = 2 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ

ANYMAL_C_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*HAA", ".*HFE", ".*KFE"),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.54),
  joint_pos={
    ".*HAA": 0.0,
    "LF_HFE": 0.4,
    "RF_HFE": 0.4,
    "LH_HFE": -0.4,
    "RH_HFE": -0.4,
    "LF_KFE": -0.8,
    "RF_KFE": -0.8,
    "LH_KFE": 0.8,
    "RH_KFE": 0.8,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = r"^[LR][FH]_foot$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision", _foot_regex),
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp={_foot_regex: (0.015, 1, 0.03)},
)

##
# Final config.
##

ANYMAL_C_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(ANYMAL_C_ACTUATOR_CFG,),
  soft_joint_pos_limit_factor=0.9,
)


def get_anymal_c_robot_cfg() -> EntityCfg:
  """Get a fresh ANYmal C robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=ANYMAL_C_ARTICULATION,
  )


ANYMAL_C_ACTION_SCALE: dict[str, float] = {}
for _a in ANYMAL_C_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  _e = _a.effort_limit
  _s = _a.stiffness
  _d = _a.damping
  _names = _a.target_names_expr
  assert _e is not None
  for _n in _names:
    ANYMAL_C_ACTION_SCALE[_n] = 0.25 * _e / _s


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_anymal_c_robot_cfg())

  viewer.launch(robot.spec.compile())
