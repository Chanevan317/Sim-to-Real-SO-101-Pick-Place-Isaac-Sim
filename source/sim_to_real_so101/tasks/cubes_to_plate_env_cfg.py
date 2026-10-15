import os
import numpy as np


import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg, ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from sim_to_real_so101 import assets
from sim_to_real_so101.assets.so101 import S0101_CONTACT_GRASP_CFG
from sim_to_real_so101.mdp import (
    reset_cubes_plate,
    reset_two_cubes_plate,
    randomize_sky_light,
    ROBOT_COLORS,
    randomize_robot_color,
    set_cube_color,
    any_cube_grasped,
    cube_placed_on_plate,
    cube_placed_on_plate_termination,
    time_out,
)

from .so101_env_cfg import EventCfg
from .task_env_cfg import (
    SO101TaskSceneCfg,
    SO101TaskEnvCfg,
    TaskEventCfg,
    TaskObservationsCfg,
)

assets_path = os.path.dirname(os.path.abspath(assets.__file__))

manipulation_object_base = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/ManipulationObject",
    spawn=sim_utils.UsdFileCfg(usd_path=""),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.06),
    ),
)

cube = manipulation_object_base.replace()
cube.spawn.usd_path = f"{assets_path}/usd/cube.usda"
cube.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.02)
cube.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(angular_damping=100.0)


plate = manipulation_object_base.replace()
plate.prim_path = "{ENV_REGEX_NS}/CubePlate"
plate.spawn.usd_path = f"{assets_path}/usd/plate.usda"
plate.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.2)
# Kinematic: immune to external forces/collisions (the arm can't shove or knock
# it), but write_root_pose_to_sim() in reset_cubes_plate still works to
# reposition it each episode - it just teleports the kinematic target instead
# of applying a physical impulse. A fully static asset would block that reset
# randomization entirely, so kinematic is the right middle ground here.
plate.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)

cube.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.02)
cube.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(angular_damping=100.0)
CUBE_SPAWN_Z = 0.05


@configclass
class CubesToPlateSceneCfg(SO101TaskSceneCfg):
    num_envs = 1

    # Override robot with contact sensors enabled
    robot: ArticulationCfg = S0101_CONTACT_GRASP_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    cube_1 = cube.replace()
    cube_1.prim_path = "{ENV_REGEX_NS}/Cube_1"
    cube_1.init_state.pos = (0.23, -0.08, CUBE_SPAWN_Z)
    cube_1.init_state.rot = euler_angles_to_quat(np.array([0, 90, 0]), degrees=True)

    # cube_2 = cube.replace()
    # cube_2.prim_path = "{ENV_REGEX_NS}/Cube_2"
    # cube_2.init_state.pos = (0.23, 0, CUBE_SPAWN_Z)
    # cube_2.init_state.rot = euler_angles_to_quat(np.array([0, 90, 0]), degrees=True)

    # cube_3 = cube.replace()
    # cube_3.prim_path = "{ENV_REGEX_NS}/Cube_3"
    # cube_3.init_state.pos = (0.23, -0.16, CUBE_SPAWN_Z)
    # cube_3.init_state.rot = euler_angles_to_quat(np.array([0, 90, 0]), degrees=True)

    plate_left = plate.replace()
    plate_left.prim_path = "{ENV_REGEX_NS}/Plate_Left"
    plate_left.init_state.pos = (0.18, 0.08, 0.0275)

    # Contact sensor on gripper jaw to detect cube grasping
    contact_grasp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Cube_1",
        ],
    )


@configclass
class CubesToPlateDRSceneCfg(CubesToPlateSceneCfg):
    sky_light = AssetBaseCfg(
        prim_path="/World/sky_light",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{assets_path}/hdri/moon_lab_1k.exr",
            visible_in_primary_ray=False,
            enable_color_temperature=True,
            color_temperature=6500.0,
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()


@configclass
class TwoCubesToPlateDRSceneCfg(CubesToPlateDRSceneCfg):
    """Two-cube variant: cube_2 shares cube_1's default anchor on purpose, so
    reset_two_cubes_plate's shared pose_range gives both cubes the same
    absolute spawn distribution -- position alone can't give away which one
    is the target, only color/instruction can."""

    cube_2 = cube.replace()
    cube_2.prim_path = "{ENV_REGEX_NS}/Cube_2"
    cube_2.init_state.pos = (0.23, -0.08, CUBE_SPAWN_Z)
    cube_2.init_state.rot = euler_angles_to_quat(np.array([0, 90, 0]), degrees=True)

    # Contact sensor must list both cubes (order matches the `cubes` list
    # passed to any_cube_grasped) or cube_2 grasps are silently never detected.
    contact_grasp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Cube_1",
            "{ENV_REGEX_NS}/Cube_2",
        ],
    )


@configclass
class CubesToPlateEventCfg(TaskEventCfg):
    """Configuration for events."""

    reset_cubes_setup = EventTerm(
        func=reset_cubes_plate,
        mode="reset",
        params={
            "cubes": ["cube_1"],
            "plate": "plate_left",
            "plate_pose_range": {
                "x": (-0.04, 0.04),
                "y": (0.08, 0.1),
                "yaw": (-0.5, 0.5),
            },
            "pose_range": {
                "x": (-0.04, 0.04),
                "y": (-0.01, 0.01),
                "roll": (-0.3, 0.3),
                "yaw": (0.0, 0.0),
            },
            "fixed_cube_z": 0.05,
            # For teleop data collection we want every episode to start with
            # the cube loose (never pre-placed on the plate), so every demo
            # shows a full pick-and-place trajectory. The function's own
            # default (0.33) is for other configs (e.g. eval/RL) that may
            # want that initial-state variety.
            "plate_placement_prob": 0.0,
        },
    )


@configclass
class CubesToPlateEventDRCfg(CubesToPlateEventCfg):

    reset_set_robot_visual_material = EventTerm(
        func=randomize_robot_color,
        mode="reset",
        params={
            "color_names": list(ROBOT_COLORS.keys()),
        },
    )

    reset_sky_light = EventTerm(
        func=randomize_sky_light,
        mode="reset",
        params={
            "exposure_range": (-4.0, 3.0),
            "temperature_range": (2500.0, 9500.0),
            "textures_root": f"{assets_path}/hdri",
            "asset_cfg": SceneEntityCfg("sky_light"),
        },
    )


# Spawn region shared by both cubes in the two-cube task, as an offset from
# their shared default anchor (0.23, -0.08): cube centres land in
# x [0.17, 0.23], y [-0.15, -0.01]. Sized from the arm's reach: a forward-
# kinematics sweep (all joints >= 10 deg from their limits) allows a steep,
# top-down grasp at cube height up to ~0.32 m from the pan axis at
# (-0.027, 0.021); the old single-cube demos already reached 0.317 m. The far
# corner here is 0.308 m away (grasp ~20 deg from vertical), leaving a margin.
# The box is long in y so two cubes still fit min_cube_separation apart:
# reset_two_cubes_plate only resamples the second cube, and in a squarer box
# it often found no free spot and kept an overlapping draw.
# reset_two_cubes_plate rejects draws that put the two cubes, or a cube and
# the plate, closer than min_cube_separation / min_plate_clearance.
TWO_CUBE_POSE_RANGE = {
    "x": (-0.06, 0.0),
    "y": (-0.07, 0.07),
    "roll": (-0.3, 0.3),
    "yaw": (0.0, 0.0),
}


@configclass
class TwoCubesToPlateEventCfg(TaskEventCfg):
    """Configuration for events (two-cube variant)."""

    reset_cube_2_color = EventTerm(
        func=set_cube_color,
        mode="startup",
        params={"cube_name": "cube_2", "color_name": "red"},
    )

    reset_cubes_setup = EventTerm(
        func=reset_two_cubes_plate,
        mode="reset",
        params={
            "cubes": ["cube_1", "cube_2"],
            "plate": "plate_left",
            "plate_pose_range": {
                "x": (-0.04, 0.04),
                "y": (0.08, 0.1),
                "yaw": (-0.5, 0.5),
            },
            "pose_range": TWO_CUBE_POSE_RANGE,
            "fixed_cube_z": 0.05,
            "cube_colors": {"cube_1": "blue", "cube_2": "red"},
            # Cubes are 2 cm wide, so 7 cm between centres leaves a 5 cm gap for the jaws.
            "min_cube_separation": 0.07,
            "min_plate_clearance": 0.12,
        },
    )


@configclass
class TwoCubesToPlateEventDRCfg(TwoCubesToPlateEventCfg):

    reset_set_robot_visual_material = EventTerm(
        func=randomize_robot_color,
        mode="reset",
        params={
            "color_names": list(ROBOT_COLORS.keys()),
        },
    )

    reset_sky_light = EventTerm(
        func=randomize_sky_light,
        mode="reset",
        params={
            "exposure_range": (-4.0, 3.0),
            "temperature_range": (2500.0, 9500.0),
            "textures_root": f"{assets_path}/hdri",
            "asset_cfg": SceneEntityCfg("sky_light"),
        },
    )


@configclass
class CubesToPlateObservationsCfg(TaskObservationsCfg):
    """Configuration for observations."""

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask tracking."""

        cube_grasped = ObsTerm(
            func=any_cube_grasped,
            params={
                "contact_sensor_cfg": SceneEntityCfg("contact_grasp"),
                "cubes": ["cube_1"],
                "min_height": 0.055,  # 5.5 cm - check debug output for actual resting height
                "warmup_steps": 30,
                "force_threshold": 2,  # N
            },
        )

        # cube_placed = ObsTerm(
        #     func=cube_placed_on_plate,
        #     params={
        #         "contact_sensor_cfg": SceneEntityCfg("contact_grasp"),
        #         "cubes": ["cube_1"],
        #         "plate_name": "plate_left",
        #         "warmup_steps": 30,
        #         "force_threshold": 2,  # N
        #         # Plate local dimensions (from plate.usda base mesh extent, centered on origin)
        #         "plate_local_x_min": -0.1,
        #         "plate_local_x_max": 0.1,
        #         "plate_local_y_min": -0.1,
        #         "plate_local_y_max": 0.1,
        #         # Slot entry at local z=0.02; cube center must be below this ceiling
        #         "plate_local_z_max": 0.1,
        #     },
        # )

        cube_placed = ObsTerm(
            func=cube_placed_on_plate,
            params={
                "cubes": ["cube_1"],
                "plate_name": "plate_left",
                # Plate local dimensions (from plate.usda base mesh extent, centered on origin)
                "plate_local_x_min": -0.1,
                "plate_local_x_max": 0.1,
                "plate_local_y_min": -0.1,
                "plate_local_y_max": 0.1,
                # Slot entry at local z=0.02; cube center must be below this ceiling
                "plate_local_z_max": 0.1,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TwoCubesToPlateObservationsCfg(TaskObservationsCfg):
    """Configuration for observations (two-cube variant)."""

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask tracking."""

        cube_grasped = ObsTerm(
            func=any_cube_grasped,
            params={
                "contact_sensor_cfg": SceneEntityCfg("contact_grasp"),
                "cubes": ["cube_1", "cube_2"],
                "min_height": 0.055,
                "warmup_steps": 30,
                "force_threshold": 2,  # N
            },
        )

        cube_placed = ObsTerm(
            func=cube_placed_on_plate,
            params={
                "cubes": ["cube_1", "cube_2"],
                "plate_name": "plate_left",
                "plate_local_x_min": -0.1,
                "plate_local_x_max": 0.1,
                "plate_local_y_min": -0.1,
                "plate_local_y_max": 0.1,
                "plate_local_z_max": 0.1,
                # Only the instructed cube counts -- see restrict_to_target in terms.py.
                "restrict_to_target": True,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class CubesToPlateTerminationsCfg:
    """Termination terms for the cubes to tray evaluation task.

    Success-only: no time_out term here on purpose, so an episode never gets
    truncated (and discarded by the teleop script) before the cube has
    actually been placed. If you want a safety-net time_out too, re-add it,
    but make sure episode_length_s leaves enough headroom for warmup_steps +
    realistic placement time + hold_steps below, or a slow-but-successful
    demo can still get discarded by racing the clock.
    """

    # success = DoneTerm(
    #     func=cube_placed_on_plate_termination,
    #     time_out=False,
    #     params={
    #         "contact_sensor_cfg": SceneEntityCfg("contact_grasp"),
    #         "cubes": ["cube_1"],
    #         "plate_name": "plate_left",
    #         "warmup_steps": 30,
    #         "force_threshold": 2,  # N
    #         "plate_local_x_min": -0.1,
    #         "plate_local_x_max": 0.1,
    #         "plate_local_y_min": -0.1,
    #         "plate_local_y_max": 0.1,
    #         "plate_local_z_max": 0.1,
    #         # ~5 seconds at ~60 control steps/sec (sim.dt=1/120, decimation=2) -
    #         # gives you time to bring the arm back to a neutral pose before reset.
    #         "hold_steps": 120,
    #     },
    # )

    success = DoneTerm(
        func=cube_placed_on_plate_termination,
        time_out=False,
        params={
            "cubes": ["cube_1"],
            "plate_name": "plate_left",
            "plate_local_x_min": -0.1,
            "plate_local_x_max": 0.1,
            "plate_local_y_min": -0.1,
            "plate_local_y_max": 0.1,
            "plate_local_z_max": 0.1,
            # ~3 seconds at ~60 control steps/sec (sim.dt=1/120, decimation=2) -
            # gives you time to bring the arm back to a neutral pose before reset.
            "hold_steps": 180,
        },
    )


@configclass
class TwoCubesToPlateTerminationsCfg:
    """Success-only (see CubesToPlateTerminationsCfg's docstring). Only the
    episode's instructed cube counts -- restrict_to_target=True."""

    success = DoneTerm(
        func=cube_placed_on_plate_termination,
        time_out=False,
        params={
            "cubes": ["cube_1", "cube_2"],
            "plate_name": "plate_left",
            "plate_local_x_min": -0.1,
            "plate_local_x_max": 0.1,
            "plate_local_y_min": -0.1,
            "plate_local_y_max": 0.1,
            "plate_local_z_max": 0.1,
            "hold_steps": 180,
            "restrict_to_target": True,
        },
    )


@configclass
class CubesToPlateEnvCfg(SO101TaskEnvCfg):
    """
    Base config.
    """
    scene: CubesToPlateSceneCfg = CubesToPlateSceneCfg()
    events: CubesToPlateEventCfg = CubesToPlateEventCfg()
    observations: CubesToPlateObservationsCfg = CubesToPlateObservationsCfg()
    terminations: CubesToPlateTerminationsCfg = CubesToPlateTerminationsCfg()


@configclass
class CubesToPlateDREnvCfg(CubesToPlateEnvCfg):
    """
    Domain Randomization config.
    """
    scene: CubesToPlateDRSceneCfg = CubesToPlateDRSceneCfg()
    events: CubesToPlateEventDRCfg = CubesToPlateEventDRCfg()


@configclass
class TwoCubesToPlateDREnvCfg(CubesToPlateEnvCfg):
    """
    Two-cube domain randomization config: cube_1 (blue) and cube_2 (red), one
    picked at random each episode as the instructed target, the other left
    as a distractor. Success requires only the target cube on the plate.
    """
    scene: TwoCubesToPlateDRSceneCfg = TwoCubesToPlateDRSceneCfg()
    events: TwoCubesToPlateEventDRCfg = TwoCubesToPlateEventDRCfg()
    observations: TwoCubesToPlateObservationsCfg = TwoCubesToPlateObservationsCfg()
    terminations: TwoCubesToPlateTerminationsCfg = TwoCubesToPlateTerminationsCfg()


@configclass
class TwoCubesToPlateEvalDREnvCfg(TwoCubesToPlateDREnvCfg):
    """
    Eval config for the two-cube task.
    """


@configclass
class CubesToPlateEvalEnvCfg(CubesToPlateEnvCfg):
    """
    Eval config.
    """

    # def __post_init__(self) -> None:
    #     super().__post_init__()
    #     # Generous ceiling - not relied on for episode boundaries since there's
    #     # no time_out term, just a backstop against a truly stuck episode.
    #     self.episode_length_s = 450 / 60.0


@configclass
class CubesToPlateEvalDREnvCfg(CubesToPlateDREnvCfg):
    """
    Eval config with Domain Randomization.
    """

    # def __post_init__(self) -> None:
    #     super().__post_init__()
    #     self.episode_length_s = 450 / 60.0