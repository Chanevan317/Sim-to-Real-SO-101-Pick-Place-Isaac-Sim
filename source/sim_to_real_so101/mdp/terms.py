import torch

from pxr import Gf, Sdf


import isaaclab.utils.math as math_utils

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg



def any_cube_grasped(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    cubes: list[str],
    min_height: float = 0.01,
    warmup_steps: int = 30,
    force_threshold: float = 0.1,
) -> torch.Tensor:
    """Check if any cube is currently grasped by the gripper.
    
    A cube is considered grasped if:
    1. The gripper has contact with it (force above threshold)
    2. The cube is above the minimum height threshold (only checked for initial grasp)
    3. We are past the warmup period
    
    Once grasped, stays grasped as long as contact is maintained (height no longer matters).
    Released only when contact is lost.
    
    Args:
        env: The environment instance.
        contact_sensor_cfg: Configuration for the contact sensor.
        cubes: List of cube asset names to check.
        min_height: Minimum Z height (meters) for cube to be considered lifted. Default 1cm.
        warmup_steps: Number of initial steps to ignore (warmup period). Default 30.
        force_threshold: Minimum contact force (Newtons) to detect contact.
    
    Returns:
        Boolean tensor of shape (num_envs, 1) indicating grasp status per environment.
    """
    num_envs = env.num_envs
    device = env.device
    
    # Initialize state trackers on first call
    if not hasattr(any_cube_grasped, "_prev_grasped"):
        any_cube_grasped._prev_grasped = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if not hasattr(any_cube_grasped, "_is_holding"):
        any_cube_grasped._is_holding = torch.zeros(num_envs, dtype=torch.bool, device=device)
    
    # Check warmup: no cube can be grasped in the first N steps
    current_step = env.episode_length_buf
    in_warmup = current_step < warmup_steps
    
    # Reset holding state for environments that just reset (step 0 or 1)
    just_reset = current_step <= 1
    any_cube_grasped._is_holding[just_reset] = False
    any_cube_grasped._prev_grasped[just_reset] = False
    
    # Get contact sensor
    contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    
    # Get contact forces - shape: (num_envs, num_bodies, num_filters, 3)
    contact_forces = contact_sensor.data.force_matrix_w
    
    # Calculate force magnitude per filter (each filter corresponds to a cube)
    # Shape: (num_envs, num_bodies, num_filters)
    contact_force_norm = torch.linalg.vector_norm(contact_forces, dim=-1)
    
    # Sum over bodies to get per-env, per-filter contact detection
    # Shape: (num_envs, num_filters)
    contact_per_filter = contact_force_norm.sum(dim=1)
    
    # Track contact and lift status across all cubes
    any_contact = torch.zeros(num_envs, dtype=torch.bool, device=device)
    new_grasp = torch.zeros(num_envs, dtype=torch.bool, device=device)
    
    for cube_idx, cube_name in enumerate(cubes):
        cube: RigidObject = env.scene[cube_name]
        cube_z = cube.data.root_pos_w[:, 2]  # Z position per environment
        
        has_contact_with_cube = contact_per_filter[:, cube_idx] > force_threshold
        cube_is_lifted = cube_z > min_height
        
        # Track if we have contact with any cube
        any_contact = any_contact | has_contact_with_cube
        
        # New grasp: contact + lifted + not already holding
        new_grasp = new_grasp | (has_contact_with_cube & cube_is_lifted & (~any_cube_grasped._is_holding))
    
    # Update holding state with hysteresis:
    # - Start holding: new grasp detected (contact + lifted)
    # - Keep holding: was holding AND still have contact
    # - Stop holding: was holding AND lost contact
    was_holding = any_cube_grasped._is_holding.clone()
    any_cube_grasped._is_holding = (was_holding & any_contact) | new_grasp
    
    # Apply warmup mask
    is_grasped = any_cube_grasped._is_holding & (~in_warmup)
    
    # Debug: print state transitions
    prev = any_cube_grasped._prev_grasped
    just_grasped = is_grasped & (~prev)
    just_released = (~is_grasped) & prev
    
    if just_grasped.any():
        env_ids = torch.where(just_grasped)[0].tolist()
        print(f"[GRASP] cube grasped in env(s): {env_ids}")
    
    if just_released.any():
        env_ids = torch.where(just_released)[0].tolist()
        print(f"[RELEASE] cube released in env(s): {env_ids}")
    
    # Update previous state
    any_cube_grasped._prev_grasped = is_grasped.clone()
    
    # Return as float tensor with shape (num_envs, 1) for observation
    return is_grasped.float().unsqueeze(-1)




def cube_placed_on_plate(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    cubes: list[str],
    plate_name: str,
    warmup_steps: int = 30,
    grasp_history_window: int = 20,  # kept for config compatibility; no longer used (see _ever_grasped below)
    force_threshold: float = 2.0,
    # Plate local dimensions (from plate.usda: base mesh spans -0.1..0.1 in x/y around origin)
    plate_local_x_min: float = -0.1,
    plate_local_x_max: float = 0.1,
    plate_local_y_min: float = -0.1,
    plate_local_y_max: float = 0.1,
    plate_local_z_max: float = 0.1,
    # Cube must have settled (near-zero linear velocity) to count as "placed",
    # not just passing over the plate mid-carry.
    settle_lin_vel_threshold: float = 0.02,
) -> torch.Tensor:
    """Check if a cube has been placed on the plate.

    A cube is considered placed if:
    1. We are past the warmup period
    2. The cube's position in plate-local coordinates is within the plate's XY bounding box
    3. The cube's plate-local Z is below plate_local_z_max
    4. The cube has settled (linear speed below settle_lin_vel_threshold) - this is what
       distinguishes "resting on the plate" from "passing over the plate while still being
       carried" or "mid-bounce right after release"
    5. THIS SPECIFIC cube was grasped at some point earlier in the episode
    6. THIS SPECIFIC cube is no longer grasped

    NOTE: there is no orientation/slot requirement - any resting position within the
    plate's footprint counts, upright or on its side.

    NOTE on grasp tracking: this used to check a short rolling window
    (grasp_history_window steps) for "was recently grasped." That window could
    expire before a dropped cube finished settling, causing intermittent misses
    even when the cube was clearly placed. It's now tracked as a persistent
    per-episode flag (_ever_grasped) that cannot expire, so as long as the cube
    is grasped at some point and later found resting on the plate, it will be
    detected regardless of how long it takes to settle. grasp_history_window is
    kept as a parameter only so existing configs don't break; it has no effect.

    Args:
        env: The environment instance.
        contact_sensor_cfg: Configuration for the contact sensor.
        cubes: List of cube asset names to check.
        plate_name: Name of the plate asset in the scene.
        warmup_steps: Number of initial steps to ignore. Default 30.
        grasp_history_window: Unused, kept for config compatibility.
        force_threshold: Minimum contact force (N) to detect grasp.
        plate_local_x_min: plate local X minimum bound.
        plate_local_x_max: plate local X maximum bound.
        plate_local_y_min: plate local Y minimum bound.
        plate_local_y_max: plate local Y maximum bound.
        plate_local_z_max: Maximum plate-local Z for cube center.
        settle_lin_vel_threshold: Max linear speed (m/s) for the cube to count as settled.

    Returns:
        Float tensor of shape (num_envs, 1) indicating placement status per environment.
        This is STICKY: once True for a given env (episode), it stays True until reset.
    """
    num_envs = env.num_envs
    device = env.device
    num_cubes = len(cubes)

    # Initialize state trackers on first call
    if not hasattr(cube_placed_on_plate, "_ever_grasped"):
        cube_placed_on_plate._ever_grasped = torch.zeros(
            num_envs, num_cubes, dtype=torch.bool, device=device
        )
        cube_placed_on_plate._prev_placed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        cube_placed_on_plate._cube_placed_flags = torch.zeros(
            num_envs, num_cubes, dtype=torch.bool, device=device
        )

    current_step = env.episode_length_buf
    in_warmup = current_step < warmup_steps

    just_reset = current_step <= 1
    if just_reset.any():
        cube_placed_on_plate._ever_grasped[just_reset] = False
        cube_placed_on_plate._prev_placed[just_reset] = False
        cube_placed_on_plate._cube_placed_flags[just_reset] = False

    # Get contact sensor for grasp detection
    contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    contact_forces = contact_sensor.data.force_matrix_w
    contact_force_norm = torch.linalg.vector_norm(contact_forces, dim=-1)
    contact_per_filter = contact_force_norm.sum(dim=1)  # (num_envs, num_filters)

    # Get plate pose in world frame
    plate_obj: RigidObject = env.scene[plate_name]
    plate_pos_w = plate_obj.data.root_pos_w       # (num_envs, 3)
    plate_quat_w = plate_obj.data.root_quat_w     # (num_envs, 4)
    plate_quat_inv = math_utils.quat_inv(plate_quat_w)

    any_cube_newly_placed = torch.zeros(num_envs, dtype=torch.bool, device=device)

    for cube_idx, cube_name in enumerate(cubes):
        cube: RigidObject = env.scene[cube_name]
        cube_pos_w = cube.data.root_pos_w       # (num_envs, 3)

        # --- Grasp detection (persistent per-episode, cannot expire) ---
        cube_grasped_now = contact_per_filter[:, cube_idx] > force_threshold
        cube_placed_on_plate._ever_grasped[:, cube_idx] = (
            cube_placed_on_plate._ever_grasped[:, cube_idx] | cube_grasped_now
        )
        cube_was_grasped = cube_placed_on_plate._ever_grasped[:, cube_idx]

        # --- Settled check ---
        # Distinguishes "resting on the plate" from "still moving through/over
        # it" (mid-carry, mid-bounce right after release, etc).
        cube_lin_speed = torch.linalg.vector_norm(cube.data.root_lin_vel_w, dim=-1)
        is_settled = cube_lin_speed < settle_lin_vel_threshold

        # --- Position check in plate-local coordinates ---
        cube_pos_relative = cube_pos_w - plate_pos_w
        cube_pos_local = math_utils.quat_apply(plate_quat_inv, cube_pos_relative)

        cube_local_x = cube_pos_local[:, 0]
        cube_local_y = cube_pos_local[:, 1]
        cube_local_z = cube_pos_local[:, 2]

        x_in_bounds = (cube_local_x >= plate_local_x_min) & (cube_local_x <= plate_local_x_max)
        y_in_bounds = (cube_local_y >= plate_local_y_min) & (cube_local_y <= plate_local_y_max)
        z_below_top = cube_local_z < plate_local_z_max

        position_ok = x_in_bounds & y_in_bounds & z_below_top

        # --- Combine all conditions ---
        cube_is_placed = (
            is_settled
            & position_ok
            & cube_was_grasped
            & (~cube_grasped_now)
            & (~in_warmup)
            & (~cube_placed_on_plate._cube_placed_flags[:, cube_idx])
        )

        # --- Debug: on the step a cube is released after having been grasped,
        # print why it did or didn't count as "placed" this time. Only fires
        # once per release (not every step) to avoid spamming the console.
        just_released_this_cube = cube_was_grasped & (~cube_grasped_now)
        if not hasattr(cube_placed_on_plate, "_prev_grasped_now"):
            cube_placed_on_plate._prev_grasped_now = torch.zeros(
                num_envs, num_cubes, dtype=torch.bool, device=device
            )
        release_edge = just_released_this_cube & cube_placed_on_plate._prev_grasped_now[:, cube_idx]
        if release_edge.any() and not cube_is_placed[release_edge].all():
            env_ids = torch.where(release_edge)[0].tolist()
            for e in env_ids:
                print(
                    f"[PLATE DEBUG] {cube_name} env {e} released but not counted as placed - "
                    f"is_settled={is_settled[e].item()} (speed={cube_lin_speed[e].item():.4f}), "
                    f"x_in_bounds={x_in_bounds[e].item()} (local_x={cube_local_x[e].item():.4f}), "
                    f"y_in_bounds={y_in_bounds[e].item()} (local_y={cube_local_y[e].item():.4f}), "
                    f"z_below_top={z_below_top[e].item()} (local_z={cube_local_z[e].item():.4f})"
                )
        cube_placed_on_plate._prev_grasped_now[:, cube_idx] = cube_grasped_now

        newly_placed = cube_is_placed
        if newly_placed.any():
            env_ids = torch.where(newly_placed)[0].tolist()
            print(f"[PLATE] {cube_name} placed in plate in env(s): {env_ids}")
            cube_placed_on_plate._cube_placed_flags[:, cube_idx] = (
                cube_placed_on_plate._cube_placed_flags[:, cube_idx] | newly_placed
            )

        any_cube_newly_placed = any_cube_newly_placed | newly_placed

    any_placed = cube_placed_on_plate._cube_placed_flags.any(dim=1) & (~in_warmup)

    prev = cube_placed_on_plate._prev_placed
    cube_placed_on_plate._prev_placed = any_placed.clone()

    return any_placed.float().unsqueeze(-1)


def cube_placed_on_plate_termination(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    cubes: list[str],
    plate_name: str,
    warmup_steps: int = 30,
    grasp_history_window: int = 20,  # unused, kept for config compatibility
    force_threshold: float = 2.0,
    plate_local_x_min: float = -0.1,
    plate_local_x_max: float = 0.1,
    plate_local_y_min: float = -0.1,
    plate_local_y_max: float = 0.1,
    plate_local_z_max: float = 0.1,
    settle_lin_vel_threshold: float = 0.02,
    hold_steps: int = 300,
) -> torch.Tensor:
    """Termination term for cube placed on plate.

    Detection is delegated entirely to cube_placed_on_plate(), which is STICKY
    (once it flags a cube as placed for an env, it stays flagged until reset).
    That means we don't need to keep re-validating settled/bounds/grasp every
    single step after the fact - a stray contact-force blip or a frame of
    jitter should not un-detect a placement that already happened.

    Instead, once cube_placed_on_plate() first reports "placed" for an env,
    this just counts hold_steps simulation steps (a grace period so the
    episode doesn't end mid-drop, and so you have time to move the arm back
    to a neutral pose before the episode resets) and then terminates - giving
    you a clean "placed -> hold -> reset" behavior. Every step during the
    hold is still recorded by the teleop script, so moving back to neutral
    during this window becomes part of the saved episode.

    Args:
        hold_steps: Number of steps to wait after the first detected placement
            before terminating the episode. With sim.dt=1/120 and
            decimation=2 (i.e. ~60 control steps/sec), 300 steps ~= 5 seconds.
            Tune to your own step rate if you want a different delay. NOTE:
            make sure episode_length_s (and any time_out termination) leaves
            enough headroom for warmup_steps + however long it actually takes
            you to place the cube + hold_steps - otherwise a time_out
            truncation can fire first and the episode gets discarded instead
            of saved.

    Returns:
        Boolean tensor of shape (num_envs,) for termination.
    """
    num_envs = env.num_envs
    device = env.device

    result = cube_placed_on_plate(
        env=env,
        contact_sensor_cfg=contact_sensor_cfg,
        cubes=cubes,
        plate_name=plate_name,
        warmup_steps=warmup_steps,
        force_threshold=force_threshold,
        plate_local_x_min=plate_local_x_min,
        plate_local_x_max=plate_local_x_max,
        plate_local_y_min=plate_local_y_min,
        plate_local_y_max=plate_local_y_max,
        plate_local_z_max=plate_local_z_max,
        settle_lin_vel_threshold=settle_lin_vel_threshold,
    )
    trigger = result.squeeze(-1).bool()

    if not hasattr(env, "_plate_success_counter"):
        env._plate_success_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        env._plate_hold_active = torch.zeros(num_envs, dtype=torch.bool, device=device)

    just_reset = env.episode_length_buf <= 1
    env._plate_success_counter[just_reset] = 0
    env._plate_hold_active[just_reset] = False

    # Start (or continue) the hold countdown for any env where placement has
    # been detected. Since `trigger` is sticky, once started this never turns
    # back off except on episode reset above.
    newly_triggered = trigger & (~env._plate_hold_active)
    if newly_triggered.any():
        env._plate_hold_active[newly_triggered] = True
        env._plate_success_counter[newly_triggered] = 0

    env._plate_success_counter = torch.where(
        env._plate_hold_active,
        env._plate_success_counter + 1,
        env._plate_success_counter,
    )

    confirmed = env._plate_hold_active & (env._plate_success_counter >= hold_steps)

    return confirmed