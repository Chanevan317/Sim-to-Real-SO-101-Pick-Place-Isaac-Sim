"""
Replay recorded teleop episodes in the simulator, without any policy.

Each episode starts with the cube and plate where they were during recording,
then the recorded actions are sent to the robot one by one. If the cube ends up
on the plate, the recording is good training data: a policy that copies it
perfectly would succeed too.

    python scripts/replay_episode.py --task Lerobot-So101-Teleop-Cubes-To-Plate --episodes raw_teleop_data/episode_0000.h5
    python scripts/replay_episode.py --task Lerobot-So101-Teleop-Cubes-To-Plate --episodes "raw_teleop_data/*.h5" --headless
"""

import argparse
import glob

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay recorded SO-101 teleop episodes.")
parser.add_argument("--task", type=str, required=True, help="Name of the registered task/env.")
parser.add_argument("--episodes", type=str, required=True, help="HDF5 file, or a glob pattern in quotes.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports follow after app launch."""

import gymnasium as gym
import h5py
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks  # noqa: F401
from sim_to_real_so101.utils.joint_conversion import make_degree_converters


def place_objects(scene, f):
    """Move every rigid object to the pose stored in the recording."""
    env_ids = torch.tensor([0], device=scene.device)
    origin = scene.env_origins[0]
    for name, obj in scene.rigid_objects.items():
        key = f"initial_pose.{name}"
        if key not in f:
            continue
        pose = torch.tensor(f[key][:], device=scene.device).unsqueeze(0)
        pose[:, :3] += origin
        obj.write_root_pose_to_sim(pose, env_ids=env_ids)
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=scene.device), env_ids=env_ids)


def main():
    files = sorted(glob.glob(args_cli.episodes))
    if not files:
        raise FileNotFoundError(f"No files match {args_cli.episodes}")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    scene = env.unwrapped.scene
    _, real_degrees_to_sim_radians = make_degree_converters(env.unwrapped.device)

    results = []
    for path in files:
        with h5py.File(path, "r") as f:
            if not any(k.startswith("initial_pose.") for k in f.keys()):
                print(f"[SKIP] {path}: no initial object poses (recorded with an older teleop script)")
                continue
            actions = torch.tensor(f["action"][:], device=env.unwrapped.device)
            env.reset()
            place_objects(scene, f)

        success_step = None
        with torch.inference_mode():
            for t, action_deg in enumerate(actions):
                _, _, terminated, truncated, _ = env.step(real_degrees_to_sim_radians(action_deg).unsqueeze(0))
                if terminated.any():
                    success_step = t
                    break
                if truncated.any() or not simulation_app.is_running():
                    break

        ok = success_step is not None
        results.append(ok)
        detail = f"success at step {success_step}" if ok else "cube NOT on the plate at the end"
        print(f"[{'OK' if ok else 'FAIL'}] {path} ({len(actions)} steps): {detail}")
        if not simulation_app.is_running():
            break

    if results:
        print(f"\n[INFO]: {sum(results)}/{len(results)} recordings replay successfully "
              f"({100 * sum(results) / len(results):.0f}%)")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
