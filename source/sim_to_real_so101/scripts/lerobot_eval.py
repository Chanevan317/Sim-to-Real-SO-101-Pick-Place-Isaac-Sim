"""
Isaac Lab SO-101 Eval Script — ACT policy via remote socket server.

The ACT policy runs in a separate `lerobot` conda environment (see
policy_server.py) and communicates with this script over a local socket,
so this environment (env_isaaclab) never needs to import lerobot.

.. code-block:: bash

    # terminal 1 (lerobot env):
    conda activate lerobot
    python policy_server.py

    # terminal 2 (env_isaaclab env):
    conda activate env_isaaclab
    ./isaaclab.sh -p eval_act_so101.py --task <task_name> --num_episodes 10

For GR00T, run lerobot/groot/groot_policy_server.py in terminal 1 instead;
this script is unchanged (pass --policy_task if your dataset instruction differs).
"""

import argparse
import random
from tqdm import tqdm

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab SO-101 Eval Client (remote ACT policy server).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=1984)
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--policy_host", type=str, default="127.0.0.1")
parser.add_argument("--policy_port", type=int, default=9999)
parser.add_argument("--policy_task", type=str, default="pick up the cube and place it on the plate",
                    help="Language instruction sent to the policy (used by GR00T, ignored by ACT).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import sim_to_real_so101.tasks  # noqa: F401

from sim_to_real_so101.utils.joint_conversion import make_degree_converters
from sim_to_real_so101.utils.policy_client import PolicyClient


def prep_image(img_hwc_uint8: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (3, H, W) float32 in [0, 1], matching how the
    training dataset's HWC uint8 frames were converted before being fed
    to the model."""
    img = img_hwc_uint8.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed

    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    torch.cuda.manual_seed_all(args_cli.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    sim_radians_to_real_degrees, real_degrees_to_sim_radians = make_degree_converters(env.unwrapped.device)

    cameras = {}
    for obj in env.unwrapped.scene.keys():
        if obj.startswith("camera_"):
            cam_name = obj.replace("camera_", "")
            cameras[cam_name] = True
            print(f"[INFO]: Found Camera: {cam_name}")
    if len(cameras) == 0:
        print("[INFO]: No cameras found - policy requires ego/external_D455 cameras")

    # --- connect to the ACT policy server (running in the `lerobot` env) ---
    policy = PolicyClient(host=args_cli.policy_host, port=args_cli.policy_port)

    obs, _ = env.reset()

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    initial_action = torch.tensor(
        [-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, -0.1465],
        device=env.unwrapped.device,
    )

    step = 0
    num_episodes = 0
    num_successes = 0
    success_rate = 0.0
    pbar = None

    while simulation_app.is_running():
        with torch.inference_mode():

            if step == 0:
                pbar = tqdm(
                    total=env.unwrapped.max_episode_length,
                    desc=f"Rollout (ep {num_episodes + 1}, success: {success_rate:.1f}%)",
                    unit="step",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
                )

            if step < 10:  # warmup, matches your GR00T eval script
                actions[:] = initial_action
            else:
                joint_pos_sim_rad = obs["policy"]["joint_pos_obs"][0]
                state_real_deg = sim_radians_to_real_degrees(joint_pos_sim_rad).cpu().numpy()

                visual_obs = obs["visual"]
                image_ego = prep_image(visual_obs["rgb_ego"][0].cpu().numpy().astype(np.uint8))
                image_external = prep_image(visual_obs["rgb_external_D455"][0].cpu().numpy().astype(np.uint8))

                action_real_deg = policy.get_action(
                    state_real_deg, image_ego, image_external,
                    task=args_cli.policy_task, reset=(step == 10),
                )
                # print(f"[CLIENT] step={step} state_real_deg={state_real_deg} "
                #     f"action_real_deg={action_real_deg}")

                action_real_deg_t = torch.tensor(action_real_deg, device=env.unwrapped.device, dtype=torch.float32)
                actions[:] = real_degrees_to_sim_radians(action_real_deg_t)
                # print(f"[CLIENT] step={step} sim_actions_rad={actions[0].cpu().numpy()}")

            obs, rewards, terminated, truncated, info = env.step(actions)
            step += 1

            if pbar is not None:
                pbar.update(1)

            is_terminated = terminated.item() if terminated.numel() == 1 else terminated.any().item()
            is_truncated = truncated.item() if truncated.numel() == 1 else truncated.any().item()

            if is_terminated or is_truncated:
                if pbar is not None:
                    pbar.close()
                    pbar = None

                num_episodes += 1
                if is_terminated and not is_truncated:
                    num_successes += 1
                success_rate = (num_successes / num_episodes) * 100

                obs, _ = env.reset()
                step = 0
                continue

            if num_episodes >= args_cli.num_episodes:
                if pbar is not None:
                    pbar.close()
                    pbar = None
                print(f"[INFO]: Evaluated {args_cli.num_episodes} episodes")
                print(f"[INFO]: Success Rate: {num_successes}/{args_cli.num_episodes} ({success_rate:.1f}%)")
                env.close()
                simulation_app.close()

    env.close()


if __name__ == "__main__":
    main()
    while True:
        simulation_app.update()
    simulation_app.close()