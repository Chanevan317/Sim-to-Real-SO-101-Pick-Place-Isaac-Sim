import argparse
import json
import os
import time
import glob
import re

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO-101 teleoperation via physical leader arm.")
parser.add_argument("--task", type=str, required=True, help="Name of the registered task/env.")
parser.add_argument("--leader_port", type=str, default="/dev/ttyACM0", help="Serial port for the physical leader arm.")
parser.add_argument("--output_dir", type=str, default="./raw_teleop_data", help="Directory to save HDF5 episodes.")
parser.add_argument("--task_name", type=str, default="teleop_pick_place", help="Task name tag stored in each episode.")
parser.add_argument("--max_episodes", type=int, default=0,
                    help="Stop after saving this many episodes in this run (0 = no limit). Used for Data Studio rounds.")
parser.add_argument("--operator", type=str, default="", help="Name of the person recording, stored in each episode.")
parser.add_argument("--studio", action="store_true", help="Print machine-readable progress lines for Data Studio.")
parser.add_argument("--no_auto_save", action="store_true",
                    help="Do not save when the cube rests on the plate: save only with N (for example after "
                         "returning the arm to its home pose).")
parser.add_argument("--viewport_only", action="store_true",
                    help="Hide the menus and panels at start, like pressing F7 (F7 or Esc shows them again).")
parser.add_argument("--idle_timeout", type=float, default=0,
                    help="Close after this many seconds without leader-arm movement (0 = never). "
                         "The try in progress is dropped; saved episodes are kept.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports follow after app launch."""

import gymnasium as gym
import h5py
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks  # noqa: F401
from sim_to_real_so101.utils.episode_balance import EpisodeBalancer
from sim_to_real_so101.utils.h5_compress import TMP_SUFFIX, CompressionQueue
from sim_to_real_so101.utils.joint_conversion import make_degree_converters
from sim_to_real_so101.utils.leader_arm import JOINT_NAMES, LeaderArm, leader_to_sim_radians, load_calibration
from sim_to_real_so101.utils.sim_keys import SimKeys
from sim_to_real_so101.utils.viewport_label import ViewportLabel
from sim_to_real_so101.utils.viewport_view import VIEW_PATH, save_current_view


class RawHdf5Recorder:

    def __init__(self, output_dir, task_name, fps, operator=""):
        self.output_dir = output_dir
        self.task_name = task_name
        self.operator = operator
        self.saved_this_run = 0
        self.fps = fps
        os.makedirs(self.output_dir, exist_ok=True)
        self.reset_buffers()
        
        # --- NEW: Auto-resume episode indexing ---
        # Episodes still waiting for compression (.h5.tmp) count too.
        existing_files = glob.glob(os.path.join(self.output_dir, "episode_*.h5"))
        leftover_tmp = sorted(glob.glob(os.path.join(self.output_dir, f"episode_*.h5{TMP_SUFFIX}")))
        existing_files += leftover_tmp
        max_idx = 0  # an empty folder starts at episode_0001
        for f in existing_files:
            # Extract the integer from filenames like "episode_0042.h5"
            match = re.search(r"episode_(\d+)\.h5", os.path.basename(f))
            if match:
                max_idx = max(max_idx, int(match.group(1)))
        
        # Start at the next available index
        self.episode_idx = max_idx + 1
        print(f"[INFO]: Data directory found. Starting new recordings at episode_{self.episode_idx:04d}.h5")

        # Images are compressed in background processes, so saving does not stall the sim.
        self.compression = CompressionQueue()
        for tmp_path in leftover_tmp:
            print(f"[INFO]: Compressing leftover {tmp_path}")
            self.compression.submit(tmp_path)

    def reset_buffers(self):
        self.actions = []
        self.states = []
        self.images = {}
        self.initial_poses = {}
        # Per-episode language instruction (two-cube task only). Left None
        # for tasks that don't set it, so convert_sim_data.py's --task
        # default keeps working unchanged for single-cube recordings.
        self.instruction = None
        # Scene layout of the episode (two-cube task only), kept for analysis.
        self.layout = None

    def start_episode(self, scene):
        """Remember where every rigid object (cube, plate) starts, relative to
        the env origin, so the episode can be replayed exactly later."""
        origin = scene.env_origins[0]
        for name, obj in scene.rigid_objects.items():
            pose = obj.data.root_state_w[0, :7].clone()
            pose[:3] -= origin
            self.initial_poses[name] = pose.cpu().numpy()

    def push_frame(self, action, state, visual_obs, camera_names):
        self.actions.append(action.cpu().numpy())
        self.states.append(state.cpu().numpy())
        for cam in camera_names:
            if f"rgb_{cam}" in visual_obs:
                if cam not in self.images:
                    self.images[cam] = []
                img_np = visual_obs[f"rgb_{cam}"][0].cpu().numpy().astype(np.uint8)
                self.images[cam].append(img_np)

    def save_episode(self):
        if len(self.actions) == 0:
            print("[INFO]: Nothing to save - episode buffer is empty.")
            return
        file_path = os.path.join(self.output_dir, f"episode_{self.episode_idx:04d}.h5")
        # Written uncompressed first (fast); the compression queue turns it into file_path.
        tmp_path = file_path + TMP_SUFFIX
        with h5py.File(tmp_path, "w") as f:
            f.attrs["task_name"] = self.task_name
            f.attrs["fps"] = self.fps
            if self.operator:
                f.attrs["operator"] = self.operator
            if self.instruction is not None:
                f.attrs["instruction"] = self.instruction
            if self.layout is not None:
                f.attrs["layout"] = self.layout
            # action = leader arm command (joint targets), not the measured follower position
            f.attrs["action_source"] = "leader_command"
            f.create_dataset("action", data=np.array(self.actions, dtype=np.float32))
            f.create_dataset("observation.state", data=np.array(self.states, dtype=np.float32))
            for cam, img_list in self.images.items():
                f.create_dataset(f"observation.images.{cam}", data=np.array(img_list, dtype=np.uint8))
            for name, pose in self.initial_poses.items():
                f.create_dataset(f"initial_pose.{name}", data=pose.astype(np.float32))
        self.compression.submit(tmp_path)
        print(f"\n[INFO]: Saved teleop demonstration to {file_path} ({len(self.actions)} frames, "
              f"{self.compression.pending()} file(s) compressing in the background)")
        
        studio_event("saved", file=os.path.basename(file_path), frames=len(self.actions))
        # Increment index ONLY after a successful save
        self.episode_idx += 1
        self.saved_this_run += 1
        self.reset_buffers()

    def discard_episode(self):
        print(f"\n[INFO]: Discarded episode buffer ({len(self.actions)} frames dropped).")
        studio_event("discarded")
        # Notice we do NOT increment self.episode_idx here
        self.reset_buffers()


def studio_event(event, **fields):
    """One JSON line per event for Data Studio (studio/), which runs this script per round."""
    if args_cli.studio:
        print("[STUDIO] " + json.dumps({"event": event, **fields}), flush=True)


def instruction_for_current_episode(env):
    """Two-cube task: build the per-episode instruction from the target cube
    the reset event just picked. Returns None for tasks that don't set
    target_cube_color (e.g. the single-cube task), so the caller's recorder
    falls back to its session-wide task_name as before."""
    color = getattr(env.unwrapped, "target_cube_color", None)
    if color is None:
        return None
    return f"pick up the {color} cube and place it on the plate"


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    if args_cli.no_auto_save and getattr(env_cfg.terminations, "success", None) is not None:
        env_cfg.terminations.success = None  # episodes then end only with N (or X)
        print("[INFO]: Automatic save is off: press N to save each episode.")
    env = gym.make(args_cli.task, cfg=env_cfg)
    scene = env.unwrapped.scene

    # Two-cube task: keep the recordings balanced over layout x target colour by asking each
    # reset for one of the least-recorded combinations in the output folder.
    reset_term = getattr(env_cfg.events, "reset_cubes_setup", None)
    reset_params = reset_term.params if reset_term is not None else {}
    balancer = None
    if "layouts" in reset_params:
        balancer = EpisodeBalancer(args_cli.output_dir, reset_params["layouts"], reset_params["cube_colors"])
    balance_note = ""

    def request_balanced_episode():
        nonlocal balance_note
        if balancer is None:
            return
        layout, color, counts = balancer.next()
        env.unwrapped.requested_layout = layout
        env.unwrapped.requested_target = balancer.cube_for_color[color]
        balance_note = (f"{layout} / {color}: {counts[(layout, color)]} recorded so far, the fewest. "
                        f"All: {balancer.summary(counts)}")

    request_balanced_episode()
    obs, _ = env.reset()

    sim_radians_to_real_degrees, real_degrees_to_sim_radians = make_degree_converters(env.unwrapped.device)

    camera_names = [obj.replace("camera_", "") for obj in scene.keys() if obj.startswith("camera_")]
    print(f"[INFO]: Dynamic Camera Tracking online for: {camera_names}")

    fps = round(1.0 / env.unwrapped.step_dt)
    recorder = RawHdf5Recorder(args_cli.output_dir, args_cli.task_name, fps, args_cli.operator)
    max_episodes = max(args_cli.max_episodes, 0)
    # Current episode number and (two-cube task) target cube in the viewport. UI only,
    # not seen by the recorded cameras.
    label = ViewportLabel()

    def update_episode_info():
        progress = f"{recorder.saved_this_run + 1} of {max_episodes}" if max_episodes else None
        label.set_episode(recorder.episode_idx, progress)
        recorder.instruction = instruction_for_current_episode(env)
        recorder.layout = getattr(env.unwrapped, "layout_name", None)
        studio_event("episode", index=recorder.episode_idx, layout=recorder.layout,
                     target=getattr(env.unwrapped, "target_cube_color", None),
                     saved=recorder.saved_this_run, max=max_episodes)
        if recorder.instruction is not None:
            print(f"[INFO]: This episode's target: {recorder.instruction}")
            if balance_note:
                print(f"[INFO]: Balancing: {balance_note}")
            color = env.unwrapped.target_cube_color
            label.set_target(f"TARGET: {color.upper()} cube", color)

    update_episode_info()
    recorder.start_episode(scene)

    try:
        leader = LeaderArm(args_cli.leader_port)
    except Exception as e:
        print(f"[ERROR]: No 'leader_arm' device found at {args_cli.leader_port} ({e}). Exiting.")
        env.close()
        return

    calibration = load_calibration()
    print(f"[INFO]: Leader calibration loaded for: {sorted(calibration) or 'no joints'}")
    for name in ("Wrist_Roll", "Jaw"):
        if name not in calibration:
            print(f"[WARNING]: {name} is not calibrated and will not follow the leader correctly. "
                  "Run scripts/calibrate_leader.py first.")

    keys = SimKeys()
    if args_cli.viewport_only and not args_cli.headless:
        try:  # the F7 action of Kit's omni.kit.ui.actions extension
            import omni.kit.actions.core

            omni.kit.actions.core.get_action_registry().get_action("omni.kit.ui.actions", "toggle_ui").execute(True)
        except Exception as e:
            print(f"[WARNING]: Could not hide the menus ({e}). Press F7 to hide them.")
    studio_event("ready")
    print(f"[INFO]: Recording at {fps} Hz. Episodes are saved automatically when the cube is placed.")
    print("[INFO]: Keys (click the Isaac Sim window first): N = save now, X = discard and restart, "
          "V = save the current view as the default. Ctrl+C in the terminal to quit.")

    robot = scene["robot"]
    leader_joint_ids = [robot.joint_names.index(name) for name in JOINT_NAMES]
    joint_limits = robot.data.soft_joint_pos_limits[0]

    def round_complete():
        if max_episodes and recorder.saved_this_run >= max_episodes:
            print(f"\n[INFO]: Round complete: {recorder.saved_this_run} episode(s) saved. Closing.")
            return True
        return False

    # Inactivity: the leader arm counts as moving when any joint changes by more than IDLE_DEG.
    IDLE_DEG = 3.0
    idle_ref = None
    last_move = time.time()

    def idle_too_long(raw_deg):
        nonlocal idle_ref, last_move
        if idle_ref is None or float(np.abs((raw_deg - idle_ref + 180.0) % 360.0 - 180.0).max()) > IDLE_DEG:
            idle_ref, last_move = raw_deg, time.time()
        return args_cli.idle_timeout > 0 and time.time() - last_move > args_cli.idle_timeout

    def reset_episode():
        request_balanced_episode()
        new_obs, _ = env.reset()
        update_episode_info()
        recorder.start_episode(scene)
        return new_obs

    while simulation_app.is_running():
        with torch.inference_mode():
            raw_deg = leader.read_raw_deg()
            if idle_too_long(raw_deg):
                print(f"\n[INFO]: No leader-arm movement for {args_cli.idle_timeout:.0f} s: closing. "
                      "The try in progress is dropped; saved episodes are kept.")
                recorder.discard_episode()
                studio_event("idle_timeout")
                break
            leader_rad = leader_to_sim_radians(raw_deg, calibration, real_degrees_to_sim_radians, env.unwrapped.device)
            joint_targets_rad = torch.zeros_like(robot.data.joint_pos[0])
            joint_targets_rad[leader_joint_ids] = leader_rad
            # The sim cannot go past its joint limits anyway; clamping keeps the
            # recorded action inside the range the robot can actually reach.
            joint_targets_rad = torch.clamp(joint_targets_rad, joint_limits[:, 0], joint_limits[:, 1])
            sim_actions = joint_targets_rad.unsqueeze(0)

            # Record the observation the operator saw *before* this step together
            # with the command sent in response to it. The action is the leader
            # arm command, not where the follower ended up: when the jaw is blocked
            # by the cube, the command stays "closed", which is what produces the
            # grip force. That is the same pairing a policy uses at test time.
            visual_obs = obs.get("visual", None)
            if visual_obs is not None:
                real_obs_deg = sim_radians_to_real_degrees(robot.data.joint_pos[0])
                real_action_deg = sim_radians_to_real_degrees(joint_targets_rad)
                recorder.push_frame(real_action_deg, real_obs_deg, visual_obs, camera_names)

            obs, rewards, terminations, truncations, infos = env.step(sim_actions)

            if keys.pop("N"):
                recorder.save_episode()
                if round_complete():
                    break
                obs = reset_episode()
                continue

            if keys.pop("X"):
                recorder.discard_episode()
                obs = reset_episode()
                continue

            if keys.pop("V"):
                try:
                    view = save_current_view()
                    print(f"\n[INFO]: Saved this view as the default ({VIEW_PATH.name}): {view}")
                except Exception as e:
                    print(f"\n[WARNING]: Could not save the view ({e}).")

            if terminations.any():
                recorder.save_episode()
                if round_complete():
                    break
                obs = reset_episode()
                continue

            if truncations.any():
                recorder.discard_episode()
                obs = reset_episode()
                continue

    print()
    if recorder.compression.pending():
        print(f"[INFO]: Waiting for {recorder.compression.pending()} episode(s) to finish compressing...")
        studio_event("compressing", pending=recorder.compression.pending())
        recorder.compression.wait()
    studio_event("finished", saved=recorder.saved_this_run)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
