import argparse
import os
import glob
import re

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO-101 teleoperation via physical leader arm.")
parser.add_argument("--task", type=str, required=True, help="Name of the registered task/env.")
parser.add_argument("--leader_port", type=str, default="/dev/ttyACM0", help="Serial port for the physical leader arm.")
parser.add_argument("--output_dir", type=str, default="./raw_teleop_data", help="Directory to save HDF5 episodes.")
parser.add_argument("--task_name", type=str, default="teleop_pick_place", help="Task name tag stored in each episode.")
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
from sim_to_real_so101.utils.joint_conversion import make_degree_converters
from sim_to_real_so101.utils.so101_ik_solver import SO101IKSolver


class RawHdf5Recorder:

    def __init__(self, output_dir, task_name, fps):
        self.output_dir = output_dir
        self.task_name = task_name
        self.fps = fps
        os.makedirs(self.output_dir, exist_ok=True)
        self.reset_buffers()
        
        # --- NEW: Auto-resume episode indexing ---
        existing_files = glob.glob(os.path.join(self.output_dir, "episode_*.h5"))
        max_idx = -1
        for f in existing_files:
            # Extract the integer from filenames like "episode_0042.h5"
            match = re.search(r"episode_(\d+)\.h5", os.path.basename(f))
            if match:
                max_idx = max(max_idx, int(match.group(1)))
        
        # Start at the next available index
        self.episode_idx = max_idx + 1
        print(f"[INFO]: Data directory found. Starting new recordings at episode_{self.episode_idx:04d}.h5")

    def reset_buffers(self):
        self.actions = []
        self.states = []
        self.images = {}
        self.initial_poses = {}

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
        with h5py.File(file_path, "w") as f:
            f.attrs["task_name"] = self.task_name
            f.attrs["fps"] = self.fps
            # action = leader arm command (joint targets), not the measured follower position
            f.attrs["action_source"] = "leader_command"
            f.create_dataset("action", data=np.array(self.actions, dtype=np.float32))
            f.create_dataset("observation.state", data=np.array(self.states, dtype=np.float32))
            for cam, img_list in self.images.items():
                f.create_dataset(
                    f"observation.images.{cam}", 
                    data=np.array(img_list, dtype=np.uint8),
                    compression="gzip",
                    compression_opts=4,  # Level 4 balances compression speed and file reduction perfectly
                    chunks=True          # Required for compression to work efficiently
                )
            for name, pose in self.initial_poses.items():
                f.create_dataset(f"initial_pose.{name}", data=pose.astype(np.float32))
        print(f"\n[INFO]: Saved teleop demonstration to {file_path} ({len(self.actions)} frames)")
        
        # Increment index ONLY after a successful save
        self.episode_idx += 1
        self.reset_buffers()

    def discard_episode(self):
        print(f"\n[INFO]: Discarded episode buffer ({len(self.actions)} frames dropped).")
        # Notice we do NOT increment self.episode_idx here
        self.reset_buffers()


def build_leader_arm(args_cli):
    """
    Direct Feetech servo communication using official scservo_sdk PacketHandler.
    Includes LeRobot's exact calibration logic (offsets and direction inversion).
    """
    try:
        from scservo_sdk import PortHandler, PacketHandler, COMM_SUCCESS
    except ImportError as e:
        raise ImportError(
            "Please install the correct Feetech SDK: `pip install feetech-servo-sdk`"
        ) from e

    port_handler = PortHandler(args_cli.leader_port)
    packet_handler = PacketHandler(0)  # Protocol version 0 for STS series servos

    if not port_handler.openPort():
        raise RuntimeError(f"Failed to open serial port {args_cli.leader_port}")

    if not port_handler.setBaudRate(1000000):
        raise RuntimeError("Failed to change the baudrate to 1000000 bps.")

    # -------------------------------------------------------------------------
    # LEROBOT-STYLE CALIBRATION CONFIGURATION
    # Format: motor_id: (homing_offset_degrees, direction_multiplier)
    # -------------------------------------------------------------------------
    CALIBRATION = {
        1: {"offset": 0.0,  "direction": 1},   # Base Rotation
        2: {"offset": 0.0,  "direction": 1},   # Pitch
        3: {"offset": 0.0,  "direction": 1},   # Elbow
        4: {"offset": 0.0,  "direction": 1},   # Wrist_Pitch
        5: {"offset": 0.0,  "direction": 1},   # Wrist_Roll
        6: {"offset": 0.0,  "direction": 1},   # Jaw / Gripper
    }

    motor_ids = [1, 2, 3, 4, 5, 6]
    ADDR_STS_PRESENT_POSITION = 56

    class LeaderArmReader:
        # def read_positions_deg(self):
        #     raw_degrees = []
        #     for motor_id in motor_ids:
        #         pos, comm_result, error = packet_handler.read2ByteTxRx(
        #             port_handler, motor_id, ADDR_STS_PRESENT_POSITION
        #         )

        #         if comm_result == COMM_SUCCESS:
        #             # 1. Convert raw 12-bit encoder ticks to raw center-offset degrees
        #             raw_deg = (pos / 4096.0) * 360.0 - 180.0

        #             # 2. Apply LeRobot Calibration: (Raw Angle - Homing Offset) * Direction
        #             cfg = CALIBRATION[motor_id]
        #             calibrated_deg = (raw_deg - cfg["offset"]) * cfg["direction"]

        #             raw_degrees.append(calibrated_deg)
        #         else:
        #             raw_degrees.append(0.0)

        #     return np.array(raw_degrees, dtype=np.float32)

        def __init__(self):
            self._last_raw_deg_wrist_roll = None
            self._unwrap_offset_wrist_roll = 0.0

        def read_positions_deg(self):
            raw_degrees = []
            for motor_id in motor_ids:
                pos, comm_result, error = packet_handler.read2ByteTxRx(
                    port_handler, motor_id, ADDR_STS_PRESENT_POSITION
                )

                if comm_result == COMM_SUCCESS:
                    raw_deg = (pos / 4096.0) * 360.0 - 180.0

                    if motor_id == 5:
                        # Wrist_Roll only: unwrap the -180/+180 encoder seam.
                        last = self._last_raw_deg_wrist_roll
                        if last is not None:
                            delta = raw_deg - last
                            if delta > 180.0:
                                self._unwrap_offset_wrist_roll -= 360.0
                            elif delta < -180.0:
                                self._unwrap_offset_wrist_roll += 360.0
                        self._last_raw_deg_wrist_roll = raw_deg
                        raw_deg = raw_deg + self._unwrap_offset_wrist_roll

                    cfg = CALIBRATION[motor_id]
                    calibrated_deg = (raw_deg - cfg["offset"]) * cfg["direction"]
                    raw_degrees.append(calibrated_deg)
                else:
                    raw_degrees.append(0.0)

            return np.array(raw_degrees, dtype=np.float32)

    return LeaderArmReader()


class EpisodeKeys:
    """Listens for key presses in the Isaac Sim window (the window must have focus)."""

    def __init__(self):
        import carb
        import omni.appwindow

        self._carb = carb
        self._pressed = set()
        self._input = carb.input.acquire_input_interface()
        keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(keyboard, self._on_event)

    def _on_event(self, event, *args, **kwargs):
        if event.type == self._carb.input.KeyboardEventType.KEY_PRESS:
            self._pressed.add(event.input.name)
        return True

    def pop(self, key):
        """True once per press of `key`."""
        if key in self._pressed:
            self._pressed.discard(key)
            return True
        return False


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    scene = env.unwrapped.scene
    obs, _ = env.reset()

    sim_radians_to_real_degrees, real_degrees_to_sim_radians = make_degree_converters(env.unwrapped.device)

    camera_names = [obj.replace("camera_", "") for obj in scene.keys() if obj.startswith("camera_")]
    print(f"[INFO]: Dynamic Camera Tracking online for: {camera_names}")

    fps = round(1.0 / env.unwrapped.step_dt)
    recorder = RawHdf5Recorder(args_cli.output_dir, args_cli.task_name, fps)
    recorder.start_episode(scene)

    try:
        leader = build_leader_arm(args_cli)
    except Exception as e:
        print(f"[ERROR]: No 'leader_arm' device found at {args_cli.leader_port} ({e}). Exiting.")
        env.close()
        return

    keys = EpisodeKeys()
    print(f"[INFO]: Recording at {fps} Hz. Episodes are saved automatically when the cube is placed.")
    print("[INFO]: Keys (click the Isaac Sim window first): N = save now, X = discard and restart. "
          "Ctrl+C in the terminal to quit.")

    robot = scene["robot"]
    arm_joint_ids, _ = robot.find_joints(SO101IKSolver.ARM_JOINT_NAMES)
    gripper_joint_id, _ = robot.find_joints([SO101IKSolver.GRIPPER_JOINT_NAME])
    gripper_joint_id = gripper_joint_id[0]
    joint_limits = robot.data.soft_joint_pos_limits[0]

    def reset_episode():
        new_obs, _ = env.reset()
        recorder.start_episode(scene)
        return new_obs

    while simulation_app.is_running():
        with torch.inference_mode():
            leader_deg = leader.read_positions_deg()

            # --- REAL-TIME TELEMETRY PRINTING ---
            # Formats the current positions of all 6 joints to 1 decimal place.
            # \r overwrites the line in real-time without scrolling your screen away.
            telemetry_str = (
                f"Rot: {leader_deg[0]:6.1f}° | "
                f"Pit: {leader_deg[1]:6.1f}° | "
                f"Elb: {leader_deg[2]:6.1f}° | "
                f"W_Pit: {leader_deg[3]:6.1f}° | "
                f"W_Roll: {leader_deg[4]:6.1f}° | "
                f"Jaw: {leader_deg[5]:6.1f}°"
            )
            # print(f"\r[Leader Telemetry] {telemetry_str}", end="", flush=True)

            leader_deg_t = torch.tensor(leader_deg, device=env.unwrapped.device)
            leader_rad = real_degrees_to_sim_radians(leader_deg_t)
            joint_targets_rad = torch.zeros_like(robot.data.joint_pos[0])
            joint_targets_rad[arm_joint_ids] = leader_rad[:5]
            joint_targets_rad[gripper_joint_id] = leader_rad[5]
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
                obs = reset_episode()
                continue

            if keys.pop("X"):
                recorder.discard_episode()
                obs = reset_episode()
                continue

            if terminations.any():
                recorder.save_episode()
                obs = reset_episode()
                continue

            if truncations.any():
                recorder.discard_episode()
                obs = reset_episode()
                continue

    # Clear telemetry trailing output on exit
    print()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
