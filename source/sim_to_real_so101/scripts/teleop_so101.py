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
from sim_to_real_so101.utils.so101_ik_solver import SO101IKSolver


class RawHdf5Recorder:

    def __init__(self, output_dir, task_name):
        self.output_dir = output_dir
        self.task_name = task_name
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
        print(f"\n[INFO]: Saved teleop demonstration to {file_path} ({len(self.actions)} frames)")
        
        # Increment index ONLY after a successful save
        self.episode_idx += 1
        self.reset_buffers()

    def discard_episode(self):
        print(f"\n[INFO]: Discarded episode buffer ({len(self.actions)} frames dropped).")
        # Notice we do NOT increment self.episode_idx here
        self.reset_buffers()


JOINT_MINS_DEG = [-110.0, -100.0, -100.0, -95.0, -160.0, -10.0]
JOINT_MAXS_DEG = [110.0, 100.0, 90.0, 95.0, 160.0, 100.0]


def make_degree_converters(device):
    joint_mins = torch.tensor(JOINT_MINS_DEG, device=device)
    joint_maxs = torch.tensor(JOINT_MAXS_DEG, device=device)

    def sim_radians_to_real_degrees(rad_tensor):
        deg = rad_tensor * 180.0 / torch.pi
        normalized = (deg - joint_mins) / (joint_maxs - joint_mins)
        raw_deg = torch.zeros_like(normalized)
        raw_deg[:-1] = normalized[:-1] * 200.0 - 100.0
        raw_deg[-1] = normalized[-1] * 100.0
        return raw_deg

    def real_degrees_to_sim_radians(raw_deg_tensor):
        """Converts the physical leader's real servo degrees back into
        simulation radians."""
        normalized = torch.zeros_like(raw_deg_tensor)
        normalized[:-1] = (raw_deg_tensor[:-1] + 100.0) / 200.0
        normalized[-1] = raw_deg_tensor[-1] / 100.0
        deg = normalized * (joint_maxs - joint_mins) + joint_mins
        return deg * torch.pi / 180.0

    return sim_radians_to_real_degrees, real_degrees_to_sim_radians


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


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    sim_radians_to_real_degrees, real_degrees_to_sim_radians = make_degree_converters(env.unwrapped.device)

    camera_names = [obj.replace("camera_", "") for obj in env.unwrapped.scene.keys() if obj.startswith("camera_")]
    print(f"[INFO]: Dynamic Camera Tracking online for: {camera_names}")

    recorder = RawHdf5Recorder(args_cli.output_dir, args_cli.task_name)

    try:
        leader = build_leader_arm(args_cli)
    except Exception as e:
        print(f"[ERROR]: No 'leader_arm' device found at {args_cli.leader_port} ({e}). Exiting.")
        env.close()
        return

    print("[INFO]: leader_arm mode active - no keyboard hotkeys available for episode "
          "control from this device. Run this script alongside a terminal you can "
          "Ctrl+C, or extend this script with a second lightweight keyboard listener "
          "if you want N/B/ESC while using the leader arm.")

    # --- episode control hotkeys (unused placeholders - see note above) ---
    should_save = {"flag": False}
    should_discard = {"flag": False}
    should_quit = {"flag": False}

    robot = env.unwrapped.scene["robot"]
    arm_joint_ids, _ = robot.find_joints(SO101IKSolver.ARM_JOINT_NAMES)
    gripper_joint_id, _ = robot.find_joints([SO101IKSolver.GRIPPER_JOINT_NAME])
    gripper_joint_id = gripper_joint_id[0]

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
            joint_targets_rad = real_degrees_to_sim_radians(leader_deg_t)
            sim_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            sim_actions[:, arm_joint_ids] = joint_targets_rad[:5]
            sim_actions[:, gripper_joint_id] = joint_targets_rad[5]

            next_obs, rewards, terminations, truncations, infos = env.step(sim_actions)
            visual_obs = next_obs.get("visual", None)

            joint_pos_sim_rad = robot.data.joint_pos[0]
            real_obs_deg = sim_radians_to_real_degrees(joint_pos_sim_rad)
            real_action_deg = real_obs_deg.clone()

            if visual_obs is not None:
                recorder.push_frame(real_action_deg, real_obs_deg, visual_obs, camera_names)

            if should_save["flag"]:
                recorder.save_episode()
                should_save["flag"] = False
                obs, _ = env.reset()
                continue

            if should_discard["flag"]:
                recorder.discard_episode()
                should_discard["flag"] = False
                obs, _ = env.reset()
                continue

            if should_quit["flag"]:
                break

            if terminations.any():
                recorder.save_episode()
                obs, _ = env.reset()
                continue

            if truncations.any():
                recorder.discard_episode()
                obs, _ = env.reset()
                continue

            obs = next_obs

    # Clear telemetry trailing output on exit
    print()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()