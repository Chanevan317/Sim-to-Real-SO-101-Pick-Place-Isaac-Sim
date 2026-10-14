"""Calibrate the leader arm visually in the sim.

For each chosen joint the sim robot moves that joint to its middle, min and max angle.
Match each pose with the leader arm and press N. The script fits a line from the leader
encoder angle to the sim angle, then shows a live preview: the sim follows the leader.
In the preview, N saves the calibration and X starts over. Ctrl+C quits without saving.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Calibrate the SO-101 leader arm in the sim.")
parser.add_argument("--task", type=str, default="Lerobot-So101-Teleop-Cubes-To-Plate", help="Scene to show the robot in.")
parser.add_argument("--leader_port", type=str, default="/dev/ttyACM0", help="Serial port for the physical leader arm.")
parser.add_argument("--joints", type=str, default="Wrist_Roll,Jaw",
                    help="Comma-separated joints to calibrate, or 'all'.")
parser.add_argument("--output", type=str, default=None,
                    help="Calibration file (default: source/sim_to_real_so101/leader_calibration.json).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports follow after app launch."""

import math

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks  # noqa: F401
from sim_to_real_so101.utils.joint_conversion import make_degree_converters
from sim_to_real_so101.utils.leader_arm import (
    DEFAULT_CALIBRATION_PATH,
    JOINT_NAMES,
    LeaderArm,
    fit_joint,
    leader_to_sim_radians,
    load_calibration,
    save_calibration,
)
from sim_to_real_so101.utils.sim_keys import SimKeys

STEP_NAMES = ["MIDDLE", "MIN", "MAX"]
# Readings averaged per captured pose, to smooth out encoder noise.
SAMPLES_PER_CAPTURE = 20


def mean_angle_deg(samples):
    """Average of angles in degrees that works across the -180/+180 seam."""
    rad = np.radians(samples)
    return math.degrees(math.atan2(np.sin(rad).mean(), np.cos(rad).mean()))


def main():
    joints = JOINT_NAMES if args_cli.joints == "all" else [j.strip() for j in args_cli.joints.split(",")]
    for j in joints:
        if j not in JOINT_NAMES:
            raise ValueError(f"Unknown joint {j!r}. Choose from {JOINT_NAMES} or 'all'.")
    output = args_cli.output or DEFAULT_CALIBRATION_PATH

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    device = env.unwrapped.device
    _, real_degrees_to_sim_radians = make_degree_converters(device)

    robot = env.unwrapped.scene["robot"]
    joint_ids = [robot.joint_names.index(n) for n in JOINT_NAMES]
    limits_deg = torch.rad2deg(robot.data.soft_joint_pos_limits[0, joint_ids]).cpu().numpy()
    home_rad = robot.data.default_joint_pos[0, joint_ids].clone()

    try:
        leader = LeaderArm(args_cli.leader_port)
    except Exception as e:
        print(f"[ERROR]: No leader arm at {args_cli.leader_port} ({e}). Exiting.")
        env.close()
        return

    keys = SimKeys()
    calibration = load_calibration(output)

    def poses():
        """(joint, step name, sim angle in degrees) for every pose to capture."""
        for j in joints:
            lo, hi = limits_deg[JOINT_NAMES.index(j)]
            for step, deg in zip(STEP_NAMES, [(lo + hi) / 2, lo, hi]):
                yield j, step, float(deg)

    def announce(pose):
        j, step, deg = pose
        print(f"\n[CALIBRATE] {j} {step} ({deg:.1f} deg): move the leader so it matches the sim, "
              "then press N in the sim window.")

    print("[INFO]: Click the Isaac Sim window first, so it receives the key presses.")
    pending = list(poses())
    announce(pending[0])
    captured = {j: ([], []) for j in joints}  # joint -> (raw angles, sim angles)
    samples = []
    preview = False

    while simulation_app.is_running():
        with torch.inference_mode():
            raw = leader.read_raw_deg()

            if not preview:
                j, step, deg = pending[0]
                target = home_rad.clone()
                target[JOINT_NAMES.index(j)] = math.radians(deg)
                samples = (samples + [raw[JOINT_NAMES.index(j)]])[-SAMPLES_PER_CAPTURE:]
                if len(samples) == SAMPLES_PER_CAPTURE and keys.pop("N"):
                    captured[j][0].append(mean_angle_deg(samples))
                    captured[j][1].append(deg)
                    print(f"[CALIBRATE] {j} {step}: leader reads {captured[j][0][-1]:.1f} deg")
                    pending.pop(0)
                    samples = []
                    if pending:
                        announce(pending[0])
                    else:
                        for name, (raw_pts, sim_pts) in captured.items():
                            calibration[name] = fit_joint(raw_pts, sim_pts)
                            c = calibration[name]
                            print(f"[RESULT] {name}: scale {c['scale']:+.3f}, offset {c['offset_deg']:.1f} deg, "
                                  f"fit error {c['max_error_deg']:.1f} deg")
                            if name != "Jaw" and not 0.7 < abs(c["scale"]) < 1.3:
                                print(f"[WARNING]: {name} scale should be close to +-1 (the leader and sim "
                                      "joints have the same geometry). Check the poses and redo with X.")
                        print("\n[PREVIEW] The sim now follows the leader with the new calibration. "
                              "N = save, X = start over, Ctrl+C = quit without saving.")
                        preview = True
            else:
                target = leader_to_sim_radians(raw, calibration, real_degrees_to_sim_radians, device)
                target = torch.clamp(target, robot.data.soft_joint_pos_limits[0, joint_ids, 0],
                                     robot.data.soft_joint_pos_limits[0, joint_ids, 1])
                if keys.pop("N"):
                    save_calibration(calibration, output)
                    print(f"[INFO]: Saved calibration to {output}")
                    break
                if keys.pop("X"):
                    calibration = load_calibration(output)
                    captured = {j: ([], []) for j in joints}
                    pending = list(poses())
                    preview = False
                    print("\n[CALIBRATE] Starting over.")
                    announce(pending[0])

            env.step(target.unsqueeze(0))

    leader.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
