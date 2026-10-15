"""
convert_sim_data.py

Converts simulation-collected HDF5 episodes (SO-101 arm) into LeRobotDataset
format for training with the lerobot framework.

Expected input HDF5 schema (per episode file):
    action                           shape=(T, 6) dtype=float32
    observation.images.ego           shape=(T, 480, 640, 3) dtype=uint8
    observation.images.external_D455 shape=(T, 480, 640, 3) dtype=uint8
    observation.state                shape=(T, 6) dtype=float32

Input path (relative to this script):
    ../source/sim_to_real_so101/episode_xxxx.h5   (xxxx = 0000-9999)

Output:
    ./lerobot_dataset/   (LeRobotDataset saved locally next to this script)
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = (SCRIPT_DIR / "../source/sim_to_real_so101/raw_teleop_data/").resolve()
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "lerobot_dataset"

STATE_DIM = 6
ACTION_DIM = 6
IMAGE_SHAPE = (480, 640, 3)

# Per-dimension names for state/action, in teleop_so101.py order
# (Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw). GR00T needs them to
# keep the gripper absolute when training with relative actions.
# Keep in sync with JOINT_NAMES in groot/fix_dataset_metadata.py.
JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

IMAGE_KEYS = {
    "observation.images.ego": "observation.images.ego",
    "observation.images.external_D455": "observation.images.external_D455",
}


def build_features():
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": JOINT_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": JOINT_NAMES,
        },
    }
    for feat_name in IMAGE_KEYS:
        features[feat_name] = {
            "dtype": "video",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channel"],
        }
    return features


def find_episode_files(raw_dir: Path):
    files = sorted(raw_dir.glob("episode_*.h5"))
    if not files:
        raise FileNotFoundError(
            f"No files matching 'episode_*.h5' found in {raw_dir}"
        )
    return files


def convert(raw_dir: Path, output_dir: Path, repo_id: str, task_description: str,
            fps: int, push_to_hub: bool, max_episodes: int | None):
    episode_files = find_episode_files(raw_dir)
    if max_episodes is not None:
        episode_files = episode_files[:max_episodes]

    print(f"Found {len(episode_files)} episode file(s) in {raw_dir}")

    # Recordings from the current teleop_so101.py store their frame rate and
    # save the leader command as the action. Older ones store neither (their
    # action is the measured follower position). Mixing the two would teach
    # the policy two different meanings of "action", so refuse that.
    sources, file_fps = set(), set()
    for ep_path in episode_files:
        with h5py.File(ep_path, "r") as f:
            sources.add(f.attrs.get("action_source", "follower_position (old recording)"))
            if "fps" in f.attrs:
                file_fps.add(int(f.attrs["fps"]))
    if len(sources) > 1:
        raise ValueError(
            f"{raw_dir} mixes recordings with different action types: {sorted(sources)}. "
            "Put old and new recordings in separate folders and convert them separately."
        )
    print(f"Action type: {sources.pop()}")
    if fps is None:
        if len(file_fps) > 1:
            raise ValueError(f"Recordings have different frame rates {sorted(file_fps)}; pass --fps.")
        fps = file_fps.pop() if file_fps else 30
    print(f"Dataset fps: {fps}")
    print(f"Writing LeRobotDataset to {output_dir}")

    if output_dir.exists():
        has_content = any(output_dir.iterdir())
        if has_content:
            raise FileExistsError(
                f"{output_dir} already exists and is not empty. "
                "LeRobotDataset.create() requires a fresh directory. "
                "Delete it (e.g. `rm -rf "
                f"{output_dir}`) or pass a different --output_dir, then re-run."
            )
        else:
            # Empty leftover dir (e.g. from a previous failed run) -- remove
            # it so LeRobotDataset.create()'s own mkdir(exist_ok=False)
            # succeeds.
            output_dir.rmdir()

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        fps=fps,
        features=build_features(),
        robot_type="so101",
    )

    for i, ep_path in enumerate(episode_files):
        with h5py.File(ep_path, "r") as f:
            # Two-cube recordings store a per-episode instruction (which cube
            # was the target); other recordings fall back to --task as before.
            episode_task = f.attrs.get("instruction", task_description)
            if isinstance(episode_task, bytes):
                episode_task = episode_task.decode("utf-8")
            suffix = f" -- task: {episode_task}" if episode_task != task_description else ""
            print(f"[{i + 1}/{len(episode_files)}] Converting {ep_path.name} ...{suffix}")

            actions = f["action"][:]
            states = f["observation.state"][:]
            images = {
                feat_name: f[hdf5_key][:]
                for feat_name, hdf5_key in IMAGE_KEYS.items()
            }

            n_steps = actions.shape[0]
            assert states.shape[0] == n_steps, (
                f"{ep_path.name}: action steps ({n_steps}) != state steps "
                f"({states.shape[0]})"
            )
            for feat_name, img_arr in images.items():
                assert img_arr.shape[0] == n_steps, (
                    f"{ep_path.name}: {feat_name} steps ({img_arr.shape[0]}) "
                    f"!= action steps ({n_steps})"
                )

            for t in range(n_steps):
                frame = {
                    "observation.state": states[t].astype(np.float32),
                    "action": actions[t].astype(np.float32),
                    "task": episode_task,
                }
                for feat_name in IMAGE_KEYS:
                    frame[feat_name] = images[feat_name][t]
                dataset.add_frame(frame)

        dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub()

    print("Done.")
    print(f"Dataset available locally at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw_dir", type=Path, default=DEFAULT_RAW_DIR,
        help="Folder containing episode_xxxx.h5 files "
             f"(default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Where to save the LeRobotDataset (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--repo_id", type=str, default="local/so101_sim_test",
        help="Dataset repo_id (used as an identifier even for local-only saves)",
    )
    parser.add_argument("--task", type=str, default="pick up the cube and place it on the plate")
    parser.add_argument(
        "--fps", type=int, default=None,
        help="Dataset frame rate (default: read from the recordings, or 30 for old recordings)",
    )
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument(
        "--max_episodes", type=int, default=None,
        help="Optional cap on number of episodes, useful for a quick test run",
    )
    args = parser.parse_args()

    convert(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        task_description=args.task,
        fps=args.fps,
        push_to_hub=args.push_to_hub,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()