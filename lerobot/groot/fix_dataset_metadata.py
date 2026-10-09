"""
fix_dataset_metadata.py

Patches the metadata of an existing LeRobotDataset (created by
convert_sim_data.py) so it is ready for GR00T fine-tuning. Only files under
meta/ are touched -- the parquet frame data and the videos are left as-is, so
no re-conversion is needed.

What it changes:
    1. meta/info.json: adds per-dimension joint names to observation.state and
       action. GR00T's relative-action processor uses these names to keep the
       gripper absolute (--policy.relative_exclude_joints='["gripper"]').
       Without names that option is silently ignored.
    2. meta/tasks.parquet + meta/episodes/*: replaces the task string with a
       real language instruction (GR00T is language-conditioned).

The original meta/ directory is backed up to meta.bak_<timestamp>/ first.

Usage (lerobot env):
    python fix_dataset_metadata.py --dataset_root ~/Documents/chan_evan/lerobot_dataset
    python fix_dataset_metadata.py --dataset_root ... --dry_run
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd

# Same order as the 6-D state/action vectors written by teleop_so101.py
# (Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw), named like LeRobot's
# real SO-101 follower so the dataset lines up with real-robot recordings.
# Keep in sync with JOINT_NAMES in ../convert_sim_data.py.
JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

DEFAULT_TASK = "pick up the cube and place it on the plate"


def patch_info(meta_dir: Path, dry_run: bool):
    info_path = meta_dir / "info.json"
    info = json.loads(info_path.read_text())
    for key in ("observation.state", "action"):
        feat = info["features"][key]
        assert feat["shape"] == [len(JOINT_NAMES)], (
            f"{key} has shape {feat['shape']}, expected [{len(JOINT_NAMES)}]"
        )
        print(f"  {key}.names: {feat.get('names')} -> {JOINT_NAMES}")
        feat["names"] = JOINT_NAMES
    if not dry_run:
        info_path.write_text(json.dumps(info, indent=4))


def patch_tasks(meta_dir: Path, task: str, dry_run: bool):
    tasks_path = meta_dir / "tasks.parquet"
    tasks = pd.read_parquet(tasks_path)
    if len(tasks) != 1:
        raise ValueError(
            f"Expected exactly one task in {tasks_path}, found {len(tasks)}: "
            f"{tasks.index.tolist()}. Edit the tasks manually for multi-task datasets."
        )
    old_task = tasks.index[0]
    print(f"  task: {old_task!r} -> {task!r}")
    tasks.index = pd.Index([task], name=tasks.index.name)

    episode_files = sorted((meta_dir / "episodes").glob("*/*.parquet"))
    episode_tables = []
    for ep_file in episode_files:
        episodes = pd.read_parquet(ep_file)
        episodes["tasks"] = episodes["tasks"].apply(
            lambda ts: [task if t == old_task else t for t in ts]
        )
        episode_tables.append((ep_file, episodes))
    print(f"  updated 'tasks' column in {len(episode_files)} episode metadata file(s)")

    if not dry_run:
        tasks.to_parquet(tasks_path)
        for ep_file, episodes in episode_tables:
            episodes.to_parquet(ep_file, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_root", type=Path, required=True, help="LeRobotDataset root (contains meta/)")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help=f"Language instruction (default: {DEFAULT_TASK!r})")
    parser.add_argument("--dry_run", action="store_true", help="Print the changes without writing anything")
    args = parser.parse_args()

    meta_dir = args.dataset_root.expanduser().resolve() / "meta"
    if not (meta_dir / "info.json").exists():
        raise FileNotFoundError(f"{meta_dir}/info.json not found -- is --dataset_root a LeRobotDataset?")

    if not args.dry_run:
        backup_dir = meta_dir.with_name(f"meta.bak_{time.strftime('%Y%m%d_%H%M%S')}")
        shutil.copytree(meta_dir, backup_dir)
        print(f"Backed up {meta_dir} -> {backup_dir}")

    print("info.json:")
    patch_info(meta_dir, args.dry_run)
    print("tasks:")
    patch_tasks(meta_dir, args.task, args.dry_run)

    print("Dry run, nothing written." if args.dry_run else "Done.")


if __name__ == "__main__":
    main()
