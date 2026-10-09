# Step 2: Record demos

You move the real leader arm, and the robot in the simulator copies you. Each successful try is
saved as one file.

## Run

```bash
conda activate env_isaaclab
cd ~/Documents/chan_evan/Sim-to-Real-SO-101-Pick-Place-Isaac-Sim/source/sim_to_real_so101
python scripts/teleop_so101.py --task Lerobot-So101-Teleop-Cubes-To-Plate-DR --leader_port /dev/ttyACM0
```

Run it from this folder, so the recordings go where the next step expects them:
`source/sim_to_real_so101/raw_teleop_data/`.

## How it works

1. Pick up the cube with the robot and put it on the plate.
2. Keep it there for about **3 seconds**.
3. The try is **saved automatically** as `episode_0000.h5`, `episode_0001.h5`, and so on.
4. The scene resets with a new random look and new cube position, and you can do the next try.

## Keys

Click the Isaac Sim window first, so it receives the key presses.

| Key | What it does |
|---|---|
| **X** | Throw away the current try and start a new one (use it when you drop the cube or make a mistake) |
| **N** | Save the current try now and start a new one |
| **Ctrl+C** (in the terminal) | Quit |

If you run the script again later, it keeps counting from the last file, so nothing is
overwritten. You can also delete a bad `.h5` file by hand.

## Tips for good demos

- **Close the leader gripper firmly** on the cube, all the way. The robot squeezes as hard as
  your command says, and the model learns that.
- **Throw away bad tries with X:** a dropped cube, a bump into the wall, a long pause. The
  model copies everything in the recordings, including mistakes.
- **Move smoothly and at a steady speed.**

## Check your recordings (optional)

This replays recordings in the simulator, without any model: same cube and plate positions, same
moves. If a replay puts the cube on the plate, the recording is good training data.

```bash
python scripts/replay_episode.py --task Lerobot-So101-Teleop-Cubes-To-Plate --episodes "raw_teleop_data/*.h5"
```

At the end it prints how many recordings work, for example `48/50 recordings replay successfully`.
Add `--headless` to run without a window. This only works for recordings made with the current
script; older ones are skipped.

## Options

| Option | Default | What it does |
|---|---|---|
| `--task` | (required) | Scene name. Use `Lerobot-So101-Teleop-Cubes-To-Plate-DR` |
| `--leader_port` | `/dev/ttyACM0` | Port of the leader arm |
| `--output_dir` | `./raw_teleop_data` | Where recordings are saved |

## What is inside each file

| Name | What it is |
|---|---|
| `observation.state` | Where the 6 joints are at each moment |
| `action` | Where **you told** the joints to go (the leader arm), at the same moment |
| `observation.images.ego` | Pictures from the camera on the gripper |
| `observation.images.external_D455` | Pictures from the fixed camera in front |
| `initial_pose.cube_1`, `initial_pose.plate_left` | Where the cube and plate started (used by the replay check) |

The recording runs at **60 frames per second**, and this is saved in the file.

`action` and `observation.state` are usually almost the same, but not when the gripper holds the
cube: the jaw stops at the cube, while your command says "closed". That difference is what makes
the robot squeeze. Recordings made before October 2026 saved the jaw position as the action,
which is why models trained on them hold the cube loosely. Record new demos for the best results.

Next: [Convert the recordings](3_convert_dataset.md)
