# Step 2: Record demos

You move the real leader arm, and the robot in the simulator copies you. Each successful try is
saved as one file.

## Calibrate the leader arm (once per leader arm)

Each leader arm reads its motor angles a little differently. This step teaches the script how
your arm's **wrist roll** and **gripper** line up with the robot in the simulator.

```bash
conda activate env_isaaclab
cd ~/Documents/chan_evan/Sim-to-Real-SO-101-Pick-Place-Isaac-Sim/source/sim_to_real_so101
python scripts/calibrate_leader.py
```

1. Click the Isaac Sim window, so it receives the key presses.
2. The robot in the simulator moves one joint to its **middle**, then its **min**, then its **max**
   position. Each time, move the leader arm to the same pose and press **N**.
3. After the last pose, the robot follows your leader arm. Check that the wrist roll and the
   gripper move the same way as your hand. Press **N** to save, or **X** to start over.

The result is saved in `source/sim_to_real_so101/leader_calibration.json` (not saved in git,
because it belongs to your leader arm). The recording script reads it automatically.
To calibrate all six joints instead of two, add `--joints all`.

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

## Two-cube task (teaching the model to pick the right one)

`Lerobot-So101-Teleop-TwoCubes-To-Plate-DR` puts **two** cubes on the table instead of one: a
**blue** cube_1 and a **red** cube_2, each spawned at a random spot over a wider area. Every
episode, one of them is picked at random as the one to move; the terminal prints which one, for
example:

```
[INFO]: This episode's target: pick up the red cube and place it on the plate
```

The sim window also shows it in the top-right corner, for example **TARGET: RED cube** in red.
This label is only on your screen: the robot's cameras do not see it, so it is not in the
recordings.

Only move **that** cube. The other one is a distractor — leave it alone. This is what teaches the
model to actually look at which cube it's told to grab, instead of always going for "the cube"
(there's only ever one in the single-cube task, so it never has to choose). The episode only
counts as placed when the **named** cube reaches the plate; placing the wrong one does nothing.

## Keys

## Keys

Click the Isaac Sim window first, so it receives the key presses.

| Key | What it does |
|---|---|
| **X** | Throw away the current try and start a new one (use it when you drop the cube or make a mistake) |
| **N** | Save the current try now and start a new one |
| **Ctrl+C** (in the terminal) | Quit |

If you run the script again later, it keeps counting from the last file, so nothing is
overwritten. You can also delete a bad `.h5` file by hand.

Saving takes less than a second. The pictures are compressed afterwards in the background,
while you record the next try. During that time the file is called `episode_0000.h5.tmp`, and
it becomes `episode_0000.h5` when done. No recording is lost when you quit: files that are
being compressed finish in the background, and any `.tmp` files left over (after **Ctrl+C** or a
crash) are compressed the next time you start the script. You can also compress them by hand:

```bash
python -m sim_to_real_so101.utils.h5_compress raw_teleop_data
```

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

(use `--task Lerobot-So101-Teleop-TwoCubes-To-Plate-DR` for two-cube recordings)

At the end it prints how many recordings work, for example `48/50 recordings replay successfully`.
Add `--headless` to run without a window. This only works for recordings made with the current
script; older ones are skipped.

## Options

| Option | Default | What it does |
|---|---|---|
| `--task` | (required) | Scene name. Use `Lerobot-So101-Teleop-Cubes-To-Plate-DR`, or `Lerobot-So101-Teleop-TwoCubes-To-Plate-DR` for the two-cube task |
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
| `instruction` | Two-cube task only: which cube was the target this episode, as the sentence sent to the model |

The recording runs at **60 frames per second**, and this is saved in the file.

`action` and `observation.state` are usually almost the same, but not when the gripper holds the
cube: the jaw stops at the cube, while your command says "closed". That difference is what makes
the robot squeeze. Recordings made before October 2026 saved the jaw position as the action,
which is why models trained on them hold the cube loosely. Record new demos for the best results.

Next: [Convert the recordings](3_convert_dataset.md)
