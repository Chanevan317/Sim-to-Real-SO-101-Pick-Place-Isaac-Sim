# Step 3: Convert the recordings

LeRobot cannot read the `.h5` files directly. This step turns them into a **LeRobot dataset**: a
folder with the joint data and the camera pictures saved as videos.

## Run

```bash
conda activate lerobot_env
cd ~/Documents/chan_evan/Sim-to-Real-SO-101-Pick-Place-Isaac-Sim/lerobot
python convert_sim_data.py
```

This reads every file in `source/sim_to_real_so101/raw_teleop_data/` and writes the dataset to
`lerobot/lerobot_dataset/`.

For a quick test with only 5 recordings:

```bash
python convert_sim_data.py --max_episodes 5 --output_dir ./lerobot_dataset_test
```

## Options

| Option | Default | What it does |
|---|---|---|
| `--raw_dir` | `../source/sim_to_real_so101/raw_teleop_data` | Folder with the `.h5` files |
| `--output_dir` | `./lerobot_dataset` | Where the dataset is written. It must not exist yet |
| `--task` | `pick up the cube and place it on the plate` | Sentence that describes the task. GR00T reads it |
| `--max_episodes` | all | Use only the first N recordings |
| `--fps` | read from the recordings (60) | Frames per second of the dataset. Old recordings without this info use 30 |

The script stops if the folder mixes old recordings (made before the October 2026 fix) with new
ones, because their `action` means different things. Keep them in separate folders.

If the output folder already exists, the script stops. To start over, delete the folder:

```bash
rm -rf ./lerobot_dataset
```

## Using the dataset

The dataset is ready for both ACT and GR00T: it already has the joint names and the task
sentence that GR00T needs. Train with the default folder, or pass another one:
- ACT: `--dataset.root=./lerobot_dataset`
- GR00T: `DATASET_ROOT=../lerobot_dataset ./train_groot.sh`

To train on another computer, copy the whole dataset folder there.

A dataset cannot be extended. If you record more demos later, convert **all** the recordings
again into a new folder (for example `--output_dir ./lerobot_dataset_100ep`).

Only datasets made with an old version of this script need the
[one-time fix](4b_train_groot.md#1-fix-an-old-dataset-only-once) before GR00T training.

Next: [Train ACT](4a_train_act.md) or [Train GR00T](4b_train_groot.md)
