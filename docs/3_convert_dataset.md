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

## Old dataset and new recordings

The existing 300-episode dataset in `lerobot/lerobot_dataset/` was recorded with the old script
(see [Record demos](2_record_demos.md#what-is-inside-each-file)). It still works for training, but
models trained on it hold the cube loosely.

When you record new demos, convert them into a **new folder**, so the old one stays untouched:

```bash
python convert_sim_data.py --output_dir ./lerobot_dataset_v2
```

Then train with that folder:
- ACT: `--dataset.root=./lerobot_dataset_v2`
- GR00T: `DATASET_ROOT=../lerobot_dataset_v2 ./train_groot.sh`

New datasets already contain everything GR00T needs. Only the old dataset needed the
[one-time fix](4b_train_groot.md#1-fix-an-old-dataset-only-once).

Next: [Train ACT](4a_train_act.md) or [Train GR00T](4b_train_groot.md)
