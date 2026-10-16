# Step 3: Convert the recordings

LeRobot cannot read the `.h5` files directly. This step turns them into a **LeRobot dataset**: a
folder with the joint data and the camera pictures saved as videos.

## Run

```bash
conda activate lerobot_env
cd ~/Documents/chan_evan/Sim-to-Real-SO-101-Pick-Place-Isaac-Sim/lerobot
python convert_sim_data.py --raw_dir raw_teleop_data_twocubes --output_dir lerobot_dataset_twocubes
```

- `--raw_dir` is the folder with your recordings. Its name is enough: the script looks for it in
  `source/sim_to_real_so101/`, where the recording script saves it. A full path works too.
- `--output_dir` is the name of the new dataset folder. It is created in `lerobot/`. A full path
  works too.

Both are optional. Without `--raw_dir`, the script reads `raw_teleop_data`. Without
`--output_dir`, the dataset folder is named after the recordings folder:

| Recordings (`--raw_dir`) | Dataset (`--output_dir` default) |
|---|---|
| `raw_teleop_data` | `lerobot/lerobot_dataset` |
| `raw_teleop_data_twocubes` | `lerobot/lerobot_dataset_twocubes` |
| any other folder, e.g. `my_demos` | `lerobot/lerobot_dataset_my_demos` |

For a quick test with only 5 recordings:

```bash
python convert_sim_data.py --raw_dir raw_teleop_data_twocubes --max_episodes 5 --output_dir lerobot_dataset_test
```

## Options

| Option | Default | What it does |
|---|---|---|
| `--raw_dir` | `raw_teleop_data` | Folder with the `.h5` files: a name in `source/sim_to_real_so101/`, or a path |
| `--output_dir` | named after `--raw_dir` (see above) | Dataset folder: a name in `lerobot/`, or a path. It must not exist yet |
| `--repo_id` | `local/<output folder name>` | Name of the dataset. Only a label for local use; set it to `<your Hugging Face user>/<name>` to upload with `--push_to_hub` |
| `--task` | `pick up the cube and place it on the plate` | Fallback sentence for recordings that don't carry their own (single-cube task). GR00T reads it. Two-cube recordings store a per-episode sentence ("pick up the red/blue cube...") and use that instead, automatically |
| `--max_episodes` | all | Use only the first N recordings |
| `--fps` | read from the recordings (60) | Frames per second of the dataset. Old recordings without this info use 30 |

The script stops if the folder mixes old recordings (made before the October 2026 fix) with new
ones, because their `action` means different things. Keep them in separate folders.

If the output folder already exists, the script stops. To start over, delete the folder:

```bash
rm -rf lerobot_dataset_twocubes    # from the lerobot/ folder
```

## Using the dataset

The dataset is ready for both ACT and GR00T: it already has the joint names and the task
sentence that GR00T needs. Pass its folder to training:
- ACT: `--dataset.root=./lerobot_dataset_twocubes --dataset.repo_id=local/lerobot_dataset_twocubes`
- GR00T: `DATASET_ROOT=../lerobot_dataset_twocubes ./train_groot.sh`

To train on another computer, copy the whole dataset folder there.

A dataset cannot be extended. If you record more demos later, convert **all** the recordings
again into a new folder (for example `--output_dir lerobot_dataset_twocubes_300ep`).

Only datasets made with an old version of this script need the
[one-time fix](4b_train_groot.md#1-fix-an-old-dataset-only-once) before GR00T training.

Next: [Train ACT](4a_train_act.md) or [Train GR00T](4b_train_groot.md)
