# Step 4b: Train GR00T

GR00T N1.7 is a large model from NVIDIA that has already learned from a lot of robot data. You
only teach it your task, so it often needs fewer recordings than ACT. It is slower and needs more
GPU memory.

All GR00T files are in `lerobot/groot/`.

Before you start, make sure you installed the GR00T packages in [Setup](1_setup.md), step 3.

## 1. Fix an old dataset (only once)

Datasets made before the GR00T support was added are missing two things GR00T uses:
- the **names of the joints**;
- a **clear sentence that describes the task**.

This script adds both. It only changes small info files, not your recordings or videos.

```bash
conda activate lerobot_env
cd ~/Documents/chan_evan/Sim-to-Real-SO-101-Pick-Place-Isaac-Sim/lerobot/groot

# see what would change (nothing is written)
python fix_dataset_metadata.py --dataset_root ../lerobot_dataset --dry_run

# apply it
python fix_dataset_metadata.py --dataset_root ../lerobot_dataset
```

Before changing anything, it saves a copy of the old info files in a `meta.bak_<date>` folder
next to them. Datasets made with the current `convert_sim_data.py` do not need this step.

## 2. Train

```bash
cd ~/Documents/chan_evan/Sim-to-Real-SO-101-Pick-Place-Isaac-Sim/lerobot/groot
GPU=1 ./train_groot.sh
```

`GPU=1` uses the second GPU. On this computer, GPU 0 also runs the screen, so it has a little
less free memory.

The first time, it downloads the GR00T model from Hugging Face (about 7 GB), so this takes a
while. You must be logged in to Hugging Face first ([Setup](1_setup.md), step 3).

On one RTX 4080 SUPER (16 GB), training uses about **13.6 GB** and runs about **3.5 steps per
second**. The default 20,000 steps take about **1.6 hours**.

Before a long training, do a quick test with 50 steps (a few minutes):

```bash
GPU=1 STEPS=50 OUTPUT_DIR=../train/outputs/groot_test ./train_groot.sh
rm -rf ../train/outputs/groot_test     # delete the test result afterwards
```

You can change settings by putting them before the command:

```bash
GPU=1 NUM_EPISODES=100 BATCH_SIZE=4 STEPS=30000 ./train_groot.sh
```

| Setting | Default | What it does |
|---|---|---|
| `GPU` | `0` | Which GPU to use: `0` or `1` |
| `NUM_GPUS` | `1` | Train on several GPUs at once. Read "Two GPUs" below first |
| `DATASET_ROOT` | `lerobot/lerobot_dataset` | Folder of the dataset. Only set it if your dataset is somewhere else |
| `NUM_EPISODES` | all | Use only the first N recordings |
| `BATCH_SIZE` | `8` | Samples per step. Lower it if you get "out of memory" |
| `STEPS` | `20000` | How long to train |
| `OUTPUT_DIR` | `lerobot/train/outputs/so101_groot` | Where models are saved. It must not exist yet |
| `LOW_MEMORY` | `1` | Lets GR00T fit in 16 GB. Use `0` only on a big GPU (about 40 GB or more) |

Things to know:
- **Out of memory?** Lower `BATCH_SIZE`, for example `BATCH_SIZE=4`.
- **Fewer recordings do not make training faster.** Training time depends on `STEPS` and
  `BATCH_SIZE`.
- **More options:** you can add any `lerobot-train` option at the end, for example
  `./train_groot.sh --wandb.enable=true`.

## Two GPUs

`NUM_GPUS=2 ./train_groot.sh` trains on both GPUs. Each GPU gets its own copy of the model and
its own `BATCH_SIZE` samples, so each step uses twice as many samples.

**On this computer it is slower, not faster.** GPU 1 sits in a slow motherboard slot (PCIe 3.0,
1 lane, about 0.8 GB/s, while GPU 0 has about 16.6 GB/s). After every step, the two GPUs must
share about 1 GB of results, and that slow slot makes each step take about 2.3 seconds instead
of 0.2 seconds. GPU 0 also gets almost full (15.7 of 16 GB), because it runs the screen too.

So on this computer, use **one GPU** (`GPU=1` or `GPU=0`). Two GPUs only help if both sit in
fast slots (PCIe x8 or x16). Check your motherboard manual and move GPU 1 if a faster slot is free.

To check how your GPUs are connected:

```bash
nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current --format=csv
```

Run this while training runs: when idle, GPUs can report lower numbers to save power.

## What `LOW_MEMORY=1` does

GR00T has about 3.5 billion numbers ("parameters"). NVIDIA's normal training keeps all of them in
high precision and trains about 1.6 billion of them. That needs a GPU with about 40 GB or more.

To fit in 16 GB, `LOW_MEMORY=1` does two things:
1. It keeps the model in **half precision**, which uses half the memory.
2. It **does not train** the biggest part of GR00T: the part that turns its plan into smooth
   movements, called the "diffusion transformer". That part already learned this skill from
   NVIDIA's data.

It still trains the parts that are specific to your robot: how it reads the joint values, how it
outputs moves, and how it connects the camera pictures to moves.

This is a reduced version of NVIDIA's training, so results may be a bit worse than with a big GPU.
If the robot does not learn the task well, you could rent a bigger cloud GPU and train with
`LOW_MEMORY=0`.

## Where the model is saved

```
lerobot/train/outputs/so101_groot/checkpoints/
    005000/pretrained_model/
    ...
    last/  →  the newest one
```

## Training stopped? Continue it

```bash
cd ~/Documents/chan_evan/Sim-to-Real-SO-101-Pick-Place-Isaac-Sim/lerobot
lerobot-train --config_path=train/outputs/so101_groot/checkpoints/last/pretrained_model/train_config.json --resume=true
```

Next: [Test the model in the simulator](5_test_model.md)
