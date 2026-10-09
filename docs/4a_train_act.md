# Step 4a: Train ACT

ACT is a small model. It trains from scratch on your recordings and runs fast.

## Run

```bash
conda activate lerobot_env
cd ~/Documents/chan_evan/Sim-to-Real-SO-101-Pick-Place-Isaac-Sim/lerobot

lerobot-train \
  --dataset.repo_id=local/so101_sim_test \
  --dataset.root=./lerobot_dataset \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --steps=500000 \
  --save_freq=5000 \
  --log_freq=200 \
  --num_workers=4 \
  --seed=1000 \
  --job_name=so101_act \
  --output_dir=train/outputs/so101_act \
  --wandb.enable=false
```

These are the same settings used for the earlier ACT model. If your dataset is somewhere else,
change `--dataset.root` to its path.

## Main options

| Option | What it does |
|---|---|
| `--dataset.root` | Folder of the dataset |
| `--steps` | How long to train. More steps take more time |
| `--batch_size` | How many samples per step. Lower it if you get an "out of memory" error |
| `--save_freq` | Save the model every N steps |
| `--output_dir` | Where the saved models go. This folder must not exist yet |
| `--dataset.episodes='[0,1,2]'` | Train on only some recordings |
| `--wandb.enable=true` | Show training graphs on wandb.ai (run `wandb login` first) |

## Where the model is saved

```
lerobot/train/outputs/so101_act/checkpoints/
    005000/pretrained_model/
    010000/pretrained_model/
    ...
    last/  →  the newest one
```

## Training stopped? Continue it

```bash
lerobot-train --config_path=train/outputs/so101_act/checkpoints/last/pretrained_model/train_config.json --resume=true
```

Next: [Test the model in the simulator](5_test_model.md)
