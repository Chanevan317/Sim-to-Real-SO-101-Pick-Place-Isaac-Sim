# Step 5: Test the model in the simulator

Testing needs **two terminals**:
- **Terminal 1** runs the model (the "policy server").
- **Terminal 2** runs the simulator. It sends camera pictures and joint values to the model, and
  the model answers with the next move.

## Terminal 1: start the model

Pick **one** of these two.

**ACT:**
```bash
conda activate lerobot_env
cd ~/Documents/chan_evan/Sim-to-Real-SO-101-Pick-Place-Isaac-Sim/lerobot
python policy_server.py --checkpoint train/outputs/so101_act/checkpoints/last/pretrained_model
```

**GR00T:**
```bash
conda activate lerobot_env
cd ~/Documents/chan_evan/Sim-to-Real-SO-101-Pick-Place-Isaac-Sim/lerobot
python groot/groot_policy_server.py --checkpoint train/outputs/so101_groot/checkpoints/last/pretrained_model
```

Wait until it prints `Policy server listening on 127.0.0.1:9999`.

## Terminal 2: start the simulator

```bash
conda activate env_isaaclab
cd ~/Documents/chan_evan/Sim-to-Real-SO-101-Pick-Place-Isaac-Sim
python source/sim_to_real_so101/scripts/lerobot_eval.py --task Lerobot-So101-Teleop-Cubes-To-Plate-DR-Eval --num_episodes 10
```

The robot tries the task 10 times. After each try it prints the result, and at the end the
success rate, for example:

```
[INFO]: Episode 1: success after 9.4 s
[INFO]: Episode 2: timeout after 30.0 s
...
[INFO]: Success Rate: 7/10 (70.0%)
```

A try is a **success** when the cube stays on the plate for about 3 seconds. If that does not
happen within 30 seconds, the try counts as a **timeout** (a failure) and the next one starts.

## Options

**Simulator (`lerobot_eval.py`):**

| Option | Default | What it does |
|---|---|---|
| `--task` | — | Scene name. Use `Lerobot-So101-Teleop-Cubes-To-Plate-DR-Eval` |
| `--num_episodes` | `5` | Number of tries |
| `--max_episode_s` | `30` | Seconds before a try counts as a failure ("timeout") |
| `--seed` | `1984` | Same seed gives the same random scenes, so you can compare models fairly |
| `--policy_task` | `pick up the cube and place it on the plate` | Task sentence for GR00T. It must match the one in the dataset |
| `--headless` | off | Run without a window (faster) |

**Model server (both ACT and GR00T):**

| Option | Default | What it does |
|---|---|---|
| `--checkpoint` | (required) | Folder of the saved model (`.../pretrained_model`) |
| `--port` | `9999` | If you change it, also give the simulator `--policy_port` with the same number |
| `--execute_steps` | 16 | GR00T only. GR00T plans 16 moves at a time. A smaller number makes it re-plan sooner |

## Good to know

- **Restart the model server before each test.** It stops by itself when the simulator closes.
- **Compare ACT and GR00T fairly:** use the same `--seed` and `--num_episodes` for both.
