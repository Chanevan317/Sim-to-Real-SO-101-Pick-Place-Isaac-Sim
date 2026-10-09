# Step 1: Setup (do this once)

In these guides, the project folder is:

```
~/Documents/chan_evan/Sim-to-Real-SO-101-Pick-Place-Isaac-Sim
```

If yours is somewhere else, change the path in the commands.

You need:
- Isaac Sim + Isaac Lab, installed in a conda environment called `env_isaaclab`;
- LeRobot, installed from source at `~/lerobot`, in a conda environment called `lerobot_env`.

## 1. Download the 3D files

The 3D files (robot, cube, plate, sky images) are stored with Git LFS, so a normal `git clone`
only downloads small placeholder files.

```bash
sudo apt install git-lfs
cd ~/Documents/chan_evan/Sim-to-Real-SO-101-Pick-Place-Isaac-Sim
git lfs install
git lfs pull
```

To check that it worked, run this command:

```bash
ls -lh source/sim_to_real_so101/assets/usd/
```

The `SO-ARM101-USD.usd` file should be about 22 MB. If it is only ~130 bytes, the download did not
work.

## 2. Set up the simulator environment

```bash
conda activate env_isaaclab
pip install feetech-servo-sdk h5py
```

- `feetech-servo-sdk` lets the scripts read the real leader arm.
- `h5py` saves the recordings.

Then install this project as a Python package, so the simulator can find its code:

```bash
cd ~/Documents/chan_evan/Sim-to-Real-SO-101-Pick-Place-Isaac-Sim
python -m pip install -e source/sim_to_real_so101
```

`-e` means "editable": Python uses the files in this folder directly, so your code changes work
right away, without installing again.

## 3. Set up the training environment

```bash
conda activate lerobot_env
pip install h5py
cd ~/lerobot
pip install -e ".[groot]"
```

`.[groot]` adds the extra packages GR00T needs. You can skip that last line if you only train ACT.

### Hugging Face login (needed for GR00T only)

GR00T uses an NVIDIA model (`nvidia/Cosmos-Reason2-2B`) that you must accept a license for
before downloading. It is free, and you get access right away.

1. Create a free account at https://huggingface.co/join
2. Open https://huggingface.co/nvidia/Cosmos-Reason2-2B while logged in, and accept the license.
3. Go to https://huggingface.co/settings/tokens → **Create new token** → type **Read** → copy it.
4. Log in from the terminal and paste the token. Nothing appears while you paste; this is normal.
   ```bash
   conda activate lerobot_env
   hf auth login
   ```
5. Check it worked. This should print your username:
   ```bash
   hf auth whoami
   ```

Without this, GR00T training stops with `You are trying to access a gated repo`.

## 4. Allow access to the leader arm (only for recording)

Plug in the leader arm, then find its port and allow access:

```bash
ls /dev/ttyACM*
sudo chmod 666 /dev/ttyACM0
```

`ls` shows the port name, usually `/dev/ttyACM0`. You have to repeat `chmod` after every reboot. To
make it permanent, run this command once and then log out and back in:

```bash
sudo usermod -aG dialout $USER
```

## 5. Check that everything works

```bash
conda activate env_isaaclab
cd ~/Documents/chan_evan/Sim-to-Real-SO-101-Pick-Place-Isaac-Sim/source/sim_to_real_so101
python scripts/list_envs.py
```

A table with the `Lerobot-So101-...` scene names should appear. To see the scene, run the robot
with zero movement:

```bash
python scripts/zero_agent.py --task Lerobot-So101-Teleop-Cubes-To-Plate-DR --enable_cameras
```

Next: [Record demos](2_record_demos.md)
