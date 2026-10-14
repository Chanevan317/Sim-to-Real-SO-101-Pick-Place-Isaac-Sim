# Simulator code (`sim_to_real_so101`)

Everything that runs inside Isaac Sim: the robot, the cube-and-plate scene, and the scripts.
Use the `env_isaaclab` conda environment. Install it once with
`python -m pip install -e source/sim_to_real_so101` (from the project folder).

| Folder | What is inside |
|---|---|
| `assets/` | 3D files: robot, cube, plate, light box, sky images |
| `tasks/` | Scene setups (the `--task` names) |
| `mdp/` | Helpers: random changes per try, "is the cube on the plate?" checks |
| `scripts/` | Scripts you run |
| `utils/` | Helper code used by the scripts |

| Script | What it does | Guide |
|---|---|---|
| `scripts/list_envs.py` | Lists the scene names | [Setup](../../docs/1_setup.md#5-check-that-everything-works) |
| `scripts/zero_agent.py` | Opens a scene, robot stays still | [Setup](../../docs/1_setup.md#5-check-that-everything-works) |
| `scripts/random_agent.py` | Opens a scene, robot moves randomly | [Setup](../../docs/1_setup.md#5-check-that-everything-works) |
| `scripts/calibrate_leader.py` | Line up the leader arm with the simulator (once per arm) | [Record demos](../../docs/2_record_demos.md#calibrate-the-leader-arm-once-per-leader-arm) |
| `scripts/teleop_so101.py` | Record demos with the leader arm | [Record demos](../../docs/2_record_demos.md) |
| `scripts/replay_episode.py` | Replay recordings to check they are good | [Record demos](../../docs/2_record_demos.md#check-your-recordings-optional) |
| `scripts/lerobot_eval.py` | Test a trained model | [Test the model](../../docs/5_test_model.md) |
