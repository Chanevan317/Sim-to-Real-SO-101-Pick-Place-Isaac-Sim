# How the project works

## The big picture

```
You move the real leader arm
        │
        ▼
Robot copies you in Isaac Sim  ──►  recordings (.h5 files)
                                          │
                                          ▼
                                 convert to LeRobot format
                                          │
                                          ▼
                                 train a model (ACT or GR00T)
                                          │
                                          ▼
                     model controls the robot in Isaac Sim  ──►  success rate
```

## Two separate Python environments

The simulator and the training tool need different package versions, so they live in two
separate conda environments.

| Environment | What runs in it |
|---|---|
| `env_isaaclab` | Isaac Sim: recording demos and testing models |
| `lerobot_env` | LeRobot: converting data, training, running the model |

Testing needs **both at once**:
- In one terminal, the model runs in `lerobot_env`. We call this the **policy server**.
- In another terminal, the simulator runs in `env_isaaclab`.
- The two talk to each other over a local network connection.

## Folders

| Folder | What is inside |
|---|---|
| `source/sim_to_real_so101/` | Everything for the simulator: robot model, scene, recording and testing scripts |
| `lerobot/` | Data conversion, the ACT model server |
| `lerobot/groot/` | GR00T training and the GR00T model server |
| `lerobot/review/` | Web page to review the recordings before converting them |
| `studio/` | SO-Arm Data Studio: desktop app for recording, review and conversion |
| `docs/` | These guides |

Inside `source/sim_to_real_so101/`:

| Folder | What is inside |
|---|---|
| `assets/` | 3D files: the robot, cube, plate, light box and sky images |
| `tasks/` | Scene setups. Each one has a name you pass with `--task` |
| `mdp/` | Small helper functions: random changes at the start of each try, and checks like "is the cube on the plate?" |
| `scripts/` | The scripts you run |
| `utils/` | Helper code used by the scripts |

## Scenes (`--task` names)

| Name | What it is |
|---|---|
| `Lerobot-So101-Teleop-Cubes-To-Plate` | The cube-and-plate scene |
| `Lerobot-So101-Teleop-Cubes-To-Plate-DR` | Same scene, but colours, lights and camera positions change randomly every try |
| `Lerobot-So101-Teleop-Cubes-To-Plate-DR-Eval` | Use this one for testing a model |

"DR" means *domain randomization*: the scene looks a bit different every time. This helps the
model work on the real robot later, because it learns not to depend on one exact look.

A try counts as a **success** when the cube stays on the plate for about 3 seconds.

## Joint values

The robot has 6 joints:

1. base rotation
2. shoulder
3. elbow
4. wrist up/down
5. wrist roll
6. gripper

Joint values are saved in LeRobot's units, the same as the real arm:
- the first five joints go from **−100 to 100**;
- the gripper goes from **0 to 100**.
