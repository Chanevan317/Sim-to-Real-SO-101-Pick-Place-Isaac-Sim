# Training code (LeRobot)

Converting recordings, training, and running the trained model.
Use the `lerobot_env` conda environment.

| File | What it does | Guide |
|---|---|---|
| `review/` | Web page to watch, check and delete `.h5` recordings | [Record demos](../docs/2_record_demos.md#review-your-recordings-in-the-browser) |
| `convert_sim_data.py` | Turns `.h5` recordings into a LeRobot dataset | [Convert](../docs/3_convert_dataset.md) |
| `policy_server.py` | Runs a trained ACT model for testing | [Test the model](../docs/5_test_model.md) |
| `groot/` | Everything for GR00T | [Train GR00T](../docs/4b_train_groot.md) |

ACT training uses the `lerobot-train` command directly: see [Train ACT](../docs/4a_train_act.md).

The `lerobot_dataset/` and `train/` folders are created here and are not saved in git.
