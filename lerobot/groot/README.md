# GR00T

Files for training and testing NVIDIA's GR00T N1.7 model. Use the `lerobot_env` conda environment.

| File | What it does |
|---|---|
| `fix_dataset_metadata.py` | Adds joint names and a task sentence to an old dataset (run once) |
| `train_groot.sh` | Starts GR00T training |
| `groot_policy_server.py` | Runs a trained GR00T model for testing |

Guides:
- [Train GR00T](../../docs/4b_train_groot.md)
- [Test the model](../../docs/5_test_model.md)
