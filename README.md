# SO-101 Robot Arm: Pick and Place in Isaac Sim

This project teaches an SO-101 robot arm to **pick up a cube and put it on a plate**.

It has four stages:

1. **Record:** you move a real "leader" arm with your hand, and the robot copies you inside the
   Isaac Sim simulator. Each try is saved as a recording.
2. **Convert:** the recordings are changed into the format that LeRobot (the training tool) reads.
3. **Train:** a model learns from the recordings. You can use **ACT** (small and fast) or
   **GR00T** (a large pretrained model from NVIDIA).
4. **Test:** the trained model controls the robot in the simulator, and you see how often it
   succeeds.

## Start here

Follow the guides in order:

| Step | Guide |
|---|---|
| 0 | [How the project works](docs/0_how_it_works.md) |
| 1 | [Setup (do this once)](docs/1_setup.md) |
| 2 | [Record demos](docs/2_record_demos.md) |
| 2b | [SO-Arm Data Studio: record, review and convert in one app](docs/2b_data_studio.md) |
| 3 | [Convert the recordings](docs/3_convert_dataset.md) |
| 4a | [Train ACT](docs/4a_train_act.md) |
| 4b | [Train GR00T](docs/4b_train_groot.md) |
| 5 | [Test the model in the simulator](docs/5_test_model.md) |
