# SO-Arm Data Studio

Desktop app to record, review, convert and send two-cube demonstrations without a terminal.

| File | What it does |
|---|---|
| `install_desktop_icon.sh` | Adds the app to the applications menu and the desktop (once, by the admin) |
| `run_studio.sh` | Starts the app (what the icon runs) |
| `studio.py` | The app's local server |
| `transfer.py` | Sends the dataset to the training computer (SSH or croc) and installs croc |
| `studio_config.json` | Settings: task, leader arm port, round sizes, transfer defaults |
| `web/` | The app's pages |

Guide: [SO-Arm Data Studio](../docs/2b_data_studio.md)
