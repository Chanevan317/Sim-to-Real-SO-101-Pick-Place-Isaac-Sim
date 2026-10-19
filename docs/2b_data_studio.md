# SO-Arm Data Studio: record, review and convert in one app

SO-Arm Data Studio is a desktop app for collecting two-cube demonstrations without using a terminal.
It guides one person through every step: recording in rounds, checking each episode, turning
the approved episodes into a dataset, and sending it to the training computer. It does the same work as [Record demos](2_record_demos.md)
and [Convert the recordings](3_convert_dataset.md), with the same files.

## For the admin (once per computer)

1. Set up the project as in [Setup](1_setup.md), including access to the leader arm port
   (`sudo usermod -aG dialout $USER`, then log out and back in), so it works after every reboot.
2. Calibrate the leader arm: `python scripts/calibrate_leader.py` (see [Record demos](2_record_demos.md)).
   Operators never do this.
3. Add the app to the menu and the desktop:

   ```bash
   cd ~/Documents/chan_evan/Sim-to-Real-SO-101-Pick-Place-Isaac-Sim/studio
   ./install_desktop_icon.sh
   ```

Settings (task, leader arm port, round sizes, default number of episodes, inactivity time
`idle_timeout_s`) are in `studio/studio_config.json`.

For sending datasets (the **Send** step):

- **SSH:** let this computer log in to the training computer without a password, once:
  `ssh-copy-id user@<training computer>`. The training computer needs `rsync`. You can put the
  default address and folder in `transfer.ssh_host` and `transfer.ssh_dir` of `studio_config.json`,
  or type them once on the Send page: the app remembers the last ones used.
- **croc:** nothing to set up. The operator can install croc from the Send page (no admin password
  needed; it goes to `~/.local/bin`). `transfer.croc_relay` can name your own croc relay; empty
  means croc's public relay (the data is end-to-end encrypted either way).

## For the operator

Open **SO-Arm Data Studio** from the desktop or the applications menu. The bar at the top shows
the steps; a step unlocks when the one before it is done.

1. **Campaign:** every time the app opens, pick a campaign and press **Open campaign** (the one
   used last is already selected), or create one: a name, how many episodes are needed in total,
   and how many per round (25 or 50). **Change campaign** at the top switches later.
2. **Check:** every line must be green. If a line is red, it says what to do.
   Then press **Start test** and move every joint of the leader arm a little, including the
   gripper, until all six are green. Put the arm back in its home pose.
   Press **Continue to recording**.
3. **Record:** read the rules and look at the picture of the **home pose** (the arm folded down
   onto its base). Put the leader arm in that pose and press **Start round**. The simulator opens
   (about one minute), showing only the scene: press **F7** if
   you need its menus. Follow the rules on the right of the app. Each try ends back in the home
   pose, then you press **N** to save: in SO-Arm Data Studio nothing saves by itself. The app counts the saved episodes; when the round is complete, the simulator closes by
   itself and the app opens the review.
   - **Pause round** closes the simulator; saved episodes are kept. Press **Continue round** later
     (even another day) to record the rest of the round, without a new arm test.
   - If the leader arm does not move for 2 minutes, the simulator closes by itself and the round
     is paused the same way, so it does not stay open for hours. The try in progress is dropped.
   - **Drop round** cancels the round. If it already saved episodes, you choose: keep them for
     review, or throw them away (they go to the trash and can still be restored by the admin).
   - The review starts only when the round is complete.
   - While the simulator opens, Ubuntu may say **"Isaac Sim is not responding"**. This is normal:
     click **Wait** (as often as it appears) or leave it, and it goes away. Never click **Force Quit**.
4. **Review:** watch every episode and decide:
   Both cameras are shown side by side.
   - **Approve** (`A`) if it is a clean demonstration;
   - **Flag as bad** (`F`) and pick the reason if not.
   Yellow warnings point at likely problems (for example "still gripping at the end"). When
   every episode has a decision, press **Finish review**: flagged episodes go to the trash, and
   the next round records new ones to replace them.
5. Repeat **Check → Record → Review** until the campaign has all its approved episodes. The app
   gives each round the layouts and cube colours that have the fewest episodes, so the dataset
   stays balanced.
6. **Convert:** press **Convert**. The dataset is written to `lerobot/<dataset folder>`. You can
   close the window while it runs and open the app again later to see the progress.
7. **Send:** sends the dataset to the training computer. Pick one of the two ways:
   - **SSH: same network.** For when this computer can reach the training computer. Type its
     address (`user@address`) and the folder that receives the dataset, press **Test connection**
     (it checks the login, the folder and the free space), then **Send**. Sending again later only
     adds what is missing, so an interrupted transfer just continues.
   - **croc: any network.** For when the training computer is somewhere else; both need internet.
     If croc is missing, press **Install croc**. Press **Send and show the code**: the app shows a
     command with a secret code, for example `CROC_SECRET=4821-maple-otter-quartz croc --yes`.
     On the training computer, open a terminal in the folder that should receive the dataset
     (for example the project's `lerobot` folder) and run that command. The transfer starts
     when the other side connects; keep the app's transfer running until it says **Sent**.
     A new code is made for every transfer.
8. **Train:** not active yet. Start training on the training computer ([Train GR00T](4b_train_groot.md)).

You can stop at any time and continue another day: everything is saved in the campaign folder.

## Where things are

| What | Where |
|---|---|
| Recordings of a campaign | `source/sim_to_real_so101/raw_teleop_data_<campaign>/` |
| Campaign goal and history (rounds, pauses, reviews, what was thrown away and why) | `studio_campaign.json` in that folder |
| Review decisions (separate from the admin's review page) | `studio_review.json` in that folder |
| Episodes thrown away | `_trash/` in that folder |
| Converted dataset | `lerobot/<dataset folder>/` |
| Last SSH address and folder used for sending | `~/.config/so101_studio/state.json` |
| App log (for the admin) | `~/.cache/so101_studio.log` |

The app runs only on this computer (`127.0.0.1:8770`). It stops by itself a few minutes after
its window is closed, but never while a round, a conversion or a transfer is running.
