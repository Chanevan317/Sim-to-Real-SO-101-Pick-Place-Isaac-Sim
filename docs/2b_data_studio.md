# SO-Arm Data Studio: record, review and convert in one app

SO-Arm Data Studio is a desktop app for collecting two-cube demonstrations without using a terminal.
It guides one person through every step: recording in rounds, checking each episode, and turning
the approved episodes into a dataset. It does the same work as [Record demos](2_record_demos.md)
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
7. **Train:** not active yet. Copy the dataset folder to the training computer and start
   training there ([Train GR00T](4b_train_groot.md)).

You can stop at any time and continue another day: everything is saved in the campaign folder.

## Where things are

| What | Where |
|---|---|
| Recordings of a campaign | `source/sim_to_real_so101/raw_teleop_data_<campaign>/` |
| Campaign goal and history (rounds, pauses, reviews, what was thrown away and why) | `studio_campaign.json` in that folder |
| Review decisions (separate from the admin's review page) | `studio_review.json` in that folder |
| Episodes thrown away | `_trash/` in that folder |
| Converted dataset | `lerobot/<dataset folder>/` |
| App log (for the admin) | `~/.cache/so101_studio.log` |

The app runs only on this computer (`127.0.0.1:8770`). It stops by itself a few minutes after
its window is closed, but never while a round or a conversion is running.
