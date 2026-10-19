"""
studio.py

SO-Arm Data Studio: a local app that guides one person through collecting two-cube
demonstrations: campaign (folder, target, round size), checks of the computer
and the leader arm, recording rounds in Isaac Sim, reviewing every episode, and conversion to a
LeRobot dataset. Start it with run_studio.sh or the desktop icon (install_desktop_icon.sh).

The server runs in env_isaaclab (it reads the leader arm for the motor test). It starts
teleop_so101.py for each round and convert_sim_data.py in lerobot_env. It listens on 127.0.0.1
only and stops by itself a few minutes after the app window is closed, never while a job runs.
Admin settings: studio_config.json.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


STUDIO_DIR = Path(__file__).resolve().parent
REPO_DIR = STUDIO_DIR.parent
SIM_DIR = REPO_DIR / "source" / "sim_to_real_so101"
LEROBOT_DIR = REPO_DIR / "lerobot"
WEB_DIR = STUDIO_DIR / "web"

from episodes import Episodes  # noqa: E402  (studio/episodes.py)

CONFIG = json.loads((STUDIO_DIR / "studio_config.json").read_text())
STATE_FILE = Path.home() / ".config" / "so101_studio" / "state.json"  # active campaign
CACHE_DIR = Path.home() / ".cache" / "so101_studio"  # review videos
CAMPAIGN_FILE = "studio_campaign.json"
# Campaign folders are raw_teleop_data_<name>; the plain raw_teleop_data folder (recordings made
# outside the app) is not a campaign.
FOLDER_RE = re.compile(r"^raw_teleop_data_[a-z0-9_]+$")
DATASET_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
STATIC_RE = re.compile(r"^[A-Za-z0-9_.-]+\.(html|css|js|svg|jpg)$")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
LAYOUTS = ["camera_side", "far_side", "plate_middle"]
COLORS = ["blue", "red"]
IDLE_EXIT_S = 180  # stop the server this long after the window stopped sending heartbeats


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def conda_python(env_name):
    """Python of a conda env, found from CONDA_EXE or the usual ~/miniconda3."""
    if Path(sys.prefix).name == env_name:
        return sys.executable
    bases = []
    if os.environ.get("CONDA_EXE"):
        bases.append(Path(os.environ["CONDA_EXE"]).resolve().parent.parent)
    bases += [Path(sys.prefix).parent.parent, Path.home() / "miniconda3", Path.home() / "anaconda3"]
    for base in bases:
        python = base / "envs" / env_name / "bin" / "python"
        if python.exists():
            return str(python)
    return None


# ---------------------------------------------------------------------------
# Campaigns: one recording folder each, with its target and history in studio_campaign.json
# ---------------------------------------------------------------------------
def campaign_dir(folder):
    if not FOLDER_RE.match(folder):
        raise ValueError(f"not a recording folder name: {folder!r}")
    return SIM_DIR / folder


CAMPAIGN_LOCK = threading.RLock()  # campaign files are updated from request threads and round readers


def update_campaign(folder, change):
    """Load, change (change(campaign) edits it in place) and save a campaign file atomically."""
    with CAMPAIGN_LOCK:
        campaign = load_campaign(folder) or {}
        result = change(campaign)
        save_campaign(folder, campaign)
        return result


def load_campaign(folder):
    try:
        return json.loads((campaign_dir(folder) / CAMPAIGN_FILE).read_text())
    except (FileNotFoundError, ValueError):
        return None


def save_campaign(folder, campaign):
    path = campaign_dir(folder) / CAMPAIGN_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(campaign, indent=1) + "\n")
    os.replace(tmp, path)


def log_event(folder, event, **fields):
    update_campaign(folder, lambda c: c.setdefault("history", []).append({"t": now_iso(), "event": event, **fields}))


def list_campaigns():
    rows = []
    for path in sorted(SIM_DIR.glob("raw_teleop_data*")):
        if path.is_dir() and FOLDER_RE.match(path.name):
            campaign = load_campaign(path.name) or {}
            rows.append({"folder": path.name, "episodes": len(list(path.glob("episode_*.h5"))),
                         "target": campaign.get("target"), "round_size": campaign.get("round_size"),
                         "created": campaign.get("created")})
    return rows


def default_dataset_name(folder):
    return "lerobot_dataset" + folder.removeprefix("raw_teleop_data")


def dataset_taken(name):
    """An existing, non-empty dataset folder (the converter reuses an empty one)."""
    path = LEROBOT_DIR / name
    return path.exists() and (not path.is_dir() or any(path.iterdir()))


# ---------------------------------------------------------------------------
# Motor test: every leader-arm motor must answer and be moved by hand
# ---------------------------------------------------------------------------
class MotorTest:
    MOVE_DEG = 15.0  # a joint counts as moved once it has travelled this far

    def __init__(self, port):
        self.port = port
        self.lock = threading.Lock()
        self.running = False
        self.error = None
        self.reset_stats()
        self._thread = None

    def reset_stats(self):
        self.values = [None] * 6
        self.low = [None] * 6
        self.high = [None] * 6
        self.answers = [deque(maxlen=30) for _ in range(6)]

    def start(self):
        if self.running:
            return
        self.error = None
        self.reset_stats()
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self):
        try:
            from sim_to_real_so101.utils.leader_arm import LeaderArm

            arm = LeaderArm(self.port)
        except Exception as e:
            self.error = f"Could not open the leader arm at {self.port}: {e}"
            self.running = False
            return
        try:
            while self.running:
                deg = arm.read_raw_deg()
                with self.lock:
                    for i in range(6):
                        ok = bool(arm.last_read_ok[i])
                        self.answers[i].append(ok)
                        if ok:
                            v = float(deg[i])
                            self.values[i] = v
                            self.low[i] = v if self.low[i] is None else min(self.low[i], v)
                            self.high[i] = v if self.high[i] is None else max(self.high[i], v)
                time.sleep(0.05)
        except Exception as e:
            self.error = f"Lost the leader arm: {e}"
        finally:
            arm.close()
            self.running = False

    def status(self):
        with self.lock:
            motors = []
            for i, name in enumerate(JOINTS):
                answers = list(self.answers[i])
                responding = len(answers) >= 10 and sum(answers) / len(answers) >= 0.9
                travel = (self.high[i] - self.low[i]) if self.low[i] is not None else 0.0
                motors.append({"name": name, "deg": self.values[i], "responding": responding,
                               "travel": round(travel, 1), "moved": travel >= self.MOVE_DEG})
        reads = min(len(a) for a in self.answers)
        # The board answers on USB but no motor does: almost always the arm's power supply.
        no_answer = reads >= 10 and not any(any(a) for a in self.answers)
        return {"running": self.running, "error": self.error, "motors": motors, "no_answer": no_answer,
                "passed": all(m["responding"] and m["moved"] for m in motors)}


# ---------------------------------------------------------------------------
# Jobs: one recording round (teleop) or one conversion, run as a child process
# ---------------------------------------------------------------------------
class Job:
    def __init__(self, kind, cmd, cwd, on_exit=None, on_event=None):
        self.kind = kind
        self.status = "running"
        self.started = time.time()
        self.ended = None
        self.returncode = None
        self.stopping = False  # Stop was pressed; status becomes "stopped" once the process has exited
        self.drop = None  # Drop round: True = keep its episodes, False = trash them (handled on exit)
        self.log = deque(maxlen=300)
        self.info = {"phase": "starting"}
        self.on_exit = on_exit
        self.on_event = on_event
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        # Own process group, so Stop reaches teleop and its children but not this server.
        self.proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, start_new_session=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            self.log.append(line)
            try:
                self.parse(line)
            except Exception:
                pass
        self.returncode = self.proc.wait()
        self.ended = time.time()
        if self.stopping:
            self.status = "stopped"
        elif self.returncode == 0:
            self.status = "finished"
        else:
            self.status = "failed"
        if self.on_exit:
            self.on_exit(self)

    def parse(self, line):
        info = self.info
        if self.kind == "record":
            if line.startswith("[STUDIO] "):
                event = json.loads(line[len("[STUDIO] "):])
                kind = event.pop("event")
                if self.on_event:
                    self.on_event(kind, event)
                if kind == "ready":
                    info["phase"] = "recording"
                elif kind == "episode":
                    info.update(phase="recording", index=event["index"], layout=event.get("layout"),
                                target=event.get("target"), saved=event.get("saved", 0), max=event.get("max"))
                elif kind == "saved":
                    info["saved"] = info.get("saved", 0) + 1
                    info["last_saved"] = event.get("file")
                elif kind == "discarded":
                    info["discarded"] = info.get("discarded", 0) + 1
                elif kind == "compressing":
                    info["phase"] = "saving"
                elif kind == "idle_timeout":
                    info["idle"] = True
                elif kind == "finished":
                    info["phase"] = "finished"
            elif line.startswith("[ERROR]"):
                info["error"] = line.removeprefix("[ERROR]:").strip()
        elif self.kind == "convert":
            m = re.match(r"\[(\d+)/(\d+)\] Converting (\S+)", line)
            if m:
                info.update(phase="converting", done=int(m.group(1)) - 1, total=int(m.group(2)), current=m.group(3))
            elif line.startswith("Found "):
                info["phase"] = "converting"
            elif line.strip() == "Done.":
                info.update(phase="done", done=info.get("total", 0))
            elif "Error" in line or line.startswith("Traceback"):
                info["error"] = line.strip()

    def stop(self):
        """Ctrl+C for the round; force it if Isaac Sim does not close."""
        if self.proc.poll() is not None or self.stopping:
            return
        self.stopping = True

        def escalate():
            for sig, wait in ((signal.SIGINT, 40), (signal.SIGTERM, 10), (signal.SIGKILL, 0)):
                try:
                    os.killpg(self.proc.pid, sig)
                except ProcessLookupError:
                    return
                try:
                    self.proc.wait(timeout=wait or 1)
                    return
                except subprocess.TimeoutExpired:
                    continue

        threading.Thread(target=escalate, daemon=True).start()

    def public(self):
        return {"kind": self.kind, "status": self.status, "stopping": self.stopping, "started": self.started, "ended": self.ended,
                "returncode": self.returncode, "info": self.info, "log": list(self.log)[-12:]}


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------
class Studio:
    def __init__(self):
        self.lock = threading.Lock()
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.state = json.loads(STATE_FILE.read_text())
        except (FileNotFoundError, ValueError):
            self.state = {}
        self.jobs = {}
        self.motor_test = MotorTest(CONFIG["leader_port"])
        self.motor_test_passed_at = None
        self.last_heartbeat = time.time()
        self.library = None
        self._open_library()

    def _save_state(self):
        STATE_FILE.write_text(json.dumps(self.state, indent=1) + "\n")

    def _open_library(self):
        folder = self.state.get("folder")
        if folder and FOLDER_RE.match(folder) and campaign_dir(folder).is_dir():
            self.library = Episodes(campaign_dir(folder), CACHE_DIR)
        else:
            self.state.pop("folder", None)
            self.library = None

    def busy(self):
        return any(j.proc.poll() is None for j in self.jobs.values())

    # --- campaign ---
    def set_campaign(self, folder, target=None, round_size=None, new_name=None):
        if self.busy():
            raise ValueError("Wait until the current round or conversion has finished.")
        if new_name is not None:
            clean = re.sub(r"[^a-z0-9_]+", "_", str(new_name).strip().lower()).strip("_")
            if not clean:
                raise ValueError("Please type a name for the new campaign (letters and numbers).")
            folder = f"raw_teleop_data_{clean}"
            if campaign_dir(folder).exists():
                raise ValueError(f"A campaign called {clean!r} already exists: pick it from the list instead.")
            if target is None or round_size is None:
                raise ValueError("Choose the number of episodes and the round size.")
        elif not folder or not campaign_dir(folder).is_dir():
            raise ValueError(f"{folder} does not exist.")
        with CAMPAIGN_LOCK:
            self._write_settings(folder, new_name, target, round_size)
        self.state["folder"] = folder
        self._save_state()
        self._open_library()
        campaign = load_campaign(folder)
        log_event(folder, "campaign_opened", target=campaign["target"], round_size=campaign["round_size"])

    def _write_settings(self, folder, new_name, target, round_size):
        campaign = (load_campaign(folder) if new_name is None else None) or {"created": now_iso(), "history": []}
        if target is not None:
            target = int(target)
            if not 6 <= target <= 5000:
                raise ValueError("The number of episodes must be between 6 and 5000.")
            campaign["target"] = target
        if round_size is not None:
            round_size = int(round_size)
            if round_size not in CONFIG["round_sizes"]:
                raise ValueError(f"The round size must be one of {CONFIG['round_sizes']}.")
            campaign["round_size"] = round_size
        if not campaign.get("target") or not campaign.get("round_size"):
            raise ValueError("Choose the number of episodes and the round size.")
        campaign_dir(folder).mkdir(parents=True, exist_ok=True)  # only once everything is valid
        save_campaign(folder, campaign)

    # --- counts and steps ---
    def counts(self):
        if not self.library:
            return None
        eps = self.library.list()
        ok = [e for e in eps if e["status"] == "ok"]
        balance = {f"{layout}/{color}": 0 for layout in LAYOUTS for color in COLORS}
        for e in ok:
            key = f"{e.get('layout')}/{e.get('target')}"
            if key in balance:
                balance[key] += 1
        approved = sum(e["approved"] for e in ok)
        flagged = sum(e["flagged"] for e in ok)
        return {"episodes": len(ok), "approved": approved, "flagged": flagged,
                "pending": len(ok) - approved - flagged,
                "compressing": sum(e["status"] == "compressing" for e in eps),
                "errors": sum(e["status"] == "error" for e in eps),
                "trash": self.library.trash_count(),
                "minutes": round(sum(e.get("seconds", 0) for e in ok) / 60, 1), "balance": balance}

    def snapshot(self):
        folder = self.state.get("folder")
        campaign = load_campaign(folder) if folder else None
        counts = self.counts()
        record_job = self.jobs.get("record")
        convert_job = self.jobs.get("convert")
        round_ = (campaign or {}).get("round")
        next_round = 0
        if round_:
            next_round = round_["size"] - round_["saved"]
        elif campaign and counts:
            remaining = campaign["target"] - counts["approved"] - counts["pending"] - counts["flagged"]
            next_round = max(0, min(campaign["round_size"], remaining))
        dataset = None
        if folder:
            dataset = (campaign or {}).get("dataset") or default_dataset_name(folder)
        return {
            "folder": folder,
            "campaign": {k: campaign.get(k) for k in ("target", "round_size", "created", "dataset")} if campaign else None,
            "history": (campaign or {}).get("history", [])[-15:],
            "round": round_,  # open round (recording or paused): {size, saved, files, started, paused}
            "counts": counts,
            "next_round": next_round,
            "dataset_name": dataset,
            "dataset_exists": bool(dataset and dataset_taken(dataset)),
            "record": record_job.public() if record_job else None,
            "convert": convert_job.public() if convert_job else None,
            "motor_test_passed": self.motor_test_passed_at is not None,
            "config": {k: CONFIG[k] for k in ("task", "leader_port", "round_sizes", "default_target")},
            "campaigns": list_campaigns(),
        }

    # --- checks ---
    def checks(self):
        port = CONFIG["leader_port"]
        exists = os.path.exists(port)
        access = exists and os.access(port, os.R_OK | os.W_OK)
        rows = [
            ("arm", "Leader arm plugged in", exists, port if exists else f"{port} not found",
             "Plug the leader arm's USB cable into this computer, then wait a few seconds."),
            ("access", "Leader arm can be opened", access, "" if access else ("no permission" if exists else ""),
             "Ask the admin to give this user access to the leader arm port (dialout group)." if exists
             else "Plug in the leader arm first."),
        ]
        try:
            from sim_to_real_so101.utils.leader_arm import DEFAULT_CALIBRATION_PATH, load_calibration

            calibration = load_calibration(DEFAULT_CALIBRATION_PATH)
            calibrated = {"Wrist_Roll", "Jaw"} <= set(calibration)
        except Exception:
            calibrated = False
        rows.append(("calibration", "Leader arm calibrated", calibrated, "",
                     "Ask the admin to calibrate the leader arm (scripts/calibrate_leader.py)."))
        envs = [CONFIG["isaaclab_env"], CONFIG["lerobot_env"]]
        missing = [e for e in envs if not conda_python(e)]
        rows.append(("envs", "Software installed", not missing, ", ".join(missing),
                     "Ask the admin: conda environment(s) missing: " + ", ".join(missing)))
        free_gb = shutil.disk_usage(SIM_DIR).free / 1e9
        rows.append(("disk", "Enough disk space", free_gb >= CONFIG["min_free_disk_gb"], f"{free_gb:.0f} GB free",
                     f"Free up disk space: at least {CONFIG['min_free_disk_gb']} GB are needed."))
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            free_mb = max(int(x) for x in out.split())
            gpu_ok, gpu_detail = free_mb >= CONFIG["min_free_gpu_mb"], f"{free_mb / 1024:.1f} GB free"
        except Exception:
            gpu_ok, gpu_detail = False, "no GPU found"
        rows.append(("gpu", "Graphics card ready", gpu_ok, gpu_detail,
                     "Close other programs that use the graphics card (another simulator, a game...)."))
        others = subprocess.run(["pgrep", "-f", "teleop_so101.py"], capture_output=True, text=True).stdout.split()
        ours = {str(j.proc.pid) for j in self.jobs.values() if j.proc.poll() is None}
        running = [p for p in others if p not in ours]
        rows.append(("sim", "No other recording open", not running, "",
                     "Another recording window is open: close it first."))
        return [{"id": i, "label": label, "ok": bool(ok), "detail": detail, "fix": fix}
                for i, label, ok, detail, fix in rows]

    # --- record ---
    def start_round(self):
        """A new round: needs the last round reviewed and the leader arm test passed."""
        snap = self.snapshot()
        folder, counts = snap["folder"], snap["counts"]
        if not folder:
            raise ValueError("Choose a campaign first.")
        if self.busy():
            raise ValueError("A round or a conversion is already running.")
        if snap["round"]:
            raise ValueError("A round is paused: continue it from the Record step.")
        if counts["pending"] or counts["flagged"]:
            raise ValueError("Review the last round first.")
        if snap["next_round"] <= 0:
            raise ValueError("The campaign already has all its episodes.")
        if not self.motor_test_passed_at:
            raise ValueError("Do the leader arm test first.")
        self._require_checks()
        new_round = {"size": snap["next_round"], "saved": 0, "files": [], "started": now_iso(), "paused": None}
        update_campaign(folder, lambda c: c.__setitem__("round", new_round))
        log_event(folder, "round_started", episodes=snap["next_round"])
        self.motor_test_passed_at = None  # test again before the next new round
        self._launch_round(folder)

    def continue_round(self):
        """Continue a paused round where it stopped (no new arm test needed)."""
        snap = self.snapshot()
        folder = snap["folder"]
        if not folder or not snap["round"]:
            raise ValueError("There is no paused round to continue.")
        if self.busy():
            raise ValueError("A round or a conversion is already running.")
        self._require_checks()
        log_event(folder, "round_continued", left=snap["next_round"])
        self._launch_round(folder)

    def _require_checks(self):
        failed = [c["label"] for c in self.checks() if not c["ok"]]
        if failed:
            raise ValueError("Fix these checks first: " + ", ".join(failed))

    def _launch_round(self, folder):
        self.motor_test.stop()  # the round needs the serial port
        round_ = load_campaign(folder)["round"]
        left = round_["size"] - round_["saved"]
        cmd = [conda_python(CONFIG["isaaclab_env"]), "scripts/teleop_so101.py",
               "--task", CONFIG["task"], "--leader_port", CONFIG["leader_port"],
               "--output_dir", str(campaign_dir(folder)), "--max_episodes", str(left),
               "--idle_timeout", str(CONFIG["idle_timeout_s"]),
               "--no_auto_save", "--viewport_only", "--studio"]

        def on_event(kind, event):
            if kind == "saved":  # count every save in the campaign file, so a pause can resume
                def add(c):
                    c["round"]["saved"] += 1
                    c["round"].setdefault("files", []).append(event.get("file"))
                update_campaign(folder, add)

        def on_exit(job):
            if job.drop is not None:  # Drop round was pressed while recording
                self._finish_drop(folder, keep=job.drop)
                return
            with CAMPAIGN_LOCK:
                campaign = load_campaign(folder)
                round_now = campaign.get("round") or {}
                if round_now.get("saved", 0) >= round_now.get("size", 0):
                    campaign.pop("round", None)
                    save_campaign(folder, campaign)
                    log_event(folder, "round_finished", saved=round_now.get("saved"))
                    return
            if job.stopping:
                reason = "paused"
            elif job.info.get("idle"):
                reason = "no movement"
            elif job.returncode == 0:
                reason = "simulator closed"
            else:
                reason = "error"
            update_campaign(folder, lambda c: c["round"].__setitem__("paused", reason))
            log_event(folder, "round_paused", reason=reason,
                      saved=round_now.get("saved"), size=round_now.get("size"))

        update_campaign(folder, lambda c: c["round"].__setitem__("paused", None))
        self.jobs["record"] = Job("record", cmd, SIM_DIR, on_exit, on_event)

    def pause_round(self):
        job = self.jobs.get("record")
        if job:
            job.stop()

    def drop_round(self, keep):
        """Cancel the open round. keep=True leaves its saved episodes for review, False trashes them."""
        folder = self.state.get("folder")
        if not folder or not (load_campaign(folder) or {}).get("round"):
            raise ValueError("There is no round to drop.")
        job = self.jobs.get("record")
        if job and job.proc.poll() is None:
            job.drop = bool(keep)  # finished in on_exit once the simulator has closed
            job.stop()
        else:
            self._finish_drop(folder, bool(keep))

    def _round_files(self, folder, round_):
        """Episodes saved in a round: its file list, or (for a round started before rounds kept one)
        the episodes saved since the round started."""
        files = [f for f in round_.get("files", []) if f]
        if len(files) >= round_.get("saved", 0):
            return files
        started = datetime.fromisoformat(round_["started"]).timestamp() if round_.get("started") else time.time()
        return sorted(p.name for p in campaign_dir(folder).glob("episode_*.h5") if p.stat().st_mtime >= started)

    def _finish_drop(self, folder, keep):
        with CAMPAIGN_LOCK:
            campaign = load_campaign(folder)
            round_ = campaign.pop("round", None) or {}
            save_campaign(folder, campaign)
        trashed = [] if keep or not self.library else self.library.move_to_trash(self._round_files(folder, round_))
        log_event(folder, "round_dropped", saved=round_.get("saved", 0),
                  kept=keep, trashed=trashed)

    # --- review ---
    def finish_review(self):
        folder = self.state.get("folder")
        if not self.library:
            raise ValueError("Choose a campaign first.")
        if (load_campaign(folder) or {}).get("round"):
            raise ValueError("Finish the round first: it is paused, continue it from the Record step.")
        counts = self.counts()
        if counts["pending"]:
            raise ValueError(f"{counts['pending']} episode(s) still need a decision.")
        flagged = self.library.review_state()["flagged"]  # name -> reason
        trashed = [{"name": n, "reason": r} for n, r in sorted(flagged.items())]
        moved = self.library.move_to_trash(sorted(flagged))
        log_event(folder, "review_finished", approved=counts["approved"], trashed=trashed)
        return moved

    # --- convert ---
    def start_convert(self, name):
        snap = self.snapshot()
        folder, counts, campaign = snap["folder"], snap["counts"], snap["campaign"]
        if self.busy():
            raise ValueError("A round or a conversion is already running.")
        if not folder or counts["pending"] or counts["flagged"] or counts["approved"] < campaign["target"]:
            raise ValueError("All episodes must be recorded and approved before converting.")
        name = str(name or snap["dataset_name"]).strip()
        if not DATASET_RE.match(name):
            raise ValueError("The dataset name may only use letters, numbers, '_', '-' and '.'.")
        if dataset_taken(name):
            raise ValueError(f"lerobot/{name} already exists: choose another name.")
        python = conda_python(CONFIG["lerobot_env"])
        cmd = [python, "convert_sim_data.py", "--raw_dir", str(campaign_dir(folder)), "--output_dir", name]
        update_campaign(folder, lambda c: c.__setitem__("dataset", name))

        def on_exit(job):
            log_event(folder, "conversion_finished", dataset=name, status=job.status, returncode=job.returncode)

        self.jobs["convert"] = Job("convert", cmd, LEROBOT_DIR, on_exit)
        log_event(folder, "conversion_started", dataset=name, episodes=counts["approved"])


STUDIO = None


class StudioHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the log quiet; errors are printed separately
        pass

    def _json(self, data, status=HTTPStatus.OK):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message):
        self._json({"error": message}, status)

    def _file(self, path: Path, content_type):
        """Send a file, honouring a byte Range request (needed for seeking in videos)."""
        size = path.stat().st_size
        start, end = 0, size - 1
        match = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", ""))
        if match and (match.group(1) or match.group(2)):
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else end
            else:  # suffix range: the last N bytes
                start = max(size - int(match.group(2)), 0)
            end = min(end, size - 1)
            if start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1 << 16, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):  # the window seeked elsewhere
                    return
                remaining -= len(chunk)

    def _static(self, name):
        if not STATIC_RE.match(name) or not (WEB_DIR / name).is_file():
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        types = {"html": "text/html; charset=utf-8", "css": "text/css; charset=utf-8",
                 "js": "text/javascript; charset=utf-8", "svg": "image/svg+xml", "jpg": "image/jpeg"}
        self._file(WEB_DIR / name, types[name.rsplit(".", 1)[1]])

    def do_GET(self):
        path = urlparse(self.path).path.strip("/")
        parts = [unquote(p) for p in path.split("/") if p]
        try:
            if not parts:
                return self._static("index.html")
            if parts[0] == "static" and len(parts) == 2:
                return self._static(parts[1])
            if path == "api/studio":
                return self._json(STUDIO.snapshot())
            if path == "api/checks":
                return self._json({"checks": STUDIO.checks()})
            if path == "api/motors":
                return self._json(STUDIO.motor_test.status())
            if path == "api/ping":
                return self._json({"ok": True})
            library = STUDIO.library
            if path == "api/episodes" or parts[0] in ("video",) or parts[:2] == ["api", "joints"]:
                if library is None:
                    return self._error(HTTPStatus.CONFLICT, "Choose a campaign first.")
            if path == "api/episodes":
                return self._json({"episodes": library.list()})
            if len(parts) == 3 and parts[:2] == ["api", "joints"]:
                return self._json(library.joints(parts[2]))
            if len(parts) == 3 and parts[0] == "video":
                return self._file(library.video(parts[1], parts[2].removesuffix(".mp4")), "video/mp4")
            self._error(HTTPStatus.NOT_FOUND, "not found")
        except (ValueError, FileNotFoundError) as e:
            self._error(HTTPStatus.NOT_FOUND, str(e))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            print(f"[ERROR] GET {self.path}: {e}", flush=True)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def do_POST(self):
        path = urlparse(self.path).path.strip("/")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if path == "api/review":  # one decision; quick, no need for the app lock
                if STUDIO.library is None:
                    return self._error(HTTPStatus.CONFLICT, "Choose a campaign first.")
                STUDIO.library.decide(body["name"], body["decision"], body.get("reason", ""))
                return self._json({"ok": True})
            with STUDIO.lock:
                if path == "api/heartbeat":
                    STUDIO.last_heartbeat = time.time()
                elif path == "api/campaign":
                    STUDIO.set_campaign(body.get("folder"), body.get("target"), body.get("round_size"),
                                        body.get("new_name"))
                elif path == "api/motors/start":
                    if STUDIO.busy():
                        raise ValueError("A round is running: the leader arm is in use.")
                    STUDIO.motor_test.start()
                elif path == "api/motors/stop":
                    if STUDIO.motor_test.status()["passed"]:
                        STUDIO.motor_test_passed_at = time.time()
                    STUDIO.motor_test.stop()
                elif path == "api/record/start":
                    if STUDIO.motor_test.status()["passed"]:
                        STUDIO.motor_test_passed_at = time.time()
                    STUDIO.start_round()
                elif path == "api/record/pause":
                    STUDIO.pause_round()
                elif path == "api/record/continue":
                    STUDIO.continue_round()
                elif path == "api/record/drop":
                    STUDIO.drop_round(body.get("keep", True))
                elif path == "api/review/finish":
                    return self._json({"moved": STUDIO.finish_review()})
                elif path == "api/convert/start":
                    STUDIO.start_convert(body.get("name"))
                else:
                    return self._error(HTTPStatus.NOT_FOUND, "not found")
            self._json({"ok": True})
        except (ValueError, KeyError) as e:
            self._error(HTTPStatus.BAD_REQUEST, str(e))
        except Exception as e:
            print(f"[ERROR] POST {self.path}: {e}", flush=True)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))


def watchdog(server):
    """Stop the server once the window has been closed for a while and nothing is running."""
    while True:
        time.sleep(15)
        idle = time.time() - STUDIO.last_heartbeat
        if idle > IDLE_EXIT_S and not STUDIO.busy() and not STUDIO.motor_test.running:
            print("[INFO] No window open for a while: stopping Data Studio.", flush=True)
            server.shutdown()
            return


def main():
    global STUDIO
    STUDIO = Studio()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", CONFIG["port"]), StudioHandler)
    except OSError as e:
        raise SystemExit(f"Port {CONFIG['port']} is in use: Data Studio may already be running ({e}).")
    threading.Thread(target=watchdog, args=(server,), daemon=True).start()
    print(f"SO-Arm Data Studio on http://127.0.0.1:{CONFIG['port']}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        STUDIO.motor_test.stop()


if __name__ == "__main__":
    main()
