"""Episodes of one Data Studio campaign: listing, automatic warnings, joint data, review
decisions, trash and the videos shown in the review.

Review decisions are kept in studio_review.json inside the recordings folder, separate from
review_flags.json of the admin's own review page (lerobot/review/). Both move deleted episodes to
the same _trash/ folder, which the converter, replay and teleop ignore.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import h5py
import numpy as np

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
EPISODE_RE = re.compile(r"^episode_(\d+)\.h5$")
TMP_RE = re.compile(r"^episode_(\d+)\.h5\.tmp$")
CAMERA_RE = re.compile(r"^[A-Za-z0-9_]+$")
COLOR_RE = re.compile(r"\b(red|blue)\b")
TRASH = "_trash"
REVIEW_FILE = "studio_review.json"
FRAMES_PER_READ = 64
# The leader's home is not the sim's reset pose (the sim robot jumps to the leader at the start),
# so the start check compares with the usual leader start of the folder. Normal starts differ by
# ~10 units; 40 flags only clear outliers.
START_TOLERANCE = 40.0


def episode_warnings(state, action, fps):
    """Automatic checks of one recording, shown as warnings in the review (they never block)."""
    warnings = []
    if len(action) < 2 * fps:
        warnings.append("very short (under 2 s)")
    gripper = action[:, 5]
    open_level = float(np.percentile(gripper, 95))
    if open_level > 10 and gripper[-1] < 0.4 * open_level:
        warnings.append("still gripping at the end (no release)")
    return warnings


def ffmpeg_binary():
    """ffmpeg from PATH, the conda env, or the imageio-ffmpeg package (installed in env_isaaclab)."""
    candidates = [shutil.which("ffmpeg"), str(Path(sys.prefix) / "bin" / "ffmpeg")]
    try:
        import imageio_ffmpeg

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    for found in candidates:
        if found and Path(found).exists():
            return found
    raise RuntimeError("ffmpeg not found")


class Episodes:
    def __init__(self, raw_dir: Path, cache_dir: Path):
        self.raw_dir = raw_dir
        self.trash_dir = raw_dir / TRASH
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._meta = {}  # name -> (mtime_ns, size, info)
        self._video_locks = {}
        self._lock = threading.Lock()

    def path(self, name, in_trash=False):
        """Path of an episode by name; rejects anything that is not episode_xxxx.h5."""
        if not EPISODE_RE.match(name):
            raise ValueError(f"not an episode name: {name!r}")
        return (self.trash_dir if in_trash else self.raw_dir) / name

    # --- review decisions ---------------------------------------------------
    def review_state(self):
        try:
            raw = json.loads((self.raw_dir / REVIEW_FILE).read_text())
        except (FileNotFoundError, ValueError):
            raw = {}
        return {"approved": set(raw.get("approved", [])), "flagged": dict(raw.get("flagged", {}))}  # flagged: name -> reason

    def _write_review(self, state):
        data = {"approved": sorted(state["approved"]), "flagged": dict(sorted(state["flagged"].items()))}
        tmp = self.raw_dir / (REVIEW_FILE + ".tmp")
        tmp.write_text(json.dumps(data, indent=1) + "\n")
        os.replace(tmp, self.raw_dir / REVIEW_FILE)

    def decide(self, name, decision, reason=""):
        """decision: "approve", "flag" (with a reason) or "clear"."""
        self.path(name)
        if decision not in ("approve", "flag", "clear"):
            raise ValueError(f"unknown decision {decision!r}")
        with self._lock:
            state = self.review_state()
            state["approved"].discard(name)
            state["flagged"].pop(name, None)
            if decision == "approve":
                state["approved"].add(name)
            elif decision == "flag":
                state["flagged"][name] = reason or "bad"
            self._write_review(state)

    # --- listing -------------------------------------------------------------
    def _info(self, path: Path):
        st = path.stat()
        cached = self._meta.get(path.name)
        if cached and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
            return cached[2]
        info = {"name": path.name, "index": int(EPISODE_RE.match(path.name).group(1)),
                "size_mb": round(st.st_size / 1e6, 1), "status": "ok"}
        try:
            with h5py.File(path, "r") as f:
                attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
                fps = int(attrs.get("fps", 30))
                action = np.asarray(f["action"][()])
                state = np.asarray(f["observation.state"][()])
                instruction = attrs.get("instruction")
                color = COLOR_RE.search(instruction) if instruction else None
                info.update(
                    fps=fps, frames=len(action), seconds=round(len(action) / fps, 1),
                    cameras=sorted(k.removeprefix("observation.images.") for k in f if k.startswith("observation.images.")),
                    instruction=instruction, target=color.group(1) if color else None,
                    layout=attrs.get("layout"), operator=attrs.get("operator"),
                    warnings=episode_warnings(state, action, fps),
                    start=np.round(action[0].astype(float), 2).tolist() if len(action) else None,
                )
        except Exception as e:  # unreadable file: still listed, so it can be thrown away
            info.update(status="error", error=str(e), frames=0, seconds=0, cameras=[], warnings=[])
        self._meta[path.name] = (st.st_mtime_ns, st.st_size, info)
        return info

    def list(self):
        review = self.review_state()
        rows = []
        for path in sorted(self.raw_dir.glob("episode_*.h5")):
            if EPISODE_RE.match(path.name):
                info = self._info(path)
                rows.append({**info, "approved": path.name in review["approved"],
                             "flagged": path.name in review["flagged"], "reason": review["flagged"].get(path.name)})
        for path in sorted(self.raw_dir.glob("episode_*.h5.tmp")):
            m = TMP_RE.match(path.name)
            final = path.name[: -len(".tmp")]
            if m and not (self.raw_dir / final).exists():
                rows.append({"name": final, "index": int(m.group(1)), "status": "compressing", "frames": 0,
                             "seconds": 0, "cameras": [], "warnings": [], "approved": False, "flagged": False})
        starts = [r["start"] for r in rows if r.get("start") and r.get("frames", 0) >= 60]
        usual = np.median(np.array(starts), axis=0) if len(starts) >= 5 else None
        for r in rows:
            start = r.pop("start", None)
            r["warnings"] = list(r.get("warnings", []))
            if usual is not None and start and r.get("frames", 0) >= 60:
                if float(np.abs(np.asarray(start[:5]) - usual[:5]).max()) > START_TOLERANCE:
                    r["warnings"].append("did not start from the usual home pose")
        return sorted(rows, key=lambda r: r["index"])

    def trash_count(self):
        return len(list(self.trash_dir.glob("episode_*.h5"))) if self.trash_dir.is_dir() else 0

    # --- joints ---------------------------------------------------------------
    def joints(self, name):
        with h5py.File(self.path(name), "r") as f:
            state = np.asarray(f["observation.state"][()], dtype=np.float32)
            action = np.asarray(f["action"][()], dtype=np.float32)
            fps = int(f.attrs.get("fps", 30))
        return {"names": JOINT_NAMES, "fps": fps,
                "state": np.round(state.T, 2).tolist(), "action": np.round(action.T, 2).tolist()}

    # --- trash ----------------------------------------------------------------
    def move_to_trash(self, names):
        self.trash_dir.mkdir(exist_ok=True)
        moved = []
        for name in names:
            src = self.path(name)
            if not src.exists():
                continue
            dst = self.trash_dir / name
            if dst.exists():  # an older trashed copy with the same name: keep both
                dst = self.trash_dir / f"{name[:-3]}_{int(time.time())}.h5"
            os.replace(src, dst)
            self.decide(name, "clear")
            moved.append(name)
        return moved

    # --- video ----------------------------------------------------------------
    def video(self, name, camera):
        """H.264 MP4 of one camera, made with ffmpeg on first request and cached."""
        path = self.path(name)
        if not CAMERA_RE.match(camera):
            raise ValueError(f"bad camera name: {camera!r}")
        st = path.stat()
        out = self.cache_dir / f"{self.raw_dir.name}_{path.stem}_{st.st_mtime_ns}_{st.st_size}_{camera}.mp4"
        with self._lock:
            lock = self._video_locks.setdefault(out.name, threading.Lock())
        with lock:
            if out.exists():
                return out
            with h5py.File(path, "r") as f:
                key = f"observation.images.{camera}"
                if key not in f:
                    raise FileNotFoundError(f"{name} has no camera {camera!r}")
                frames = f[key]
                count, height, width = frames.shape[:3]
                fps = int(f.attrs.get("fps", 30))
                part = out.with_suffix(".part.mp4")
                cmd = [ffmpeg_binary(), "-y", "-loglevel", "error",
                       "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                       "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                       "-g", str(max(fps // 2, 1)), "-movflags", "+faststart", str(part)]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    for start in range(0, count, FRAMES_PER_READ):
                        proc.stdin.write(np.ascontiguousarray(frames[start:start + FRAMES_PER_READ], dtype=np.uint8).tobytes())
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
                if proc.wait() != 0:
                    part.unlink(missing_ok=True)
                    raise RuntimeError(f"ffmpeg failed: {proc.stderr.read().decode(errors='replace')}")
            os.replace(part, out)
            return out
