"""
review_data.py

Review raw teleop recordings (episode_xxxx.h5) in the browser before converting them:
watch the cameras, check the joints, flag bad episodes and move them to a trash folder.

Runs in either conda env (lerobot_env or env_isaaclab: it needs only h5py, numpy and an
ffmpeg with H.264), then open the printed address:
    python review_data.py --raw_dir ../../source/sim_to_real_so101/raw_teleop_data_twocubes

Deleted episodes go to <raw_dir>/_trash/ and can be restored from the page. The converter
and the replay script only read episode_*.h5 directly in raw_dir, so they ignore the trash.
The page is served on 127.0.0.1 only: other computers on the network cannot reach it.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PAGE = SCRIPT_DIR / "review_data.html"
RECORDINGS_DIR = (SCRIPT_DIR / "../../source/sim_to_real_so101").resolve()  # where teleop saves folders
DEFAULT_RAW_DIR = RECORDINGS_DIR / "raw_teleop_data"

# Same order as the 6-D state/action vectors (see convert_sim_data.py).
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
EPISODE_RE = re.compile(r"^episode_(\d+)\.h5$")
TMP_RE = re.compile(r"^episode_(\d+)\.h5\.tmp$")
COLOR_RE = re.compile(r"\b(red|blue)\b")
TRASH = "_trash"
FLAGS_FILE = "review_flags.json"
FRAMES_PER_READ = 64


class Library:
    """The recordings folder: listing, flags, trash and cached videos."""

    def __init__(self, raw_dir: Path, cache_dir: Path):
        self.raw_dir = raw_dir
        self.trash_dir = raw_dir / TRASH
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._meta = {}  # name -> (mtime_ns, size, info)
        self._video_locks = {}
        self._lock = threading.Lock()

    # --- paths -------------------------------------------------------------
    def episode_path(self, name, in_trash=False):
        """Path of an episode by name; rejects anything that is not episode_xxxx.h5."""
        if not EPISODE_RE.match(name):
            raise ValueError(f"not an episode name: {name!r}")
        return (self.trash_dir if in_trash else self.raw_dir) / name

    # --- flags -------------------------------------------------------------
    def _flags_path(self):
        return self.raw_dir / FLAGS_FILE

    def flags(self):
        try:
            return set(json.loads(self._flags_path().read_text())["flagged"])
        except (FileNotFoundError, ValueError, KeyError):
            return set()

    def set_flag(self, name, flagged):
        self.episode_path(name)
        with self._lock:
            flags = self.flags()
            (flags.add if flagged else flags.discard)(name)
            self._flags_path().write_text(json.dumps({"flagged": sorted(flags)}, indent=1) + "\n")

    # --- listing -----------------------------------------------------------
    def _info(self, path: Path):
        st = path.stat()
        cached = self._meta.get(path.name)
        if cached and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
            return cached[2]
        info = {"name": path.name, "index": int(EPISODE_RE.match(path.name).group(1)),
                "size_mb": round(st.st_size / 1e6, 1), "mtime": int(st.st_mtime), "status": "ok"}
        try:
            with h5py.File(path, "r") as f:
                attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
                fps = int(attrs.get("fps", 30))
                frames = int(f["action"].shape[0])
                instruction = attrs.get("instruction")
                color = COLOR_RE.search(instruction) if instruction else None
                info.update(
                    fps=fps, frames=frames, seconds=round(frames / fps, 1),
                    cameras=sorted(k.removeprefix("observation.images.") for k in f if k.startswith("observation.images.")),
                    instruction=instruction, target=color.group(1) if color else None,
                    layout=attrs.get("layout"), action_source=attrs.get("action_source", "follower_position (old)"),
                )
        except Exception as e:  # unreadable file: still listed, so it can be deleted
            info.update(status="error", error=str(e), frames=0, seconds=0, cameras=[])
        self._meta[path.name] = (st.st_mtime_ns, st.st_size, info)
        return info

    def episodes(self):
        flags = self.flags()
        rows = []
        for path in sorted(self.raw_dir.glob("episode_*.h5")):
            if EPISODE_RE.match(path.name):
                rows.append({**self._info(path), "flagged": path.name in flags})
        for path in sorted(self.raw_dir.glob("episode_*.h5.tmp")):
            m = TMP_RE.match(path.name)
            final = path.name[: -len(".tmp")]
            if m and not (self.raw_dir / final).exists():
                rows.append({"name": final, "index": int(m.group(1)), "status": "compressing",
                             "size_mb": round(path.stat().st_size / 1e6, 1), "mtime": int(path.stat().st_mtime),
                             "frames": 0, "seconds": 0, "cameras": [], "flagged": False})
        return sorted(rows, key=lambda r: r["index"])

    def trash(self):
        if not self.trash_dir.is_dir():
            return []
        return [{"name": p.name, "size_mb": round(p.stat().st_size / 1e6, 1)}
                for p in sorted(self.trash_dir.glob("episode_*.h5")) if EPISODE_RE.match(p.name)]

    # --- joints ------------------------------------------------------------
    def joints(self, name):
        with h5py.File(self.episode_path(name), "r") as f:
            state = np.asarray(f["observation.state"][()], dtype=np.float32)
            action = np.asarray(f["action"][()], dtype=np.float32)
            fps = int(f.attrs.get("fps", 30))
        return {"names": JOINT_NAMES, "fps": fps,
                "state": np.round(state.T, 2).tolist(), "action": np.round(action.T, 2).tolist()}

    # --- trash -------------------------------------------------------------
    def delete(self, names):
        self.trash_dir.mkdir(exist_ok=True)
        moved = []
        for name in names:
            src = self.episode_path(name)
            if not src.exists():
                continue
            dst = self.trash_dir / name
            if dst.exists():  # an older trashed copy with the same name: keep both
                dst = self.trash_dir / f"{name[:-3]}_{int(time.time())}.h5"
            os.replace(src, dst)
            self.set_flag(name, False)
            moved.append(name)
        return moved

    def restore(self, names):
        restored, skipped = [], []
        for name in names:
            src = self.episode_path(name, in_trash=True)
            dst = self.episode_path(name)
            if not src.exists():
                continue
            if dst.exists():  # the recorder reused the number after the delete
                skipped.append(name)
                continue
            os.replace(src, dst)
            restored.append(name)
        return restored, skipped

    def empty_trash(self):
        removed = 0
        for item in self.trash():
            (self.trash_dir / item["name"]).unlink()
            removed += 1
        return removed

    # --- video -------------------------------------------------------------
    def video(self, name, camera):
        """H.264 MP4 of one camera, made with ffmpeg on first request and cached."""
        path = self.episode_path(name)
        if not re.match(r"^[A-Za-z0-9_]+$", camera):
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
    raise RuntimeError("ffmpeg not found: run this in lerobot_env, or install ffmpeg")


def recording_folders(extra=None):
    """Every recording folder (raw_teleop_data, raw_teleop_data_<name>), plus --raw_dir if it is elsewhere."""
    paths = sorted(p for p in RECORDINGS_DIR.glob("raw_teleop_data*") if p.is_dir())
    if extra is not None and extra not in paths:
        paths.append(extra)
    return [{"name": p.name if p.parent == RECORDINGS_DIR else str(p), "path": str(p),
             "episodes": len(list(p.glob("episode_*.h5")))} for p in paths]


class Handler(BaseHTTPRequestHandler):
    library: Library = None
    start_dir: Path = None  # --raw_dir
    cache_dir: Path = None

    def log_message(self, fmt, *args):  # keep the terminal quiet; errors are printed separately
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
                except (BrokenPipeError, ConnectionResetError):  # the browser seeked elsewhere
                    return
                remaining -= len(chunk)

    def do_GET(self):
        parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/") if p]
        lib = self.library
        try:
            if not parts:
                self._file(PAGE, "text/html; charset=utf-8")
            elif parts == ["api", "episodes"]:
                self._json({"raw_dir": str(lib.raw_dir), "folders": recording_folders(self.start_dir),
                            "episodes": lib.episodes(), "trash": lib.trash()})
            elif len(parts) == 3 and parts[:2] == ["api", "joints"]:
                self._json(lib.joints(parts[2]))
            elif len(parts) == 3 and parts[0] == "video":
                self._file(lib.video(parts[1], parts[2].removesuffix(".mp4")), "video/mp4")
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
        except (ValueError, FileNotFoundError) as e:
            self._error(HTTPStatus.NOT_FOUND, str(e))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            print(f"[ERROR] GET {self.path}: {e}", flush=True)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def do_POST(self):
        lib = self.library
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            action = urlparse(self.path).path.strip("/")
            if action == "api/folder":  # switch to another recording folder from the list
                allowed = {f["path"] for f in recording_folders(self.start_dir)}
                if body.get("path") not in allowed:
                    raise ValueError("not a recording folder")
                Handler.library = Library(Path(body["path"]), self.cache_dir)
                print(f"[INFO] reviewing {body['path']}", flush=True)
                self._json({"ok": True})
            elif action == "api/flag":
                lib.set_flag(body["name"], bool(body["flagged"]))
                self._json({"ok": True})
            elif action == "api/delete":
                moved = lib.delete(body["names"])
                print(f"[INFO] moved to {TRASH}/: {', '.join(moved) or 'nothing'}", flush=True)
                self._json({"moved": moved})
            elif action == "api/restore":
                restored, skipped = lib.restore(body["names"])
                self._json({"restored": restored, "skipped": skipped})
            elif action == "api/empty_trash":
                removed = lib.empty_trash()
                print(f"[INFO] emptied {TRASH}/: {removed} file(s) deleted", flush=True)
                self._json({"removed": removed})
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
        except (ValueError, KeyError) as e:
            self._error(HTTPStatus.BAD_REQUEST, str(e))
        except Exception as e:
            print(f"[ERROR] POST {self.path}: {e}", flush=True)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_DIR,
                        help="Folder to start with (any recording folder can be picked in the page)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cache_dir", type=Path, default=Path.home() / ".cache" / "so101_review",
                        help="Where the converted videos are kept")
    args = parser.parse_args()

    raw_dir = args.raw_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise SystemExit(f"{raw_dir} does not exist")
    ffmpeg_binary()
    Handler.start_dir = raw_dir
    Handler.cache_dir = args.cache_dir.expanduser()
    Handler.library = Library(raw_dir, Handler.cache_dir)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Reviewing {raw_dir}\nOpen http://127.0.0.1:{args.port} in your browser. Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
