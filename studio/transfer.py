"""
transfer.py

Sending a converted dataset to the training computer, for SO-Arm Data Studio:
  - SSH (rsync): when this computer can reach the training computer, with key login set up by the admin.
  - croc: works between any two computers with internet. This computer sends, shows a secret code,
    and the training computer receives with that code. croc can be installed from the app,
    without sudo, into ~/.local/bin.
"""

import hashlib
import io
import json
import platform
import re
import secrets
import shlex
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

LOCAL_BIN = Path.home() / ".local" / "bin"
CROC_RELEASES = "https://api.github.com/repos/schollz/croc/releases/latest"
CROC_FALLBACK_TAG = "v11.5.4"  # used when the GitHub API cannot be asked for the newest one
CROC_ARCH = {"x86_64": "64bit", "amd64": "64bit", "aarch64": "ARM64", "arm64": "ARM64"}
HOST_RE = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9][A-Za-z0-9_.:-]*$")
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new"]
SSH_CMD = ["ssh"]  # replaced in tests

# Easy words for the croc secret: read aloud or typed on the other computer.
WORDS = """
apple anchor arrow badge basil beacon berry blade bloom bolt brick bridge cabin candle canyon carbon
cedar cherry cliff cloud clover cobalt comet copper coral cotton crane crystal daisy delta desert
dolphin dragon eagle ember falcon fern field flame forest fossil garden ginger glacier granite harbor
hazel honey island ivory jade jungle kettle lantern lemon lily lotus maple marble meadow melon
meteor mint mirror nectar noble ocean olive onyx orbit orchid otter panda pearl pepper pilot
planet plum polar pumpkin quartz rabbit radar raven river rocket ruby saddle salmon silver sky
spark spruce stone summit sunset tiger timber topaz tulip valley velvet violet walnut willow zebra
""".split()


# ---------------------------------------------------------------------------
# croc
# ---------------------------------------------------------------------------
def croc_path():
    found = shutil.which("croc")
    if found:
        return found
    local = LOCAL_BIN / "croc"  # the app's own install; the desktop launcher's PATH may not include it
    return str(local) if local.is_file() else None


def croc_version(path=None):
    path = path or croc_path()
    if not path:
        return None
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"(\d+\.\d+\.\d+)", out)
    return match.group(1) if match else out.strip() or "unknown"


def _download(url, timeout=60):
    request = urllib.request.Request(url, headers={"User-Agent": "so-arm-data-studio"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def install_croc(dest_dir=LOCAL_BIN):
    """Download the croc release for this computer from GitHub, check its SHA-256 against the
    release's checksum file, and put the program in dest_dir. Returns the installed path."""
    arch = CROC_ARCH.get(platform.machine().lower())
    if platform.system() != "Linux" or not arch:
        raise ValueError(f"No croc download for {platform.system()} {platform.machine()}: ask the admin.")
    try:
        tag = json.loads(_download(CROC_RELEASES, timeout=15))["tag_name"]
    except Exception:
        tag = CROC_FALLBACK_TAG
    base = f"https://github.com/schollz/croc/releases/download/{tag}"
    archive_name = f"croc_{tag}_Linux-{arch}.tar.gz"
    try:
        sums = _download(f"{base}/croc_{tag}_checksums.txt").decode()
        archive = _download(f"{base}/{archive_name}", timeout=180)
    except Exception as e:
        raise ValueError(f"Could not download croc (is this computer online?): {e}")
    expected = next((line.split()[0] for line in sums.splitlines() if line.strip().endswith(archive_name)), None)
    if not expected or hashlib.sha256(archive).hexdigest() != expected:
        raise ValueError("The croc download did not match its checksum: not installed. Try again later.")
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        member = next((m for m in tar.getmembers() if m.isfile() and Path(m.name).name == "croc"), None)
        if member is None:
            raise ValueError("The croc download has no croc program in it: not installed.")
        data = tar.extractfile(member).read()
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / "croc.part"
    tmp.write_bytes(data)
    tmp.chmod(0o755)
    tmp.replace(dest_dir / "croc")
    return str(dest_dir / "croc")


def new_croc_code():
    """A fresh secret for one transfer, like 4821-maple-otter-quartz."""
    return f"{secrets.randbelow(10000):04d}-" + "-".join(secrets.choice(WORDS) for _ in range(3))


def croc_send_cmd(croc, dataset_dir, relay=None):
    """The code goes in CROC_SECRET (croc refuses it on the command line on Linux)."""
    cmd = [croc, "--yes", "--disable-clipboard"]
    if relay:
        cmd += ["--relay", relay]
    return cmd + ["send", str(dataset_dir)]


def croc_receive_cmd(code, relay=None):
    """What to type on the training computer, inside the folder that should receive the dataset."""
    relay_arg = f" --relay {shlex.quote(relay)}" if relay else ""
    return f"CROC_SECRET={shlex.quote(code)} croc --yes{relay_arg}"


# ---------------------------------------------------------------------------
# SSH
# ---------------------------------------------------------------------------
def check_target(host, remote_dir):
    host, remote_dir = str(host or "").strip(), str(remote_dir or "").strip()
    if not HOST_RE.match(host):
        raise ValueError("Type the training computer as user@address, for example user@192.168.0.10.")
    if not remote_dir or remote_dir.startswith("-") or "\n" in remote_dir:
        raise ValueError("Type the folder on the training computer that should receive the dataset.")
    return host, remote_dir.rstrip("/") or "/"


def _remote_path(remote_dir):
    """Quote a remote path for the remote shell, keeping a leading ~/ so it means that user's home."""
    if remote_dir == "~" or remote_dir.startswith("~/"):
        return "~/" + shlex.quote(remote_dir[2:] or ".")
    return shlex.quote(remote_dir)


def ssh_test(host, remote_dir, name):
    """Log in without a password and look at the receiving folder. Returns a dict for the page."""
    host, remote_dir = check_target(host, remote_dir)
    folder = _remote_path(remote_dir)
    script = (f"cd {folder} 2>/dev/null || {{ echo NOFOLDER; exit 0; }}; "
              f"command -v rsync >/dev/null || echo NORSYNC; "
              f"[ -e {shlex.quote(name)} ] && echo EXISTS; "
              f"echo FREE $(df -Pk . | awk 'NR==2 {{print $4}}')")
    try:
        result = subprocess.run(SSH_CMD + SSH_OPTS + [host, script], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return {"ok": False, "message": "The training computer did not answer. Is it on and reachable from here?"}
    if result.returncode != 0:
        err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else f"exit code {result.returncode}"
        if "Permission denied" in err:
            hint = "This computer is not allowed to log in without a password: ask the admin to set up the SSH key (ssh-copy-id)."
        elif "Could not resolve" in err or "No route" in err or "timed out" in err or "refused" in err:
            hint = "The training computer cannot be reached from here. If it is on another network, use croc."
        else:
            hint = "Could not log in to the training computer."
        return {"ok": False, "message": f"{hint} ({err})"}
    out = result.stdout.split()
    if "NOFOLDER" in out:
        return {"ok": False, "message": f"The folder {remote_dir} does not exist on the training computer."}
    if "NORSYNC" in out:
        return {"ok": False, "message": "rsync is not installed on the training computer: ask the admin (sudo apt install rsync)."}
    free_gb = None
    if "FREE" in out and out.index("FREE") + 1 < len(out):
        try:
            free_gb = int(out[out.index("FREE") + 1]) / 1e6
        except ValueError:
            pass
    return {"ok": True, "exists": "EXISTS" in out, "free_gb": free_gb,
            "message": "Connected." + (f" {free_gb:.0f} GB free there." if free_gb is not None else "")
                       + (f" {name} is already there: sending again only adds what is missing or changed." if "EXISTS" in out else "")}


def rsync_cmd(dataset_dir, host, remote_dir):
    host, remote_dir = check_target(host, remote_dir)
    ssh = " ".join(shlex.quote(a) for a in SSH_CMD + SSH_OPTS)
    if remote_dir == "~" or remote_dir.startswith("~/"):  # rsync paths without a leading / start in the home folder
        remote_dir = remote_dir[2:] or "."
    # No trailing slash on the source: the folder itself is created inside remote_dir.
    return ["rsync", "-a", "--partial", "--info=progress2", "--no-inc-recursive", "-e", ssh,
            str(dataset_dir), f"{host}:{remote_dir}/"]


def folder_size(path):
    files = [p for p in Path(path).rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)
