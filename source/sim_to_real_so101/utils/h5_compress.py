"""Compress recorded episodes in the background, so the sim never waits for gzip.

teleop_so101.py writes each episode uncompressed to episode_XXXX.h5.tmp and queues it here.
The camera images are gzip-compressed into episode_XXXX.h5.part, which is renamed to
episode_XXXX.h5 only when complete, so the converter never sees a half-written file.

Compress leftover .tmp files by hand (for example after a crash):
    python -m sim_to_real_so101.utils.h5_compress raw_teleop_data
"""

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import h5py

TMP_SUFFIX = ".tmp"
# Frames copied per write, to keep memory use low.
FRAMES_PER_WRITE = 64


def compress_episode(tmp_path):
    """episode_XXXX.h5.tmp -> episode_XXXX.h5, with gzip-compressed camera images."""
    tmp_path = Path(tmp_path)
    final_path = tmp_path.with_suffix("")  # drop ".tmp"
    part_path = final_path.with_name(final_path.name + ".part")
    with h5py.File(tmp_path, "r") as src, h5py.File(part_path, "w") as dst:
        dst.attrs.update(src.attrs)
        for name, data in src.items():
            if not name.startswith("observation.images."):
                dst.create_dataset(name, data=data[()])
                continue
            out = dst.create_dataset(
                name, shape=data.shape, dtype=data.dtype,
                compression="gzip",
                compression_opts=4,  # Level 4 balances compression speed and file reduction
                chunks=True,         # Required for compression to work efficiently
            )
            for start in range(0, data.shape[0], FRAMES_PER_WRITE):
                out[start:start + FRAMES_PER_WRITE] = data[start:start + FRAMES_PER_WRITE]
    os.replace(part_path, final_path)
    tmp_path.unlink()


class CompressionQueue:
    """Runs compress_episode in separate processes, a few at a time, in submission order."""

    def __init__(self, num_workers=2):
        self._queue = queue.Queue()
        for _ in range(num_workers):
            threading.Thread(target=self._work, daemon=True).start()

    def submit(self, tmp_path):
        self._queue.put(str(tmp_path))

    def pending(self):
        return self._queue.unfinished_tasks

    def wait(self):
        self._queue.join()

    def _work(self):
        while True:
            tmp_path = self._queue.get()
            # A new session keeps Ctrl+C in the terminal from killing the compression.
            result = subprocess.run(
                [sys.executable, "-m", "sim_to_real_so101.utils.h5_compress", tmp_path],
                start_new_session=True,
            )
            if result.returncode != 0:
                print(f"\n[ERROR]: Compressing {tmp_path} failed; the uncompressed file is kept.")
            self._queue.task_done()


def main(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob(f"episode_*.h5{TMP_SUFFIX}")) if p.is_dir() else [p]
    for f in files:
        compress_episode(f)
        print(f"Compressed {f.with_suffix('')}")


if __name__ == "__main__":
    main(sys.argv[1:])
