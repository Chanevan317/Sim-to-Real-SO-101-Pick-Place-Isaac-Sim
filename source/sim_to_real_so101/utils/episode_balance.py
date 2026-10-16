"""Keep two-cube recordings balanced over layout x target colour.

Before each reset, teleop_so101.py asks for one of the least-recorded combinations in the output
folder. Discarded tries leave no file, so a layout or colour that keeps getting discarded is
offered again until it catches up. Deleting files (for example in the review page) is picked up
at the next reset.
"""

import glob
import os
import random
import re

import h5py

EPISODE_RE = re.compile(r"^episode_(\d+)\.h5(\.tmp)?$")
COLOR_RE = re.compile(r"\b(red|blue)\b")


class EpisodeBalancer:
    def __init__(self, output_dir, layouts, cube_colors):
        self.output_dir = output_dir
        colors = sorted(set(cube_colors.values()))
        self.cells = [(layout, color) for layout in layouts for color in colors]
        self.cube_for_color = {color: cube for cube, color in cube_colors.items()}
        self._cache = {}  # file name -> (layout, color), read once per file

    def _cell(self, path):
        name = os.path.basename(path)
        if name not in self._cache:
            try:
                with h5py.File(path, "r") as f:
                    layout = f.attrs.get("layout")
                    instruction = f.attrs.get("instruction", "")
                layout = layout.decode() if isinstance(layout, bytes) else layout
                instruction = instruction.decode() if isinstance(instruction, bytes) else instruction
                color = COLOR_RE.search(instruction or "")
                self._cache[name] = (layout, color.group(1) if color else None)
            except OSError:  # being written right now: count it at the next reset
                return None
        return self._cache[name]

    def counts(self):
        """Saved episodes per (layout, colour); an episode still compressing counts once."""
        counts = {cell: 0 for cell in self.cells}
        seen = set()
        for path in sorted(glob.glob(os.path.join(self.output_dir, "episode_*.h5*"))):
            m = EPISODE_RE.match(os.path.basename(path))
            if not m or m.group(1) in seen:
                continue
            cell = self._cell(path)
            if cell in counts:
                counts[cell] += 1
                seen.add(m.group(1))
        return counts

    def next(self):
        """(layout, colour, counts): a random pick among the least-recorded combinations."""
        counts = self.counts()
        fewest = min(counts.values())
        layout, color = random.choice([cell for cell, n in counts.items() if n == fewest])
        return layout, color, counts

    def summary(self, counts):
        """One line like 'camera_side 82/81 · far_side 7/7 · plate_middle 36/35 (blue/red)'."""
        colors = sorted({c for _, c in self.cells})
        layouts = list(dict.fromkeys(layout for layout, _ in self.cells))
        parts = [f"{layout} " + "/".join(str(counts[(layout, c)]) for c in colors) for layout in layouts]
        return " · ".join(parts) + f" ({'/'.join(colors)})"
