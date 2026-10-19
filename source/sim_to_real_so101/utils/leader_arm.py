"""SO-101 leader arm: read the Feetech servos and map them to sim joint targets.

Joints listed in the calibration file (made by scripts/calibrate_leader.py) are mapped
with a straight line fitted in the sim:  sim_deg = scale * wrap(raw_deg - raw_mid) + offset_deg.
The other joints keep the old mapping: the raw encoder angle is used as LeRobot units
and converted with joint_conversion.real_degrees_to_sim_radians.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch

# Motor IDs 1..6 drive these sim joints, in this order.
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "leader_calibration.json"

ADDR_STS_PRESENT_POSITION = 56


def wrap_deg(deg):
    """Wrap an angle (or array of angles) to [-180, 180)."""
    return (np.asarray(deg) + 180.0) % 360.0 - 180.0


class LeaderArm:
    """Reads the six leader-arm servos over the serial port (read only, never moves them)."""

    def __init__(self, port):
        try:
            from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler
        except ImportError as e:
            raise ImportError("Please install the Feetech SDK: `pip install feetech-servo-sdk`") from e

        self._comm_success = COMM_SUCCESS
        self._port = PortHandler(port)
        self._packet = PacketHandler(0)  # protocol 0 for STS servos
        if not self._port.openPort():
            raise RuntimeError(f"Failed to open serial port {port}")
        if not self._port.setBaudRate(1000000):
            raise RuntimeError("Failed to change the baudrate to 1000000 bps.")
        self._last = np.zeros(len(JOINT_NAMES), dtype=np.float32)
        self.last_read_ok = np.zeros(len(JOINT_NAMES), dtype=bool)  # did each motor answer the last read?

    def read_raw_deg(self):
        """Encoder angle of each motor in degrees (-180..180). A failed read keeps the last value."""
        for i in range(len(JOINT_NAMES)):
            pos, result, _ = self._packet.read2ByteTxRx(self._port, i + 1, ADDR_STS_PRESENT_POSITION)
            self.last_read_ok[i] = result == self._comm_success
            if self.last_read_ok[i]:
                self._last[i] = pos / 4096.0 * 360.0 - 180.0
        return self._last.copy()

    def close(self):
        self._port.closePort()


def fit_joint(raw_deg, sim_deg):
    """Fit sim_deg = scale * wrap(raw - raw_deg[0]) + offset_deg through the captured poses.

    raw_deg[0] must be the middle pose: angles are measured from it, so the encoder's
    -180/+180 seam never falls inside the joint's range.
    """
    raw_mid = float(raw_deg[0])
    rel = wrap_deg(np.asarray(raw_deg, dtype=np.float64) - raw_mid)
    sim = np.asarray(sim_deg, dtype=np.float64)
    scale, offset = np.polyfit(rel, sim, 1)
    max_error = float(np.max(np.abs(scale * rel + offset - sim)))
    return {"raw_mid": raw_mid, "scale": float(scale), "offset_deg": float(offset), "max_error_deg": max_error}


def load_calibration(path=DEFAULT_CALIBRATION_PATH):
    """Per-joint fits from the calibration file, or {} if there is no file."""
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())["joints"]


def save_calibration(joints, path=DEFAULT_CALIBRATION_PATH):
    Path(path).write_text(json.dumps({"joints": joints}, indent=2) + "\n")


def leader_to_sim_radians(raw_deg, calibration, real_degrees_to_sim_radians, device):
    """Sim joint targets (radians, JOINT_NAMES order) for one leader reading."""
    targets = real_degrees_to_sim_radians(torch.tensor(raw_deg, device=device))
    for i, name in enumerate(JOINT_NAMES):
        cal = calibration.get(name)
        if cal is not None:
            sim_deg = cal["scale"] * wrap_deg(raw_deg[i] - cal["raw_mid"]) + cal["offset_deg"]
            targets[i] = math.radians(float(sim_deg))
    return targets
