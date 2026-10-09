"""Conversion between simulator joint radians and LeRobot SO-101 calibrated units.

LeRobot units: -100..100 for the five arm joints, 0..100 for the gripper.
Used for recording, replay and evaluation, so the policy always sees the same units.
"""

import torch

JOINT_MINS_DEG = [-110.0, -100.0, -100.0, -95.0, -160.0, -10.0]
JOINT_MAXS_DEG = [110.0, 100.0, 90.0, 95.0, 160.0, 100.0]


def make_degree_converters(device):
    joint_mins = torch.tensor(JOINT_MINS_DEG, device=device)
    joint_maxs = torch.tensor(JOINT_MAXS_DEG, device=device)

    def sim_radians_to_real_degrees(rad_tensor):
        deg = rad_tensor * 180.0 / torch.pi
        normalized = (deg - joint_mins) / (joint_maxs - joint_mins)
        raw_deg = torch.zeros_like(normalized)
        raw_deg[:-1] = normalized[:-1] * 200.0 - 100.0
        raw_deg[-1] = normalized[-1] * 100.0
        return raw_deg

    def real_degrees_to_sim_radians(raw_deg_tensor):
        """Converts the physical leader's real servo degrees back into
        simulation radians."""
        normalized = torch.zeros_like(raw_deg_tensor)
        normalized[:-1] = (raw_deg_tensor[:-1] + 100.0) / 200.0
        normalized[-1] = raw_deg_tensor[-1] / 100.0
        deg = normalized * (joint_maxs - joint_mins) + joint_mins
        return deg * torch.pi / 180.0

    return sim_radians_to_real_degrees, real_degrees_to_sim_radians
