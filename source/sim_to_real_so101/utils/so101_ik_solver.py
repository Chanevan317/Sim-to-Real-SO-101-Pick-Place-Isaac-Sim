"""Shared Cartesian-delta-to-joint-target IK solver for SO-101.

Extracted from oracle_action.py's validated pipeline so both the autonomous oracle
and the teleop script use IDENTICAL, already-debugged math (base-frame Jacobian
rotation, TCP offset compensation, joint clamping) rather than two copies that can
silently drift apart.
"""

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat, quat_inv


class SO101IKSolver:
    ARM_JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
    GRIPPER_JOINT_NAME = "Jaw"
    EE_BODY_NAME = "gripper"
    TCP_LOCAL_OFFSET = (0.0101, 0.0094, -0.0117)  # confirmed match to ee_frame FrameTransformerCfg

    # Real Jaw hardware range from SO-ARM101-USD.usda: -0.1745 to 1.745 rad.
    GRIPPER_OPEN = 0.8
    GRIPPER_CLOSED = -0.15

    WRIST_PITCH_SAFE_MAX = 1.0  # UNVERIFIED guess - camera collision avoidance

    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device
        ik_cfg = DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.05},
        )
        self.ik_controller = DifferentialIKController(ik_cfg, num_envs=num_envs, device=device)
        self._initialized = False

    def reset_idx(self, env_ids):
        if self._initialized:
            self.ik_controller.reset(env_ids)

    def _lazy_init(self, env):
        robot = env.unwrapped.scene["robot"]
        arm_ids, _ = robot.find_joints(self.ARM_JOINT_NAMES)
        gripper_ids, _ = robot.find_joints([self.GRIPPER_JOINT_NAME])
        body_ids, _ = robot.find_bodies(self.EE_BODY_NAME)

        self.arm_joint_ids = arm_ids
        self.gripper_joint_id = gripper_ids[0]
        self.ee_body_id = body_ids[0]
        self.ee_jacobi_idx = self.ee_body_id - 1
        self.wrist_pitch_local_idx = self.ARM_JOINT_NAMES.index("Wrist_Pitch")
        self._initialized = True

    def get_current_ee_pose_w(self, env):
        """Returns the current TCP world position and the wrist body's world quaternion."""
        robot = env.unwrapped.scene["robot"]
        if not self._initialized:
            self._lazy_init(env)
        ee_frame = env.unwrapped.scene["ee_frame"]
        ee_tip_pos_w = ee_frame.data.target_pos_w[:, 0, :]
        ee_pose_w = robot.data.body_state_w[:, self.ee_body_id, :7]
        return ee_tip_pos_w, ee_pose_w[:, 3:7]

    def solve(self, env, target_tip_pos_w: torch.Tensor, max_step_rad: float = 0.06) -> torch.Tensor:
        """Given a desired TCP world position, returns joint targets (5-dim, arm only)."""
        robot = env.unwrapped.scene["robot"]
        if not self._initialized:
            self._lazy_init(env)

        jacobian_w = robot.root_physx_view.get_jacobians()[
            :, self.ee_jacobi_idx, :, self.arm_joint_ids
        ]
        ee_pose_w = robot.data.body_state_w[:, self.ee_body_id, :7]
        root_pose_w = robot.data.root_state_w[:, :7]

        base_rot_matrix = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))
        jacobian = jacobian_w.clone()
        jacobian[:, 0:3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 0:3, :])
        jacobian[:, 3:6, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:6, :])

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], ee_pose_w[:, :3], ee_pose_w[:, 3:7]
        )
        current_arm_joints = robot.data.joint_pos[:, self.arm_joint_ids]

        ee_frame = env.unwrapped.scene["ee_frame"]
        ee_tip_pos_w = ee_frame.data.target_pos_w[:, 0, :]
        tcp_offset_w = ee_tip_pos_w - ee_pose_w[:, :3]

        target_pos_w_body = target_tip_pos_w - tcp_offset_w
        target_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], target_pos_w_body, ee_quat_b
        )

        self.ik_controller.set_command(target_pos_b, ee_quat=ee_quat_b)
        joint_pos_des = self.ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, current_arm_joints)

        delta = torch.clamp(joint_pos_des - current_arm_joints, -max_step_rad, max_step_rad)
        joint_pos_des = current_arm_joints + delta

        joint_limits = robot.data.soft_joint_pos_limits[:, self.arm_joint_ids, :].clone()
        joint_limits[:, self.wrist_pitch_local_idx, 1] = torch.clamp(
            joint_limits[:, self.wrist_pitch_local_idx, 1], max=self.WRIST_PITCH_SAFE_MAX
        )
        joint_pos_des = torch.clamp(joint_pos_des, joint_limits[..., 0], joint_limits[..., 1])

        return joint_pos_des