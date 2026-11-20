"""Franka Panda motion planning and physical grasp execution controller for ManiSkill 3."""

from __future__ import annotations

import time
from typing import Any

import mplib
import numpy as np
import sapien
from scipy.spatial.transform import Rotation

from omnigda.core.gda_math import rotation_matrix_to_quat_xyzw
from omnigda.core.schemas import ExecutionResult, GraspCandidate
from omnigda.sim.clutter_env import PANDA_READY_QPOS, ClutteredTabletopSim

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSED = -1.0


def _orthogonalize_rotation(rot: np.ndarray) -> np.ndarray:
    """Project a 3x3 matrix onto SO(3) via SVD to eliminate numerical drift."""
    u, _, vt = np.linalg.svd(np.asarray(rot, dtype=np.float64))
    r = u @ vt
    if np.linalg.det(r) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def _matrix_to_sapien_pose(mat_4x4: np.ndarray) -> sapien.Pose:
    """Convert a 4x4 homogeneous transform to a sapien.Pose (p, q_wxyz)."""
    pos = np.asarray(mat_4x4[:3, 3], dtype=np.float64)
    rot = _orthogonalize_rotation(mat_4x4[:3, :3])
    quat_xyzw = Rotation.from_matrix(rot).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
    return sapien.Pose(p=pos, q=quat_wxyz)


def _pose_to_mplib_vec(pose: sapien.Pose) -> np.ndarray:
    """Convert sapien.Pose to mplib's 7D [x, y, z, qw, qx, qy, qz] pose vector."""
    return np.concatenate([np.asarray(pose.p, dtype=np.float64), np.asarray(pose.q, dtype=np.float64)])


def _flip_gripper_z_180(pose: sapien.Pose) -> sapien.Pose:
    """Rotate a parallel-jaw TCP pose by 180 deg about its local +Z approach axis."""
    rot_flip = sapien.Pose(q=[0.0, 0.0, 0.0, 1.0])  # w=0, x=0, y=0, z=1 -> 180 deg around Z
    return pose * rot_flip


def build_top_down_grasp(
    position: np.ndarray,
    yaw_rad: float = 0.0,
    width_m: float = 0.05,
    target_class_name: str = "",
    target_instance_id: int = -1,
    score: float = 1.0,
) -> GraspCandidate:
    """Build a Panda-convention top-down GraspCandidate at world `position`.

    In ManiSkill's ``panda_hand_tcp`` frame:
        - +Z is the approaching direction (pointing straight down ``[0, 0, -1]``)
        - +Y is the gripper closing axis (rotated by ``yaw_rad`` in the horizontal plane)
        - +X = closing x approaching
    """
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    approaching = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    closing = np.array([-np.sin(yaw_rad), np.cos(yaw_rad), 0.0], dtype=np.float64)
    ortho = np.cross(closing, approaching)

    pose_4x4 = np.eye(4, dtype=np.float64)
    pose_4x4[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
    pose_4x4[:3, 3] = pos

    return GraspCandidate(
        pose_4x4=pose_4x4,
        position=pos.copy(),
        quaternion_xyzw=rotation_matrix_to_quat_xyzw(pose_4x4[:3, :3]),
        width_m=float(width_m),
        score=float(score),
        force_closure_score=float(score),
        antipodal_score=float(score),
        collision_free=True,
        approach_vector=approaching,
        target_instance_id=int(target_instance_id),
        target_class_name=str(target_class_name),
    )


class FrankaGraspController:
    """Motion-planned physical Pick and Pick-and-Place controller for Franka Panda in ManiSkill 3."""

    MOVE_GROUP = "panda_hand_tcp"

    def __init__(
        self,
        sim: ClutteredTabletopSim,
        pregrasp_offset_m: float = 0.10,
        lift_height_m: float = 0.16,
        min_tcp_z_m: float = 0.010,
        refine_steps: int = 8,
        grip_steps: int = 18,
        hold_steps: int = 10,
        joint_vel_scale: float = 0.85,
        joint_acc_scale: float = 0.85,
        lift_success_threshold_m: float = 0.05,
        record_frames: bool = False,
        record_stride: int = 4,
    ) -> None:
        self.sim = sim
        self.pregrasp_offset_m = float(pregrasp_offset_m)
        self.lift_height_m = float(lift_height_m)
        self.min_tcp_z_m = float(min_tcp_z_m)
        self.refine_steps = int(refine_steps)
        self.grip_steps = int(grip_steps)
        self.hold_steps = int(hold_steps)
        self.lift_success_threshold_m = float(lift_success_threshold_m)
        self.record_frames = bool(record_frames)
        self.record_stride = max(1, int(record_stride))
        self._recorded_frames: list[np.ndarray] = []
        self._exec_step_counter = 0

        self.planner = self._build_mplib_planner(
            joint_vel_scale=joint_vel_scale,
            joint_acc_scale=joint_acc_scale,
        )

    def _build_mplib_planner(
        self,
        joint_vel_scale: float,
        joint_acc_scale: float,
    ) -> mplib.Planner:
        agent = self.sim.base_env.agent
        robot = agent.robot
        link_names = [link.get_name() for link in robot.get_links()]
        joint_names = [joint.get_name() for joint in robot.get_active_joints()]
        urdf_path = agent.urdf_path
        srdf_path = urdf_path.replace(".urdf", ".srdf")

        planner = mplib.Planner(
            urdf=urdf_path,
            srdf=srdf_path,
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=self.MOVE_GROUP,
        )
        base_pose = self.sim.robot_base_pose
        planner.set_base_pose(_pose_to_mplib_vec(base_pose))
        planner.joint_vel_limits = np.asarray(planner.joint_vel_limits) * float(joint_vel_scale)
        planner.joint_acc_limits = np.asarray(planner.joint_acc_limits) * float(joint_acc_scale)
        return planner

    def get_recorded_frames(self) -> list[np.ndarray]:
        return list(self._recorded_frames)

    def clear_recorded_frames(self) -> None:
        self._recorded_frames.clear()
        self._exec_step_counter = 0

    def _maybe_record(self, enabled: bool, force: bool = False) -> None:
        if not enabled:
            return
        if force or (self._exec_step_counter % self.record_stride == 0):
            self._recorded_frames.append(self.sim.render_overview())

    def _step_arm(self, arm_qpos: np.ndarray, gripper_cmd: float, record: bool) -> None:
        action = np.concatenate([np.asarray(arm_qpos, dtype=np.float32)[:7], [np.float32(gripper_cmd)]])
        self.sim.step(action)
        self._exec_step_counter += 1
        self._maybe_record(record)

    def _follow_plan(
        self,
        plan: dict[str, Any],
        gripper_cmd: float,
        refine_steps: int,
        record: bool,
    ) -> None:
        positions = np.asarray(plan["position"], dtype=np.float64)
        n_steps = positions.shape[0]
        for idx in range(n_steps + refine_steps):
            qpos = positions[min(idx, n_steps - 1)]
            self._step_arm(qpos, gripper_cmd=gripper_cmd, record=record)

    def _set_gripper(self, gripper_cmd: float, steps: int, record: bool) -> None:
        qpos = self.sim.get_robot_qpos()[:7]
        for _ in range(steps):
            self._step_arm(qpos, gripper_cmd=gripper_cmd, record=record)

    def _plan_pose(
        self,
        target_pose: sapien.Pose,
        prefer_screw: bool = True,
    ) -> dict[str, Any] | None:
        cur_qpos = self.sim.get_robot_qpos()
        pose_vec = _pose_to_mplib_vec(target_pose)
        dt = self.sim.control_timestep

        if prefer_screw:
            res = self.planner.plan_screw(
                pose_vec,
                cur_qpos,
                time_step=dt,
                use_point_cloud=False,
            )
            if res.get("status") == "Success":
                return res

        res = self.planner.plan_qpos_to_pose(
            pose_vec,
            cur_qpos,
            time_step=dt,
            use_point_cloud=False,
            wrt_world=True,
        )
        if res.get("status") == "Success":
            return res

        if not prefer_screw:
            res = self.planner.plan_screw(
                pose_vec,
                cur_qpos,
                time_step=dt,
                use_point_cloud=False,
            )
            if res.get("status") == "Success":
                return res
        return None

    def move_to_ready(self, record_frames: bool | None = None) -> bool:
        """Smoothly interpolate the Franka arm back to its unobstructed ready pose."""
        record = self.record_frames if record_frames is None else bool(record_frames)
        cur_q = self.sim.get_robot_qpos()[:7]
        target_q = PANDA_READY_QPOS[:7]
        n_steps = 30
        for step_idx in range(1, n_steps + 1):
            alpha = step_idx / n_steps
            q = (1.0 - alpha) * cur_q + alpha * target_q
            self._step_arm(q, gripper_cmd=_GRIPPER_OPEN, record=record)
        return True

    def execute_pick(
        self,
        grasp: GraspCandidate,
        target_object_name: str | None = None,
        place_position: np.ndarray | None = None,
        lift_height_m: float | None = None,
        record_frames: bool | None = None,
    ) -> ExecutionResult:
        """Execute a physical Pre-grasp -> Descend -> Close -> Lift (and optional Place) sequence."""
        t_start = time.perf_counter()
        record = self.record_frames if record_frames is None else bool(record_frames)
        lift_h = self.lift_height_m if lift_height_m is None else float(lift_height_m)

        # Resolve target object name for automated lift verification
        all_objects = self.sim.get_object_names()
        resolved_target = target_object_name or grasp.target_class_name
        if resolved_target not in all_objects:
            resolved_target = ""

        initial_positions = self.sim.get_object_positions()
        self._maybe_record(record, force=True)

        # Normalize grasp pose and clamp z above table surface
        grasp_mat = np.asarray(grasp.pose_4x4, dtype=np.float64).copy()
        grasp_mat[2, 3] = max(float(grasp_mat[2, 3]), self.min_tcp_z_m)
        grasp_pose = _matrix_to_sapien_pose(grasp_mat)

        approach = np.asarray(grasp.approach_vector, dtype=np.float64).reshape(3)
        approach_norm = float(np.linalg.norm(approach))
        if approach_norm < 1e-6:
            approach = grasp_mat[:3, 2]
            approach_norm = float(np.linalg.norm(approach))
        approach = approach / max(approach_norm, 1e-6)

        pregrasp_pos = np.asarray(grasp_pose.p, dtype=np.float64) - self.pregrasp_offset_m * approach
        pregrasp_pose = sapien.Pose(p=pregrasp_pos, q=grasp_pose.q)

        # 1. Open gripper before approaching
        self._set_gripper(_GRIPPER_OPEN, steps=6, record=record)

        # 2. Plan to pre-grasp (exploiting 180-deg parallel-jaw symmetry around TCP +Z if needed)
        plan_pre = self._plan_pose(pregrasp_pose, prefer_screw=True)
        if plan_pre is None:
            flipped_grasp = _flip_gripper_z_180(grasp_pose)
            flipped_pre = sapien.Pose(p=pregrasp_pos, q=flipped_grasp.q)
            plan_pre = self._plan_pose(flipped_pre, prefer_screw=False)
            if plan_pre is not None:
                grasp_pose = flipped_grasp
                pregrasp_pose = flipped_pre

        if plan_pre is None:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return ExecutionResult(
                success=False,
                message="Motion planning failed for pre-grasp pose.",
                target_instance_id=grasp.target_instance_id,
                target_class_name=resolved_target or grasp.target_class_name,
                grasp_candidate=grasp,
                planning_latency_ms=elapsed_ms,
            )

        self._follow_plan(plan_pre, gripper_cmd=_GRIPPER_OPEN, refine_steps=4, record=record)

        # 3. Linearly descend to grasp pose
        plan_grasp = self._plan_pose(grasp_pose, prefer_screw=True)
        if plan_grasp is None:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return ExecutionResult(
                success=False,
                message="Motion planning failed for grasp descent pose.",
                target_instance_id=grasp.target_instance_id,
                target_class_name=resolved_target or grasp.target_class_name,
                grasp_candidate=grasp,
                planning_latency_ms=elapsed_ms,
            )

        self._follow_plan(
            plan_grasp,
            gripper_cmd=_GRIPPER_OPEN,
            refine_steps=self.refine_steps,
            record=record,
        )

        # 4. Close parallel-jaw fingers firmly around the object
        self._set_gripper(_GRIPPER_CLOSED, steps=self.grip_steps, record=record)

        # 5. Lift vertically while maintaining grasp closure
        lift_pos = np.asarray(grasp_pose.p, dtype=np.float64) + np.array([0.0, 0.0, lift_h])
        lift_pose = sapien.Pose(p=lift_pos, q=grasp_pose.q)
        plan_lift = self._plan_pose(lift_pose, prefer_screw=True)
        if plan_lift is None:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return ExecutionResult(
                success=False,
                message="Motion planning failed for post-grasp lift.",
                target_instance_id=grasp.target_instance_id,
                target_class_name=resolved_target or grasp.target_class_name,
                grasp_candidate=grasp,
                planning_latency_ms=elapsed_ms,
            )

        self._follow_plan(
            plan_lift,
            gripper_cmd=_GRIPPER_CLOSED,
            refine_steps=self.hold_steps,
            record=record,
        )
        self._maybe_record(record, force=True)

        # Evaluate physical lift delta
        lifted_positions = self.sim.get_object_positions()
        if resolved_target:
            lift_delta = float(lifted_positions[resolved_target][2] - initial_positions[resolved_target][2])
            lifted_name = resolved_target
        else:
            deltas = {
                name: float(lifted_positions[name][2] - initial_positions[name][2])
                for name in initial_positions
            }
            lifted_name = max(deltas, key=lambda k: deltas[k]) if deltas else ""
            lift_delta = deltas.get(lifted_name, 0.0)

        lifted_ok = lift_delta >= self.lift_success_threshold_m

        # 6. Optional placement sequence
        placed_msg = ""
        if place_position is not None and lifted_ok:
            place_xyz = np.asarray(place_position, dtype=np.float64).reshape(3)
            above_place = sapien.Pose(
                p=np.array([place_xyz[0], place_xyz[1], max(place_xyz[2] + lift_h, 0.16)]),
                q=grasp_pose.q,
            )
            drop_pose = sapien.Pose(
                p=np.array([place_xyz[0], place_xyz[1], max(place_xyz[2] + 0.025, 0.04)]),
                q=grasp_pose.q,
            )

            plan_above = self._plan_pose(above_place, prefer_screw=True)
            if plan_above is None:
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                return ExecutionResult(
                    success=False,
                    message=f"Lifted {lifted_name} by {lift_delta:.3f}m but failed to plan transit to place.",
                    target_instance_id=grasp.target_instance_id,
                    target_class_name=lifted_name,
                    grasp_candidate=grasp,
                    planning_latency_ms=elapsed_ms,
                )
            self._follow_plan(plan_above, gripper_cmd=_GRIPPER_CLOSED, refine_steps=4, record=record)

            plan_drop = self._plan_pose(drop_pose, prefer_screw=True)
            if plan_drop is not None:
                self._follow_plan(plan_drop, gripper_cmd=_GRIPPER_CLOSED, refine_steps=6, record=record)

            # Release object and retreat upward
            self._set_gripper(_GRIPPER_OPEN, steps=12, record=record)
            plan_retreat = self._plan_pose(above_place, prefer_screw=True)
            if plan_retreat is not None:
                self._follow_plan(plan_retreat, gripper_cmd=_GRIPPER_OPEN, refine_steps=4, record=record)
            self._maybe_record(record, force=True)

            final_positions = self.sim.get_object_positions()
            xy_err = float(
                np.linalg.norm(final_positions[lifted_name][:2] - place_xyz[:2])
            )
            placed_msg = f" Placed within {xy_err:.3f}m of target."
            lifted_ok = xy_err <= 0.08

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        status = "succeeded" if lifted_ok else "failed"
        message = (
            f"Pick {status} for {lifted_name or 'object'} "
            f"(lift_delta={lift_delta:.3f}m, threshold={self.lift_success_threshold_m:.3f}m)."
            f"{placed_msg}"
        )
        return ExecutionResult(
            success=bool(lifted_ok),
            message=message,
            target_instance_id=grasp.target_instance_id,
            target_class_name=lifted_name or grasp.target_class_name,
            grasp_candidate=grasp,
            planning_latency_ms=elapsed_ms,
        )

    def execute_pick_and_place(
        self,
        grasp: GraspCandidate,
        place_position: np.ndarray,
        target_object_name: str | None = None,
        lift_height_m: float | None = None,
        record_frames: bool | None = None,
    ) -> ExecutionResult:
        """Convenience wrapper executing a full Pick-and-Place trajectory."""
        return self.execute_pick(
            grasp=grasp,
            target_object_name=target_object_name,
            place_position=place_position,
            lift_height_m=lift_height_m,
            record_frames=record_frames,
        )
