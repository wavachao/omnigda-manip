"""Vectorized Franka Panda parallel-jaw gripper collision checking.

TCP Coordinate Frame Convention (Franka Panda Standard)
-------------------------------------------------------
All gripper geometry is defined in the local Tool Center Point (TCP) frame:
- Origin ``(0, 0, 0)``: midpoint between the fingertips at the nominal grasp contact center.
- ``+Z`` axis: approach vector pointing forward into the object (e.g., ``(0, 0, -1)`` in
  world coordinates for a pure top-down tabletop grasp).
- ``+Y`` axis: jaw closing/opening axis between the two parallel fingers (left finger on
  ``-Y``, right finger on ``+Y``).
- ``+X`` axis: right-hand orthogonal completion ``X = Y x Z``.

Behind the TCP along ``-Z`` lie the finger bodies, the palm/actuator housing, and the
cylindrical wrist approach shaft.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from omnigda.core.schemas import GraspCandidate


@dataclass(slots=True)
class FrankaGripperGeometry:
    """Physical dimensions (in meters) of the Franka Panda parallel-jaw gripper in TCP frame."""

    finger_thickness_m: float = 0.016
    finger_width_x_m: float = 0.020
    finger_depth_m: float = 0.045
    finger_tip_extension_m: float = 0.010

    palm_width_y_m: float = 0.200
    palm_thickness_x_m: float = 0.060
    palm_depth_m: float = 0.060

    wrist_radius_m: float = 0.038
    wrist_length_m: float = 0.100

    gripper_min_width_m: float = 0.012
    gripper_max_width_m: float = 0.080

    @property
    def finger_z_max(self) -> float:
        return self.finger_tip_extension_m

    @property
    def finger_z_min(self) -> float:
        return self.finger_tip_extension_m - self.finger_depth_m

    @property
    def palm_z_max(self) -> float:
        return self.finger_z_min

    @property
    def palm_z_min(self) -> float:
        return self.finger_z_min - self.palm_depth_m

    @property
    def wrist_z_max(self) -> float:
        return self.palm_z_min

    @property
    def wrist_z_min(self) -> float:
        return self.palm_z_min - self.wrist_length_m

    @property
    def max_reach_m(self) -> float:
        """Conservative Euclidean radius from TCP origin enclosing the entire gripper."""
        r_palm = float(
            np.hypot(
                np.hypot(self.palm_thickness_x_m * 0.5, self.palm_width_y_m * 0.5),
                max(abs(self.palm_z_min), abs(self.palm_z_max)),
            )
        )
        r_wrist = float(np.hypot(self.wrist_radius_m, abs(self.wrist_z_min)))
        r_finger = float(
            np.hypot(
                np.hypot(
                    self.finger_width_x_m * 0.5,
                    self.gripper_max_width_m * 0.5 + self.finger_thickness_m,
                ),
                max(abs(self.finger_z_min), abs(self.finger_z_max)),
            )
        )
        return max(r_palm, r_wrist, r_finger) + 0.01


@dataclass(slots=True)
class CollisionCheckResult:
    """Detailed batch collision check result."""

    collision_free: np.ndarray  # (B,) bool, True = safe
    table_collision: np.ndarray  # (B,) bool, True = collides with tabletop
    obstacle_collision: np.ndarray  # (B,) bool, True = collides with obstacle points
    min_world_z: np.ndarray  # (B,) float64, lowest gripper point Z in world frame
    obstacle_hit_counts: np.ndarray  # (B,) int32, number of obstacle points inside gripper


class FrankaGripperCollisionChecker:
    """Fast vectorized collision checker for the Franka Panda parallel-jaw gripper.

    Checks candidate 6DoF grasps in batch against:
      1. Tabletop plane ``z <= table_z + table_z_margin_m`` across sampled gripper bounding
         corners and wrist cylinder rim points.
      2. Cluttered scene obstacle point cloud by batch-transforming obstacle points into
         each candidate's local TCP frame and testing inclusion against the left finger,
         right finger, palm body, and wrist approach shaft.
    """

    def __init__(
        self,
        geometry: FrankaGripperGeometry | None = None,
        table_z_margin_m: float = 0.008,
        min_collision_points: int = 1,
        collision_padding_m: float = 0.0,
        chunk_size: int = 4096,
    ) -> None:
        self.geometry = geometry if geometry is not None else FrankaGripperGeometry()
        self.table_z_margin_m = float(table_z_margin_m)
        self.min_collision_points = max(1, int(min_collision_points))
        self.collision_padding_m = max(0.0, float(collision_padding_m))
        self.chunk_size = max(256, int(chunk_size))

        # Precompute fixed palm & wrist local-frame sample points for tabletop clearance checks
        self._static_table_pts = self._build_static_table_points()
        # Precompute finger corner sign/offset template parameterized by half_w
        self._finger_x_tpl, self._finger_y_sign, self._finger_y_off, self._finger_z_tpl = (
            self._build_finger_table_templates()
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> FrankaGripperCollisionChecker:
        """Construct checker from either full YAML config or the ``grasp`` sub-dict."""
        grasp_cfg = cfg.get("grasp", cfg)
        geom = FrankaGripperGeometry(
            finger_thickness_m=float(grasp_cfg.get("finger_thickness_m", 0.016)),
            finger_width_x_m=float(grasp_cfg.get("finger_width_x_m", 0.020)),
            finger_depth_m=float(grasp_cfg.get("finger_depth_m", 0.045)),
            finger_tip_extension_m=float(grasp_cfg.get("finger_tip_extension_m", 0.010)),
            palm_width_y_m=float(grasp_cfg.get("palm_width_y_m", 0.200)),
            palm_thickness_x_m=float(grasp_cfg.get("palm_thickness_x_m", 0.060)),
            palm_depth_m=float(grasp_cfg.get("palm_depth_m", 0.060)),
            wrist_radius_m=float(grasp_cfg.get("wrist_radius_m", 0.038)),
            wrist_length_m=float(grasp_cfg.get("wrist_length_m", 0.100)),
            gripper_min_width_m=float(grasp_cfg.get("gripper_min_width_m", 0.012)),
            gripper_max_width_m=float(grasp_cfg.get("gripper_max_width_m", 0.080)),
        )
        return cls(
            geometry=geom,
            table_z_margin_m=float(grasp_cfg.get("table_z_margin_m", 0.008)),
            min_collision_points=int(grasp_cfg.get("min_collision_points", 1)),
            collision_padding_m=float(grasp_cfg.get("collision_padding_m", 0.0)),
        )

    def _build_static_table_points(self) -> np.ndarray:
        """Build local-frame corners of the palm box and rim points of the wrist cylinder."""
        g = self.geometry
        pts: list[list[float]] = []

        # 8 palm AABB corners
        px = g.palm_thickness_x_m * 0.5
        py = g.palm_width_y_m * 0.5
        for sx in (-px, px):
            for sy in (-py, py):
                for sz in (g.palm_z_min, g.palm_z_max):
                    pts.append([sx, sy, sz])

        # 16 points on wrist cylinder front and back rims (8 angles x 2 ends)
        angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        for sz in (g.wrist_z_min, g.wrist_z_max):
            for ang in angles:
                pts.append(
                    [
                        g.wrist_radius_m * float(np.cos(ang)),
                        g.wrist_radius_m * float(np.sin(ang)),
                        sz,
                    ]
                )

        return np.asarray(pts, dtype=np.float64)  # (24, 3)

    def _build_finger_table_templates(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build template arrays for the 16 corners of left and right fingers.

        For a candidate with half opening width ``hw = width_m / 2``, the 16 local finger
        corners have ``y = y_sign * (hw + y_off)``.
        """
        g = self.geometry
        fx = g.finger_width_x_m * 0.5
        ft = g.finger_thickness_m

        xs: list[float] = []
        y_signs: list[float] = []
        y_offs: list[float] = []
        zs: list[float] = []

        for side_sign in (-1.0, 1.0):  # left finger (-Y), right finger (+Y)
            for sx in (-fx, fx):
                for y_off in (0.0, ft):
                    for sz in (g.finger_z_min, g.finger_z_max):
                        xs.append(sx)
                        y_signs.append(side_sign)
                        y_offs.append(y_off)
                        zs.append(sz)

        return (
            np.asarray(xs, dtype=np.float64),
            np.asarray(y_signs, dtype=np.float64),
            np.asarray(y_offs, dtype=np.float64),
            np.asarray(zs, dtype=np.float64),
        )

    def _check_table_batch(
        self,
        rotations: np.ndarray,
        translations: np.ndarray,
        widths_m: np.ndarray,
        table_z: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized tabletop plane clearance check across 40 gripper surface points."""
        batch_size = rotations.shape[0]
        if table_z is None:
            return (
                np.zeros(batch_size, dtype=bool),
                np.full(batch_size, np.inf, dtype=np.float64),
            )

        rz = rotations[:, 2, :]  # (B, 3) world-Z projection of local axes
        tz = translations[:, 2:3]  # (B, 1)

        # Static palm + wrist points: (B, 24)
        z_static = rz @ self._static_table_pts.T + tz

        # Width-dependent finger corners: (B, 16)
        half_w = np.maximum(widths_m, 0.0) * 0.5  # (B,)
        y_fingers = self._finger_y_sign[None, :] * (
            half_w[:, None] + self._finger_y_off[None, :]
        )
        z_fingers = (
            rz[:, 0:1] * self._finger_x_tpl[None, :]
            + rz[:, 1:2] * y_fingers
            + rz[:, 2:3] * self._finger_z_tpl[None, :]
            + tz
        )

        min_z = np.minimum(z_static.min(axis=1), z_fingers.min(axis=1))
        table_colliding = min_z <= (float(table_z) + self.table_z_margin_m)
        return table_colliding, min_z

    def _check_obstacles_batch(
        self,
        rotations: np.ndarray,
        translations: np.ndarray,
        widths_m: np.ndarray,
        obstacle_xyz: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized obstacle point cloud inclusion check against fingers, palm, and wrist."""
        batch_size = rotations.shape[0]
        hit_counts = np.zeros(batch_size, dtype=np.int32)

        if obstacle_xyz is None or obstacle_xyz.size == 0:
            return np.zeros(batch_size, dtype=bool), hit_counts

        obs = np.asarray(obstacle_xyz, dtype=np.float32)
        if obs.ndim != 2 or obs.shape[1] != 3:
            raise ValueError(f"Expected obstacle_xyz shape (M, 3), got {obs.shape}")

        # Lossless broad-phase AABB filter around all candidate TCP positions
        g = self.geometry
        pad = np.float32(self.collision_padding_m)
        reach = np.float32(g.max_reach_m + self.collision_padding_m)
        t_f32 = translations.astype(np.float32, copy=False)
        r_f32 = rotations.astype(np.float32, copy=False)

        aabb_min = t_f32.min(axis=0) - reach
        aabb_max = t_f32.max(axis=0) + reach
        in_aabb = np.all((obs >= aabb_min) & (obs <= aabb_max), axis=1)
        obs = obs[in_aabb]
        if obs.shape[0] == 0:
            return np.zeros(batch_size, dtype=bool), hit_counts

        # Precompute geometry bounds in float32
        half_w = (np.maximum(widths_m, 0.0) * 0.5).astype(np.float32)[:, None]  # (B, 1)
        finger_half_x = np.float32(g.finger_width_x_m * 0.5) + pad
        finger_thick = np.float32(g.finger_thickness_m) + pad
        finger_z_min = np.float32(g.finger_z_min) - pad
        finger_z_max = np.float32(g.finger_z_max) + pad

        palm_half_x = np.float32(g.palm_thickness_x_m * 0.5) + pad
        palm_half_y = np.float32(g.palm_width_y_m * 0.5) + pad
        palm_z_min = np.float32(g.palm_z_min) - pad
        palm_z_max = np.float32(g.palm_z_max) + pad

        wrist_r_sq = np.float32((g.wrist_radius_m + self.collision_padding_m) ** 2)
        wrist_z_min = np.float32(g.wrist_z_min) - pad
        wrist_z_max = np.float32(g.wrist_z_max) + pad

        # Pre-extract contiguous local-axis rows so each coordinate (lx, ly, lz) is
        # computed as a contiguous (B, M_c) matrix via BLAS SGEMM with zero transposes:
        # p_local_k = (R[:, :, k] @ chunk.T) - (t . R[:, :, k])
        rx = np.ascontiguousarray(r_f32[:, :, 0])  # (B, 3)
        ry = np.ascontiguousarray(r_f32[:, :, 1])  # (B, 3)
        rz = np.ascontiguousarray(r_f32[:, :, 2])  # (B, 3)
        tx = np.sum(t_f32 * rx, axis=1, keepdims=True)  # (B, 1)
        ty = np.sum(t_f32 * ry, axis=1, keepdims=True)  # (B, 1)
        tz = np.sum(t_f32 * rz, axis=1, keepdims=True)  # (B, 1)

        num_pts = obs.shape[0]
        for start in range(0, num_pts, self.chunk_size):
            chunk_t = np.ascontiguousarray(obs[start : start + self.chunk_size].T)  # (3, M_c)

            lx = rx @ chunk_t
            lx -= tx
            ly = ry @ chunk_t
            ly -= ty
            lz = rz @ chunk_t
            lz -= tz

            abs_lx = np.abs(lx)
            abs_ly = np.abs(ly)

            # 1 & 2. Left finger (-Y) and right finger (+Y) boxes (symmetric in |y|)
            in_finger_xz = (abs_lx <= finger_half_x) & (lz >= finger_z_min) & (lz <= finger_z_max)
            in_left_finger = in_finger_xz & (ly <= -half_w) & (ly >= -(half_w + finger_thick))
            in_right_finger = in_finger_xz & (ly >= half_w) & (ly <= (half_w + finger_thick))

            # 3. Palm / hand body box
            in_palm = (
                (abs_lx <= palm_half_x)
                & (abs_ly <= palm_half_y)
                & (lz >= palm_z_min)
                & (lz <= palm_z_max)
            )

            # 4. Wrist / approach shaft cylinder
            in_wrist = (
                (lz >= wrist_z_min)
                & (lz <= wrist_z_max)
                & ((lx * lx + ly * ly) <= wrist_r_sq)
            )

            in_any = in_left_finger | in_right_finger | in_palm | in_wrist
            hit_counts += np.count_nonzero(in_any, axis=1).astype(np.int32)

        obstacle_colliding = hit_counts >= self.min_collision_points
        return obstacle_colliding, hit_counts

    def check_poses_batch_detailed(
        self,
        poses_4x4: np.ndarray,
        widths_m: np.ndarray,
        obstacle_xyz: np.ndarray | None = None,
        table_z: float | None = 0.0,
    ) -> CollisionCheckResult:
        """Evaluate batch of raw 4x4 TCP poses and jaw widths with detailed diagnostics."""
        poses = np.asarray(poses_4x4, dtype=np.float64)
        widths = np.asarray(widths_m, dtype=np.float64)

        if poses.size == 0:
            empty_b = np.zeros((0,), dtype=bool)
            return CollisionCheckResult(
                collision_free=empty_b,
                table_collision=empty_b.copy(),
                obstacle_collision=empty_b.copy(),
                min_world_z=np.zeros((0,), dtype=np.float64),
                obstacle_hit_counts=np.zeros((0,), dtype=np.int32),
            )

        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(f"Expected poses_4x4 shape (B, 4, 4), got {poses.shape}")
        if widths.shape != (poses.shape[0],):
            raise ValueError(f"Expected widths_m shape ({poses.shape[0]},), got {widths.shape}")

        rotations = poses[:, :3, :3]
        translations = poses[:, :3, 3]

        table_colliding, min_z = self._check_table_batch(rotations, translations, widths, table_z)
        obstacle_colliding, hit_counts = self._check_obstacles_batch(
            rotations, translations, widths, obstacle_xyz
        )
        collision_free = ~(table_colliding | obstacle_colliding)

        return CollisionCheckResult(
            collision_free=collision_free,
            table_collision=table_colliding,
            obstacle_collision=obstacle_colliding,
            min_world_z=min_z,
            obstacle_hit_counts=hit_counts,
        )

    def check_poses_batch(
        self,
        poses_4x4: np.ndarray,
        widths_m: np.ndarray,
        obstacle_xyz: np.ndarray | None = None,
        table_z: float | None = 0.0,
    ) -> np.ndarray:
        """Return boolean array ``(B,)`` where ``True`` means collision-free."""
        return self.check_poses_batch_detailed(
            poses_4x4, widths_m, obstacle_xyz=obstacle_xyz, table_z=table_z
        ).collision_free

    def check_batch_detailed(
        self,
        candidates: Sequence[GraspCandidate],
        obstacle_xyz: np.ndarray | None = None,
        table_z: float | None = 0.0,
    ) -> CollisionCheckResult:
        """Evaluate a sequence of ``GraspCandidate`` objects with detailed collision outputs."""
        if len(candidates) == 0:
            empty_b = np.zeros((0,), dtype=bool)
            return CollisionCheckResult(
                collision_free=empty_b,
                table_collision=empty_b.copy(),
                obstacle_collision=empty_b.copy(),
                min_world_z=np.zeros((0,), dtype=np.float64),
                obstacle_hit_counts=np.zeros((0,), dtype=np.int32),
            )

        poses = np.stack([c.pose_4x4 for c in candidates], axis=0).astype(np.float64, copy=False)
        widths = np.asarray([c.width_m for c in candidates], dtype=np.float64)
        return self.check_poses_batch_detailed(
            poses, widths, obstacle_xyz=obstacle_xyz, table_z=table_z
        )

    def check_batch(
        self,
        candidates: Sequence[GraspCandidate],
        obstacle_xyz: np.ndarray | None = None,
        table_z: float | None = 0.0,
    ) -> np.ndarray:
        """Check collisions for a batch of grasp candidates.

        Args:
            candidates: Sequence of ``GraspCandidate`` instances.
            obstacle_xyz: Optional ``(M, 3)`` obstacle point cloud in world coordinates.
            table_z: Height of the supporting tabletop plane in world Z, or ``None`` to
                skip tabletop collision checking.

        Returns:
            Boolean NumPy array of shape ``(len(candidates),)`` where ``True`` indicates
            the grasp is collision-free and ``False`` indicates collision with either the
            tabletop plane or scene obstacles.
        """
        return self.check_batch_detailed(
            candidates, obstacle_xyz=obstacle_xyz, table_z=table_z
        ).collision_free

    def check_single(
        self,
        candidate: GraspCandidate,
        obstacle_xyz: np.ndarray | None = None,
        table_z: float | None = 0.0,
    ) -> bool:
        """Check whether a single ``GraspCandidate`` is collision-free."""
        return bool(self.check_batch([candidate], obstacle_xyz=obstacle_xyz, table_z=table_z)[0])
