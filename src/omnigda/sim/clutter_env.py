"""Deterministic ManiSkill 3 cluttered tabletop environment for OmniGDA-Manip."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

from omnigda.core.schemas import CameraIntrinsics, RGBDFrame

# Retracted ready joint configuration so the Franka arm does not occlude tabletop cameras
# while remaining well inside joint limits for fast mplib planning.
PANDA_READY_QPOS = np.array(
    [0.0, -0.32, 0.0, -2.28, 0.0, 1.96, np.pi / 4, 0.04, 0.04],
    dtype=np.float64,
)

PANDA_BASE_POSITION = (-0.615, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class ObjectSpec:
    """Physical and visual specification of a tabletop manipulation object."""

    name: str
    category: str  # "base" or "novel"
    color: tuple[float, float, float, float]
    z_offset: float
    footprint_radius: float
    grasp_width_m: float
    mass_kg: float
    description: str


OBJECT_CATALOG: dict[str, ObjectSpec] = {
    "red_mug": ObjectSpec(
        name="red_mug",
        category="base",
        color=(0.85, 0.14, 0.14, 1.0),
        z_offset=0.025,
        footprint_radius=0.036,
        grasp_width_m=0.044,
        mass_kg=0.12,
        description="red cylindrical mug with a side handle",
    ),
    "blue_bottle": ObjectSpec(
        name="blue_bottle",
        category="base",
        color=(0.12, 0.32, 0.85, 1.0),
        z_offset=0.042,
        footprint_radius=0.028,
        grasp_width_m=0.044,
        mass_kg=0.14,
        description="tall blue cylindrical bottle with a white cap",
    ),
    "green_box": ObjectSpec(
        name="green_box",
        category="base",
        color=(0.12, 0.72, 0.24, 1.0),
        z_offset=0.019,
        footprint_radius=0.032,
        grasp_width_m=0.036,
        mass_kg=0.10,
        description="green rectangular box",
    ),
    "yellow_block": ObjectSpec(
        name="yellow_block",
        category="base",
        color=(0.95, 0.80, 0.10, 1.0),
        z_offset=0.022,
        footprint_radius=0.034,
        grasp_width_m=0.044,
        mass_kg=0.10,
        description="yellow square wooden block",
    ),
    "orange_cylinder": ObjectSpec(
        name="orange_cylinder",
        category="base",
        color=(0.95, 0.45, 0.08, 1.0),
        z_offset=0.032,
        footprint_radius=0.026,
        grasp_width_m=0.042,
        mass_kg=0.11,
        description="orange upright cylindrical can",
    ),
    "purple_bar": ObjectSpec(
        name="purple_bar",
        category="base",
        color=(0.56, 0.18, 0.78, 1.0),
        z_offset=0.016,
        footprint_radius=0.052,
        grasp_width_m=0.032,
        mass_kg=0.11,
        description="elongated purple rectangular bar",
    ),
    "gray_tray": ObjectSpec(
        name="gray_tray",
        category="base",
        color=(0.52, 0.54, 0.58, 1.0),
        z_offset=0.010,
        footprint_radius=0.054,
        grasp_width_m=0.056,
        mass_kg=0.14,
        description="shallow metallic gray rectangular tray",
    ),
    "cyan_t_tool": ObjectSpec(
        name="cyan_t_tool",
        category="novel",
        color=(0.08, 0.82, 0.86, 1.0),
        z_offset=0.013,
        footprint_radius=0.048,
        grasp_width_m=0.026,
        mass_kg=0.10,
        description="bright cyan T-shaped tool",
    ),
    "magenta_cube": ObjectSpec(
        name="magenta_cube",
        category="novel",
        color=(0.88, 0.14, 0.68, 1.0),
        z_offset=0.019,
        footprint_radius=0.030,
        grasp_width_m=0.038,
        mass_kg=0.09,
        description="vivid magenta cube with a light stripe on top",
    ),
}

DEFAULT_BASE_OBJECTS: tuple[str, ...] = (
    "red_mug",
    "blue_bottle",
    "green_box",
    "yellow_block",
    "orange_cylinder",
    "purple_bar",
    "gray_tray",
)

HELD_OUT_NOVEL_OBJECTS: tuple[str, ...] = (
    "cyan_t_tool",
    "magenta_cube",
)


def _upright_cylinder_pose(z: float = 0.0) -> sapien.Pose:
    """In PhysX/SAPIEN cylinders extend along local +X; rotate by +pi/2 around Y to stand along Z."""
    return sapien.Pose(p=[0.0, 0.0, z], q=euler2quat(0.0, np.pi / 2.0, 0.0))


def _make_render_material(
    color: Sequence[float],
    roughness: float = 0.45,
    metallic: float = 0.05,
) -> sapien.render.RenderMaterial:
    return sapien.render.RenderMaterial(
        base_color=list(color),
        roughness=roughness,
        metallic=metallic,
    )


def build_catalog_object(
    scene: Any,
    spec: ObjectSpec,
    initial_pose: sapien.Pose | None = None,
    body_type: str = "dynamic",
) -> Actor:
    """Construct a SAPIEN/ManiSkill Actor from an ObjectSpec with realistic friction & inertia."""
    builder = scene.create_actor_builder()
    physx_mat = sapien.pysapien.physx.PhysxMaterial(
        static_friction=2.5,
        dynamic_friction=1.8,
        restitution=0.05,
    )
    main_mat = _make_render_material(
        spec.color,
        roughness=0.35 if spec.name == "gray_tray" else 0.45,
        metallic=0.35 if spec.name == "gray_tray" else 0.05,
    )
    add_collision = body_type != "kinematic"
    name = spec.name

    if name == "red_mug":
        radius, half_h = 0.022, 0.025
        cyl_pose = _upright_cylinder_pose(0.0)
        builder.add_cylinder_visual(
            pose=cyl_pose, radius=radius, half_length=half_h, material=main_mat
        )
        dark_mat = _make_render_material((0.18, 0.04, 0.04, 1.0), roughness=0.7)
        builder.add_cylinder_visual(
            pose=_upright_cylinder_pose(half_h - 0.001),
            radius=radius * 0.78,
            half_length=0.0015,
            material=dark_mat,
        )
        handle_parts = [
            (sapien.Pose(p=[radius + 0.006, 0.0, 0.012]), [0.008, 0.0045, 0.0045]),
            (sapien.Pose(p=[radius + 0.006, 0.0, -0.012]), [0.008, 0.0045, 0.0045]),
            (sapien.Pose(p=[radius + 0.012, 0.0, 0.0]), [0.0045, 0.0045, 0.016]),
        ]
        for h_pose, h_half in handle_parts:
            builder.add_box_visual(pose=h_pose, half_size=h_half, material=main_mat)
        if add_collision:
            builder.add_cylinder_collision(
                pose=cyl_pose, radius=radius, half_length=half_h, material=physx_mat
            )
            for h_pose, h_half in handle_parts:
                builder.add_box_collision(pose=h_pose, half_size=h_half, material=physx_mat)

    elif name == "blue_bottle":
        body_r, body_hh = 0.022, 0.032
        neck_r, neck_hh = 0.012, 0.010
        cap_r, cap_hh = 0.013, 0.004
        body_pose = _upright_cylinder_pose(-0.010)
        neck_pose = _upright_cylinder_pose(0.028)
        cap_pose = _upright_cylinder_pose(0.041)
        cap_mat = _make_render_material((0.94, 0.95, 0.97, 1.0), roughness=0.35)

        builder.add_cylinder_visual(
            pose=body_pose, radius=body_r, half_length=body_hh, material=main_mat
        )
        builder.add_cylinder_visual(
            pose=neck_pose, radius=neck_r, half_length=neck_hh, material=main_mat
        )
        builder.add_cylinder_visual(
            pose=cap_pose, radius=cap_r, half_length=cap_hh, material=cap_mat
        )
        if add_collision:
            builder.add_cylinder_collision(
                pose=body_pose, radius=body_r, half_length=body_hh, material=physx_mat
            )
            builder.add_cylinder_collision(
                pose=neck_pose, radius=neck_r, half_length=neck_hh, material=physx_mat
            )
            builder.add_cylinder_collision(
                pose=cap_pose, radius=cap_r, half_length=cap_hh, material=physx_mat
            )

    elif name == "green_box":
        half_size = [0.022, 0.018, 0.019]
        builder.add_box_visual(half_size=half_size, material=main_mat)
        if add_collision:
            builder.add_box_collision(half_size=half_size, material=physx_mat)

    elif name == "yellow_block":
        half_size = [0.022, 0.022, 0.022]
        builder.add_box_visual(half_size=half_size, material=main_mat)
        if add_collision:
            builder.add_box_collision(half_size=half_size, material=physx_mat)

    elif name == "orange_cylinder":
        cyl_pose = _upright_cylinder_pose(0.0)
        builder.add_cylinder_visual(
            pose=cyl_pose, radius=0.021, half_length=0.032, material=main_mat
        )
        if add_collision:
            builder.add_cylinder_collision(
                pose=cyl_pose, radius=0.021, half_length=0.032, material=physx_mat
            )

    elif name == "purple_bar":
        half_size = [0.048, 0.016, 0.016]
        builder.add_box_visual(half_size=half_size, material=main_mat)
        if add_collision:
            builder.add_box_collision(half_size=half_size, material=physx_mat)

    elif name == "gray_tray":
        # Shallow tray: 8.8cm x 5.6cm x 2.0cm total (graspable across 5.6cm width)
        rim_mat = _make_render_material((0.38, 0.40, 0.44, 1.0), roughness=0.30, metallic=0.45)
        parts = [
            (sapien.Pose(p=[0.0, 0.0, -0.005]), [0.044, 0.028, 0.005], main_mat),
            (sapien.Pose(p=[0.041, 0.0, 0.003]), [0.003, 0.028, 0.007], rim_mat),
            (sapien.Pose(p=[-0.041, 0.0, 0.003]), [0.003, 0.028, 0.007], rim_mat),
            (sapien.Pose(p=[0.0, 0.025, 0.003]), [0.038, 0.003, 0.007], rim_mat),
            (sapien.Pose(p=[0.0, -0.025, 0.003]), [0.038, 0.003, 0.007], rim_mat),
        ]
        for p_pose, p_half, p_mat in parts:
            builder.add_box_visual(pose=p_pose, half_size=p_half, material=p_mat)
            if add_collision:
                builder.add_box_collision(pose=p_pose, half_size=p_half, material=physx_mat)

    elif name == "cyan_t_tool":
        # T-shaped tool: stem + crossbar, both 2.6cm wide & 2.6cm tall
        stem_pose = sapien.Pose(p=[-0.008, 0.0, 0.0])
        stem_half = [0.034, 0.013, 0.013]
        bar_pose = sapien.Pose(p=[0.022, 0.0, 0.0])
        bar_half = [0.013, 0.032, 0.013]
        builder.add_box_visual(pose=stem_pose, half_size=stem_half, material=main_mat)
        builder.add_box_visual(pose=bar_pose, half_size=bar_half, material=main_mat)
        if add_collision:
            builder.add_box_collision(pose=stem_pose, half_size=stem_half, material=physx_mat)
            builder.add_box_collision(pose=bar_pose, half_size=bar_half, material=physx_mat)

    elif name == "magenta_cube":
        half_size = [0.019, 0.019, 0.019]
        stripe_mat = _make_render_material((0.97, 0.85, 0.94, 1.0), roughness=0.35)
        builder.add_box_visual(half_size=half_size, material=main_mat)
        builder.add_box_visual(
            pose=sapien.Pose(p=[0.0, 0.0, 0.0185]),
            half_size=[0.0192, 0.0055, 0.0015],
            material=stripe_mat,
        )
        if add_collision:
            builder.add_box_collision(half_size=half_size, material=physx_mat)

    else:
        raise ValueError(f"Unknown catalog object name: {name!r}")

    if add_collision:
        builder._mass = float(spec.mass_kg)
    if initial_pose is not None:
        builder.initial_pose = initial_pose

    if body_type == "dynamic":
        return builder.build(name=name)
    if body_type == "kinematic":
        return builder.build_kinematic(name=name)
    if body_type == "static":
        return builder.build_static(name=name)
    raise ValueError(f"Unsupported body_type: {body_type!r}")


def resolve_object_names(
    object_names: Sequence[str] | None = None,
    num_objects: int = 6,
    include_novel: bool = False,
) -> list[str]:
    """Resolve the ordered list of catalog object names to spawn."""
    if object_names is not None:
        resolved = list(object_names)
    else:
        n_base = int(np.clip(num_objects, 1, len(DEFAULT_BASE_OBJECTS)))
        resolved = list(DEFAULT_BASE_OBJECTS[:n_base])
        if include_novel:
            for novel_name in HELD_OUT_NOVEL_OBJECTS:
                if novel_name not in resolved:
                    resolved.append(novel_name)

    for obj_name in resolved:
        if obj_name not in OBJECT_CATALOG:
            raise ValueError(
                f"Unknown object {obj_name!r}. Valid options: {sorted(OBJECT_CATALOG.keys())}"
            )
    return resolved


@register_env("OmniGDAClutter-v1", max_episode_steps=500)
class OmniGDAClutterEnv(BaseEnv):
    """ManiSkill 3 cluttered tabletop environment with Franka Panda and dual RGB-D cameras."""

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    def __init__(
        self,
        *args: Any,
        robot_uids: str = "panda",
        object_names: Sequence[str] | None = None,
        num_objects: int = 6,
        include_novel: bool = False,
        image_width: int = 640,
        image_height: int = 480,
        robot_init_qpos_noise: float = 0.0,
        spawn_x_range: tuple[float, float] = (-0.20, 0.16),
        spawn_y_range: tuple[float, float] = (-0.27, 0.27),
        min_object_clearance: float = 0.036,
        **kwargs: Any,
    ) -> None:
        self.object_names = resolve_object_names(
            object_names=object_names,
            num_objects=num_objects,
            include_novel=include_novel,
        )
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self.spawn_x_range = (float(spawn_x_range[0]), float(spawn_x_range[1]))
        self.spawn_y_range = (float(spawn_y_range[0]), float(spawn_y_range[1]))
        self.min_object_clearance = float(min_object_clearance)
        self.objects: dict[str, Actor] = {}
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        # Primary angled overhead tabletop view (front-top looking back toward objects)
        base_pose = sapien_utils.look_at(
            eye=[0.48, 0.02, 0.50],
            target=[-0.02, 0.0, 0.02],
        )
        # Secondary side-angled view for Next-Best-View active perception
        nbv_pose = sapien_utils.look_at(
            eye=[0.10, -0.52, 0.46],
            target=[-0.02, 0.0, 0.02],
        )
        return [
            CameraConfig(
                uid="base_camera",
                pose=base_pose,
                width=self.image_width,
                height=self.image_height,
                fov=np.pi / 3.0,
                near=0.05,
                far=5.0,
            ),
            CameraConfig(
                uid="nbv_camera",
                pose=nbv_pose,
                width=self.image_width,
                height=self.image_height,
                fov=np.pi / 3.0,
                near=0.05,
                far=5.0,
            ),
        ]

    @property
    def _default_human_render_camera_configs(self) -> CameraConfig:
        overview_pose = sapien_utils.look_at(
            eye=[0.52, -0.40, 0.50],
            target=[-0.10, 0.0, 0.08],
        )
        return CameraConfig(
            uid="render_camera",
            pose=overview_pose,
            width=self.image_width,
            height=self.image_height,
            fov=np.pi / 3.0,
            near=0.05,
            far=5.0,
        )

    def _load_agent(self, options: dict[str, Any]) -> None:
        super()._load_agent(options, sapien.Pose(p=list(PANDA_BASE_POSITION)))

    def _load_scene(self, options: dict[str, Any]) -> None:
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.objects = {}
        for idx, name in enumerate(self.object_names):
            spec = OBJECT_CATALOG[name]
            init_pose = sapien.Pose(p=[0.0, -0.3 + idx * 0.1, spec.z_offset])
            self.objects[name] = build_catalog_object(
                self.scene,
                spec=spec,
                initial_pose=init_pose,
                body_type="dynamic",
            )

    def _sample_non_overlapping_poses(
        self, rng: np.random.RandomState | np.random.Generator
    ) -> dict[str, tuple[float, float, float]]:
        """Sample collision-free (x, y, yaw) tabletop placements deterministically."""
        placed: list[tuple[float, float, float]] = []  # (x, y, radius)
        poses: dict[str, tuple[float, float, float]] = {}

        x_lo, x_hi = self.spawn_x_range
        y_lo, y_hi = self.spawn_y_range

        # Precompute a dense candidate grid for deterministic max-clearance fallback
        grid_xs = np.linspace(x_lo, x_hi, 7)
        grid_ys = np.linspace(y_lo, y_hi, 9)
        grid_pts = [(float(gx), float(gy)) for gx in grid_xs for gy in grid_ys]

        for name in self.object_names:
            spec = OBJECT_CATALOG[name]
            r = spec.footprint_radius
            chosen: tuple[float, float] | None = None

            for _ in range(500):
                cx = float(rng.uniform(x_lo, x_hi))
                cy = float(rng.uniform(y_lo, y_hi))
                if all(
                    np.hypot(cx - px, cy - py) >= (r + pr + self.min_object_clearance)
                    for px, py, pr in placed
                ):
                    chosen = (cx, cy)
                    break

            if chosen is None:
                # Choose the grid point with the largest minimum clearance to placed objects
                best_pt = grid_pts[0]
                best_margin = -1e9
                for gx, gy in grid_pts:
                    margin = min(
                        (np.hypot(gx - px, gy - py) - (r + pr) for px, py, pr in placed),
                        default=1e9,
                    )
                    if margin > best_margin:
                        best_margin = margin
                        best_pt = (gx, gy)
                chosen = best_pt

            yaw = float(rng.uniform(-np.pi, np.pi))
            placed.append((chosen[0], chosen[1], r))
            poses[name] = (chosen[0], chosen[1], yaw)

        return poses

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Set Panda to a clean, camera-unobstructed ready pose
            qpos = np.tile(PANDA_READY_QPOS[None, :], (b, 1))
            if self.robot_init_qpos_noise > 0.0:
                noise = self._episode_rng.normal(0.0, self.robot_init_qpos_noise, (b, 7))
                qpos[:, :7] += noise
            qpos[:, -2:] = 0.04
            self.agent.reset(qpos)
            self.agent.robot.set_pose(sapien.Pose(p=list(PANDA_BASE_POSITION)))

            # Place objects deterministically without interpenetration
            custom_poses = (options or {}).get("object_poses")
            sampled = self._sample_non_overlapping_poses(self._episode_rng)

            for name, actor in self.objects.items():
                spec = OBJECT_CATALOG[name]
                if custom_poses is not None and name in custom_poses:
                    cx, cy, yaw = custom_poses[name]
                else:
                    cx, cy, yaw = sampled[name]
                xyz = torch.tensor([[cx, cy, spec.z_offset]], dtype=torch.float32).repeat(b, 1)
                quat_wxyz = np.asarray(euler2quat(0.0, 0.0, yaw), dtype=np.float32)
                qs = torch.from_numpy(quat_wxyz).unsqueeze(0).repeat(b, 1)
                actor.set_pose(Pose.create_from_pq(xyz, qs))
                actor.set_linear_velocity(torch.zeros((b, 3), dtype=torch.float32))
                actor.set_angular_velocity(torch.zeros((b, 3), dtype=torch.float32))

    def evaluate(self) -> dict[str, torch.Tensor]:
        return {
            "success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        }

    def compute_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict[str, Any]
    ) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict[str, Any]
    ) -> torch.Tensor:
        return self.compute_dense_reward(obs=obs, action=action, info=info)


def _to_numpy(arr: Any) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def extract_rgbd_from_obs(
    obs: dict[str, Any],
    camera_name: str,
    segmentation_id_map: dict[int, Any],
    timestamp_sec: float = 0.0,
) -> tuple[RGBDFrame, dict[int, str]]:
    """Convert ManiSkill 3 sensor observations into a strict OpenCV-convention RGBDFrame."""
    sensor_data = obs["sensor_data"][camera_name]
    sensor_param = obs["sensor_param"][camera_name]

    rgb = _to_numpy(sensor_data["rgb"])[0]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb[..., :3])

    raw_depth = _to_numpy(sensor_data["depth"])[0]
    if raw_depth.ndim == 3:
        raw_depth = raw_depth[..., 0]
    if np.issubdtype(raw_depth.dtype, np.integer):
        # ManiSkill 3 stores depth as int16 in millimeters
        depth_m = raw_depth.astype(np.float32) / 1000.0
    else:
        depth_m = raw_depth.astype(np.float32)
    depth_m = np.where(depth_m > 0.0, depth_m, 0.0).astype(np.float32)

    gt_seg: np.ndarray | None = None
    if "segmentation" in sensor_data:
        raw_seg = _to_numpy(sensor_data["segmentation"])[0]
        if raw_seg.ndim == 3:
            raw_seg = raw_seg[..., 0]
        gt_seg = raw_seg.astype(np.int32)

    intrinsic_cv = _to_numpy(sensor_param["intrinsic_cv"])[0].astype(np.float64)
    extrinsic_cv = _to_numpy(sensor_param["extrinsic_cv"])[0].astype(np.float64)
    # extrinsic_cv is (3, 4) world -> OpenCV camera optical frame (+X right, +Y down, +Z forward)
    world_to_cam_cv = np.eye(4, dtype=np.float64)
    world_to_cam_cv[:3, :4] = extrinsic_cv
    camera_to_world = np.linalg.inv(world_to_cam_cv)

    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    intrinsics = CameraIntrinsics(
        fx=float(intrinsic_cv[0, 0]),
        fy=float(intrinsic_cv[1, 1]),
        cx=float(intrinsic_cv[0, 2]),
        cy=float(intrinsic_cv[1, 2]),
        width=width,
        height=height,
    )

    seg_map: dict[int, str] = {}
    for seg_id, entity in segmentation_id_map.items():
        seg_map[int(seg_id)] = str(getattr(entity, "name", entity))

    frame = RGBDFrame(
        rgb=rgb,
        depth=depth_m,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
        timestamp_sec=float(timestamp_sec),
        frame_id=f"{camera_name}_optical_frame",
        gt_segmentation=gt_seg,
    )
    return frame, seg_map


class ClutteredTabletopSim:
    """High-level wrapper for OmniGDAClutter-v1 providing RGB-D capture, poses, and rendering."""

    def __init__(
        self,
        object_names: Sequence[str] | None = None,
        num_objects: int = 6,
        include_novel: bool = False,
        image_width: int = 640,
        image_height: int = 480,
        control_mode: str = "pd_joint_pos",
        render_mode: str = "rgb_array",
        seed: int = 42,
        robot_init_qpos_noise: float = 0.0,
        sim_backend: str = "cpu",
        settle_steps: int = 10,
    ) -> None:
        self.default_seed = int(seed)
        self.control_mode = str(control_mode)
        self.settle_steps = int(settle_steps)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self._step_count = 0

        self.env = gym.make(
            "OmniGDAClutter-v1",
            obs_mode="rgb+depth+segmentation",
            control_mode=control_mode,
            render_mode=render_mode,
            sim_backend=sim_backend,
            object_names=object_names,
            num_objects=num_objects,
            include_novel=include_novel,
            image_width=image_width,
            image_height=image_height,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )
        self._last_obs: dict[str, Any] | None = None
        self.reset(seed=self.default_seed)

    @property
    def base_env(self) -> OmniGDAClutterEnv:
        return self.env.unwrapped  # type: ignore[return-value]

    @property
    def control_timestep(self) -> float:
        return float(self.base_env.control_timestep)

    @property
    def robot_base_pose(self) -> sapien.Pose:
        pose = self.base_env.agent.robot.pose
        p = _to_numpy(pose.p)[0].astype(np.float64)
        q = _to_numpy(pose.q)[0].astype(np.float64)
        return sapien.Pose(p=p, q=q)

    def get_object_names(self) -> list[str]:
        return list(self.base_env.objects.keys())

    def get_camera_names(self) -> list[str]:
        return list(self.base_env._sensors.keys())

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        reset_seed = self.default_seed if seed is None else int(seed)
        obs, _ = self.env.reset(seed=reset_seed, options=options)
        self._step_count = 0
        self._last_obs = obs

        # Hold initial joint positions briefly so dynamic tabletop objects settle cleanly
        if self.settle_steps > 0 and self.control_mode == "pd_joint_pos":
            hold_qpos = self.get_robot_qpos()[:7]
            hold_action = np.concatenate([hold_qpos, [1.0]]).astype(np.float32)
            for _ in range(self.settle_steps):
                self._last_obs, _, _, _, _ = self.env.step(hold_action)
                self._step_count += 1

        return self.get_object_poses()

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(
            np.asarray(action, dtype=np.float32)
        )
        self._step_count += 1
        self._last_obs = obs
        return obs, float(reward), bool(terminated), bool(truncated), info

    def get_robot_qpos(self) -> np.ndarray:
        return _to_numpy(self.base_env.agent.robot.get_qpos())[0].astype(np.float64)

    def get_tcp_pose(self) -> np.ndarray:
        return _to_numpy(self.base_env.agent.tcp_pose.to_transformation_matrix())[0].astype(
            np.float64
        )

    def get_object_poses(self) -> dict[str, np.ndarray]:
        """Return world-frame 4x4 homogeneous poses for all tabletop objects."""
        poses: dict[str, np.ndarray] = {}
        for name, actor in self.base_env.objects.items():
            mat = _to_numpy(actor.pose.to_transformation_matrix())[0].astype(np.float64)
            poses[name] = mat
        return poses

    def get_object_positions(self) -> dict[str, np.ndarray]:
        """Return world-frame 3D positions (x, y, z) for all tabletop objects."""
        return {name: pose[:3, 3].copy() for name, pose in self.get_object_poses().items()}

    def capture_rgbd(self, camera_name: str = "base_camera") -> tuple[RGBDFrame, dict[int, str]]:
        """Capture an RGB-D frame and segmentation-id-to-name map from the named camera."""
        valid_cams = self.get_camera_names()
        if camera_name not in valid_cams:
            raise ValueError(f"Unknown camera {camera_name!r}. Available: {valid_cams}")

        obs = self.base_env.get_obs()
        self._last_obs = obs
        timestamp_sec = self._step_count * self.control_timestep
        return extract_rgbd_from_obs(
            obs=obs,
            camera_name=camera_name,
            segmentation_id_map=self.base_env.segmentation_id_map,
            timestamp_sec=timestamp_sec,
        )

    def render_overview(self) -> np.ndarray:
        """Render a third-person overview RGB image (H, W, 3) uint8 for demo recording."""
        frame = self.env.render()
        arr = _to_numpy(frame)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr[..., :3])

    def close(self) -> None:
        if self.env is not None:
            self.env.close()

    def __enter__(self) -> ClutteredTabletopSim:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
