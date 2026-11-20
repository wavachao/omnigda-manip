"""Isolated multi-view few-shot template generator for OmniGDA-Manip."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

from omnigda.core.gda_math import backproject_depth
from omnigda.core.schemas import SupportSample
from omnigda.sim.clutter_env import (
    OBJECT_CATALOG,
    build_catalog_object,
    extract_rgbd_from_obs,
    resolve_object_names,
)

# Off-stage parking pose for inactive objects during single-object template capture
_PARK_POSITION = np.array([5.0, 5.0, -5.0], dtype=np.float32)


@register_env("OmniGDATemplateStudio-v1", max_episode_steps=100)
class OmniGDATemplateStudioEnv(BaseEnv):
    """Robot-free tabletop studio matching OmniGDAClutter-v1 lighting for clean template capture."""

    SUPPORTED_ROBOTS = ["panda"]

    def __init__(
        self,
        *args: Any,
        robot_uids: str = "panda",
        object_names: Sequence[str] | None = None,
        image_width: int = 640,
        image_height: int = 480,
        **kwargs: Any,
    ) -> None:
        self.object_names = resolve_object_names(object_names=object_names)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.objects: dict[str, Actor] = {}
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        # Two complementary elevation views matching tabletop scale without occlusion
        cam_mid = sapien_utils.look_at(
            eye=[0.34, 0.0, 0.28],
            target=[0.0, 0.0, 0.025],
        )
        cam_high = sapien_utils.look_at(
            eye=[0.24, -0.22, 0.36],
            target=[0.0, 0.0, 0.025],
        )
        return [
            CameraConfig(
                uid="studio_mid",
                pose=cam_mid,
                width=self.image_width,
                height=self.image_height,
                fov=np.pi / 3.0,
                near=0.05,
                far=5.0,
            ),
            CameraConfig(
                uid="studio_high",
                pose=cam_high,
                width=self.image_width,
                height=self.image_height,
                fov=np.pi / 3.0,
                near=0.05,
                far=5.0,
            ),
        ]

    def _load_agent(self, options: dict[str, Any]) -> None:
        super()._load_agent(options, sapien.Pose(p=[-5.0, 0.0, -5.0]))

    def _load_scene(self, options: dict[str, Any]) -> None:
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0.0)
        self.table_scene.build()
        self.objects = {}
        for idx, name in enumerate(self.object_names):
            spec = OBJECT_CATALOG[name]
            park_pose = sapien.Pose(p=[5.0 + idx * 0.5, 5.0, -5.0])
            self.objects[name] = build_catalog_object(
                self.scene,
                spec=spec,
                initial_pose=park_pose,
                body_type="kinematic",
            )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        self.table_scene.initialize(env_idx)
        # Keep Panda far off-stage so studio captures contain only the isolated target object
        self.agent.robot.set_pose(sapien.Pose(p=[-5.0, 0.0, -5.0]))
        self.park_all_objects()

    def park_all_objects(self) -> None:
        """Move all objects off-stage below the floor."""
        for idx, actor in enumerate(self.objects.values()):
            pos = torch.tensor([[5.0 + idx * 0.5, 5.0, -5.0]], dtype=torch.float32)
            quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
            actor.set_pose(Pose.create_from_pq(pos, quat))

    def stage_single_object(self, name: str, yaw_rad: float = 0.0) -> None:
        """Place only `name` centered on the tabletop at the specified yaw angle."""
        self.park_all_objects()
        spec = OBJECT_CATALOG[name]
        pos = torch.tensor([[0.0, 0.0, spec.z_offset]], dtype=torch.float32)
        quat_wxyz = np.asarray(euler2quat(0.0, 0.0, float(yaw_rad)), dtype=np.float32)
        quat = torch.from_numpy(quat_wxyz).unsqueeze(0)
        self.objects[name].set_pose(Pose.create_from_pq(pos, quat))
        # Advance render transforms so updated kinematic pose reflects immediately in cameras
        self.scene.update_render()

    def evaluate(self) -> dict[str, torch.Tensor]:
        return {"success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}

    def compute_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict[str, Any]
    ) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict[str, Any]
    ) -> torch.Tensor:
        return self.compute_dense_reward(obs=obs, action=action, info=info)


def capture_few_shot_templates(
    object_names: Sequence[str],
    num_shots: int = 3,
    seed: int = 42,
    image_width: int = 640,
    image_height: int = 480,
) -> dict[str, list[SupportSample]]:
    """Render clean isolated multi-view RGB-D + exact binary mask SupportSamples for each object.

    Args:
        object_names: Catalog object names (including held-out novel objects) to capture.
        num_shots: Number of distinct viewpoints/orientations to capture per object.
        seed: Deterministic RNG seed controlling starting yaw offsets.
        image_width: Rendered template image width in pixels.
        image_height: Rendered template image height in pixels.

    Returns:
        Mapping ``{object_name: [SupportSample, ...]}`` with exact binary masks, depth in meters,
        OpenCV camera intrinsics/extrinsics, and backprojected 3D object point clouds.
    """
    resolved_names = resolve_object_names(object_names=object_names)
    if num_shots < 1:
        raise ValueError(f"num_shots must be >= 1, got {num_shots}")

    rng = np.random.default_rng(int(seed))
    studio_cams = ("studio_mid", "studio_high")
    templates: dict[str, list[SupportSample]] = {}

    env = gym.make(
        "OmniGDATemplateStudio-v1",
        obs_mode="rgb+depth+segmentation",
        sim_backend="cpu",
        object_names=resolved_names,
        image_width=int(image_width),
        image_height=int(image_height),
    )
    try:
        env.reset(seed=int(seed))
        base_env: OmniGDATemplateStudioEnv = env.unwrapped  # type: ignore[assignment]

        for name in resolved_names:
            spec = OBJECT_CATALOG[name]
            base_yaw = float(rng.uniform(0.0, 2.0 * np.pi))
            samples: list[SupportSample] = []

            for shot_idx in range(int(num_shots)):
                yaw = float(base_yaw + shot_idx * (2.0 * np.pi / float(num_shots)))
                cam_name = studio_cams[shot_idx % len(studio_cams)]
                base_env.stage_single_object(name=name, yaw_rad=yaw)

                obs = base_env.get_obs()
                frame, seg_map = extract_rgbd_from_obs(
                    obs=obs,
                    camera_name=cam_name,
                    segmentation_id_map=base_env.segmentation_id_map,
                    timestamp_sec=float(shot_idx),
                )

                obj_seg_ids = [sid for sid, ename in seg_map.items() if ename == name]
                if not obj_seg_ids or frame.gt_segmentation is None:
                    raise RuntimeError(f"Failed to locate segmentation ID for {name!r}")
                mask = np.isin(frame.gt_segmentation, obj_seg_ids)

                pts_xyz, _ = backproject_depth(
                    depth=frame.depth,
                    fx=frame.intrinsics.fx,
                    fy=frame.intrinsics.fy,
                    cx=frame.intrinsics.cx,
                    cy=frame.intrinsics.cy,
                    camera_to_world=frame.camera_to_world,
                    mask=mask,
                    depth_min=0.05,
                    depth_max=2.5,
                )

                samples.append(
                    SupportSample(
                        class_name=name,
                        rgb=frame.rgb,
                        depth=frame.depth,
                        mask=mask,
                        intrinsics=frame.intrinsics,
                        camera_to_world=frame.camera_to_world,
                        point_cloud_xyz=pts_xyz,
                        text_prompt=spec.description,
                        metadata={
                            "shot_idx": shot_idx,
                            "camera_name": cam_name,
                            "yaw_rad": yaw,
                            "category": spec.category,
                            "grasp_width_m": spec.grasp_width_m,
                        },
                    )
                )

            templates[name] = samples
    finally:
        env.close()

    return templates
