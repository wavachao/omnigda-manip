import numpy as np

from omnigda.core import (
    CameraIntrinsics,
    GraspCandidate,
    Object3DDetection,
    SceneGraph3D,
    SceneRelation,
    load_config,
)


def test_load_default_config():
    cfg = load_config()
    assert cfg["project"]["name"] == "omnigda-manip"
    assert cfg["perception_2d"]["shrinkage_method"] in {"oas", "ledoit_wolf", "ridge"}
    assert cfg["grasp"]["gripper_max_width_m"] > cfg["grasp"]["gripper_min_width_m"]


def test_camera_intrinsics_matrix():
    intr = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
    K = intr.as_matrix()
    assert K.shape == (3, 3)
    assert np.isclose(K[0, 0], 600.0)
    assert np.isclose(K[1, 2], 240.0)


def test_scene_graph_and_grasp_contracts():
    det = Object3DDetection(
        instance_id=1,
        class_name="mug",
        confidence=0.92,
        entropy=0.21,
        ood_score=8.5,
        is_ood=False,
        centroid=np.array([0.1, 0.2, 0.05]),
        obb_center=np.array([0.1, 0.2, 0.05]),
        obb_extent=np.array([0.08, 0.08, 0.10]),
        obb_rotation=np.eye(3),
        point_cloud_xyz=np.zeros((64, 3), dtype=np.float32),
    )
    sg = SceneGraph3D(
        objects=[det],
        relations=[SceneRelation(subject_id=1, relation="near", object_id=1, confidence=1.0, distance_m=0.0)],
    )
    assert sg.get_by_id(1) is det
    assert sg.get_by_id(99) is None

    grasp = GraspCandidate(
        pose_4x4=np.eye(4),
        position=np.array([0.1, 0.2, 0.05]),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        width_m=0.06,
        score=0.88,
        force_closure_score=0.9,
        antipodal_score=0.86,
        collision_free=True,
        approach_vector=np.array([0.0, 0.0, -1.0]),
        target_instance_id=1,
        target_class_name="mug",
    )
    assert grasp.collision_free
