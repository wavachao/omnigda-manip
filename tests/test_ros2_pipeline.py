"""Unit and live rclpy integration tests for the OmniGDA-Manip ROS 2 pipeline."""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from omnigda_interfaces.msg import (
    GraspPoseArray,
    Object3DArray,
    PerceptionMetrics,
    SceneGraph,
)
from omnigda_interfaces.srv import ListObjects, QueryScene, RegisterObject
from omnigda_manipulation.coordinator_node import CoordinatorNode
from omnigda_manipulation.grasp_planner_node import GraspPlannerNode
from omnigda_perception.perception_3d_node import Perception3DNode
from omnigda_perception.registry_service_node import RegistryServiceNode
from omnigda_sim.conversions import (
    grasp_to_ros_msg,
    grasps_to_marker_array,
    image_msg_to_numpy,
    make_header,
    matrix_to_transform_stamped,
    numpy_to_image_msg,
    object3d_to_ros_msg,
    objects_to_marker_array,
    rgbd_frame_to_ros_msgs,
    ros_msg_to_grasp,
    ros_msg_to_object3d,
    ros_msg_to_scene_graph,
    ros_msgs_to_rgbd_frame,
    scene_graph_to_ros_msg,
    transform_to_matrix,
)
from omnigda_sim.sim_bridge_node import SimBridgeNode
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, String
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from omnigda.core.schemas import (
    CameraIntrinsics,
    GraspCandidate,
    Object3DDetection,
    RGBDFrame,
    SceneGraph3D,
    SceneRelation,
)


@pytest.fixture(scope="module", autouse=True)
def rclpy_runtime() -> Any:
    """Initialize rclpy once for the module and guarantee clean shutdown."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def _spin_until(
    executor: SingleThreadedExecutor,
    predicate: Any,
    timeout_sec: float = 8.0,
) -> bool:
    """Spin an executor in short slices until predicate() is True or timeout expires."""
    deadline = time.perf_counter() + timeout_sec
    while time.perf_counter() < deadline:
        executor.spin_once(timeout_sec=0.05)
        if predicate():
            return True
    return bool(predicate())


# ---------------------------------------------------------------------------
# 1. Pure ROS 2 message round-trip conversion tests
# ---------------------------------------------------------------------------


def test_rgbd_frame_ros_roundtrip() -> None:
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, size=(64, 80, 3), dtype=np.uint8)
    depth = rng.uniform(0.25, 1.5, size=(64, 80)).astype(np.float32)
    intr = CameraIntrinsics(fx=525.0, fy=520.0, cx=40.0, cy=32.0, width=80, height=64)
    cam_to_world = np.eye(4, dtype=np.float64)
    cam_to_world[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.35, 0.5]).as_matrix()
    cam_to_world[:3, 3] = np.array([0.45, -0.1, 0.52])

    frame = RGBDFrame(
        rgb=rgb,
        depth=depth,
        intrinsics=intr,
        camera_to_world=cam_to_world,
        timestamp_sec=12.3456,
        frame_id="camera_color_optical_frame",
    )

    rgb_msg, depth_msg, info_msg = rgbd_frame_to_ros_msgs(frame)
    assert rgb_msg.encoding == "rgb8"
    assert depth_msg.encoding == "32FC1"
    assert rgb_msg.header.frame_id == "camera_color_optical_frame"
    assert info_msg.width == 80 and info_msg.height == 64

    recovered = ros_msgs_to_rgbd_frame(rgb_msg, depth_msg, info_msg, camera_to_world=cam_to_world)
    assert np.array_equal(recovered.rgb, rgb)
    assert np.allclose(recovered.depth, depth, atol=1e-6)
    assert recovered.intrinsics.fx == pytest.approx(intr.fx)
    assert recovered.intrinsics.fy == pytest.approx(intr.fy)
    assert recovered.intrinsics.cx == pytest.approx(intr.cx)
    assert recovered.intrinsics.cy == pytest.approx(intr.cy)
    assert recovered.intrinsics.width == intr.width
    assert recovered.intrinsics.height == intr.height
    assert np.allclose(recovered.camera_to_world, cam_to_world, atol=1e-6)
    assert recovered.timestamp_sec == pytest.approx(12.3456, abs=1e-6)
    assert recovered.frame_id == "camera_color_optical_frame"

    # Also verify mono8 + TransformStamped roundtrip
    mask = (rng.random((64, 80)) > 0.5).astype(np.uint8) * 3
    mask_msg = numpy_to_image_msg(mask, "mono8")
    assert np.array_equal(image_msg_to_numpy(mask_msg), mask)

    tf_msg = matrix_to_transform_stamped(cam_to_world, "world", "camera_color_optical_frame")
    assert np.allclose(transform_to_matrix(tf_msg), cam_to_world, atol=1e-6)


def test_object3d_grasp_scene_graph_and_markers_roundtrip() -> None:
    rot_a = Rotation.from_euler("z", 0.4).as_matrix()
    rot_b = Rotation.from_euler("z", -0.25).as_matrix()

    obj_a = Object3DDetection(
        instance_id=1,
        class_name="red_mug",
        confidence=0.91,
        entropy=0.22,
        ood_score=8.4,
        is_ood=False,
        centroid=np.array([0.05, 0.14, 0.045]),
        obb_center=np.array([0.05, 0.14, 0.045]),
        obb_extent=np.array([0.075, 0.075, 0.09]),
        obb_rotation=rot_a,
        point_cloud_xyz=np.zeros((10, 3), dtype=np.float32),
        bbox_2d=(100, 120, 180, 210),
        modality_weights={"seg2d": 0.35, "visual_clip": 0.15, "geom_3d": 0.2, "size_3d": 0.1, "text_clip": 0.2},
    )
    obj_b = Object3DDetection(
        instance_id=2,
        class_name="novel_toy",
        confidence=0.41,
        entropy=1.15,
        ood_score=64.0,
        is_ood=True,
        centroid=np.array([0.04, -0.14, 0.04]),
        obb_center=np.array([0.04, -0.14, 0.04]),
        obb_extent=np.array([0.06, 0.06, 0.08]),
        obb_rotation=rot_b,
        point_cloud_xyz=np.zeros((10, 3), dtype=np.float32),
        bbox_2d=(300, 130, 370, 220),
        modality_weights={"2d": 0.4, "3d": 0.4, "text": 0.2},
    )

    msg_a = object3d_to_ros_msg(obj_a)
    rec_a = ros_msg_to_object3d(msg_a)
    assert rec_a.instance_id == obj_a.instance_id
    assert rec_a.class_name == obj_a.class_name
    assert rec_a.confidence == pytest.approx(obj_a.confidence, abs=1e-5)
    assert rec_a.entropy == pytest.approx(obj_a.entropy, abs=1e-5)
    assert rec_a.ood_score == pytest.approx(obj_a.ood_score, abs=1e-5)
    assert rec_a.is_ood is False
    assert np.allclose(rec_a.centroid, obj_a.centroid, atol=1e-6)
    assert np.allclose(rec_a.obb_center, obj_a.obb_center, atol=1e-6)
    assert np.allclose(rec_a.obb_extent, obj_a.obb_extent, atol=1e-6)
    assert np.allclose(rec_a.obb_rotation, obj_a.obb_rotation, atol=1e-5)
    assert rec_a.bbox_2d == obj_a.bbox_2d
    assert rec_a.modality_weights["2d"] == pytest.approx(0.50, abs=1e-5)
    assert rec_a.modality_weights["3d"] == pytest.approx(0.30, abs=1e-5)
    assert rec_a.modality_weights["text"] == pytest.approx(0.20, abs=1e-5)

    # GraspCandidate roundtrip
    grasp_rot = Rotation.from_euler("xyz", [np.pi, 0.0, 0.35]).as_matrix()
    pose_4x4 = np.eye(4, dtype=np.float64)
    pose_4x4[:3, :3] = grasp_rot
    pose_4x4[:3, 3] = np.array([0.05, 0.14, 0.07])
    quat = Rotation.from_matrix(grasp_rot).as_quat()
    grasp = GraspCandidate(
        pose_4x4=pose_4x4,
        position=pose_4x4[:3, 3].copy(),
        quaternion_xyzw=quat,
        width_m=0.055,
        score=0.88,
        force_closure_score=0.92,
        antipodal_score=0.85,
        collision_free=True,
        approach_vector=grasp_rot[:, 2].copy(),
        target_instance_id=1,
        target_class_name="red_mug",
    )
    grasp_msg = grasp_to_ros_msg(grasp)
    rec_grasp = ros_msg_to_grasp(grasp_msg)
    assert rec_grasp.target_instance_id == 1
    assert rec_grasp.target_class_name == "red_mug"
    assert rec_grasp.width_m == pytest.approx(0.055, abs=1e-5)
    assert rec_grasp.score == pytest.approx(0.88, abs=1e-5)
    assert rec_grasp.collision_free is True
    assert np.allclose(rec_grasp.pose_4x4, pose_4x4, atol=1e-5)
    assert np.allclose(rec_grasp.approach_vector, grasp.approach_vector, atol=1e-5)

    # SceneGraph3D roundtrip
    sg = SceneGraph3D(
        objects=[obj_a, obj_b],
        relations=[
            SceneRelation(subject_id=1, relation="left_of", object_id=2, confidence=0.93, distance_m=0.28)
        ],
        timestamp_sec=4.25,
    )
    sg_msg = scene_graph_to_ros_msg(sg, frame_id="world")
    rec_sg = ros_msg_to_scene_graph(sg_msg)
    assert len(rec_sg.objects) == 2
    assert len(rec_sg.relations) == 1
    assert rec_sg.relations[0].relation == "left_of"
    assert rec_sg.relations[0].distance_m == pytest.approx(0.28, abs=1e-5)
    assert rec_sg.timestamp_sec == pytest.approx(4.25, abs=1e-5)

    # RViz2 MarkerArrays
    obj_markers = objects_to_marker_array([obj_a, obj_b], frame_id="world")
    assert obj_markers.markers[0].action == Marker.DELETEALL
    assert len(obj_markers.markers) == 1 + 2 * 2
    assert any("[OOD]" in m.text for m in obj_markers.markers if m.type == Marker.TEXT_VIEW_FACING)

    grasp_markers = grasps_to_marker_array([grasp], frame_id="world")
    assert grasp_markers.markers[0].action == Marker.DELETEALL
    assert len(grasp_markers.markers) == 1 + 2
    assert grasp_markers.markers[1].type == Marker.ARROW
    assert grasp_markers.markers[2].type == Marker.LINE_LIST
    assert len(grasp_markers.markers[2].points) == 6


# ---------------------------------------------------------------------------
# 2. Live rclpy integration: SimBridgeNode + RGB-D + CameraInfo + TF2 + Switch
# ---------------------------------------------------------------------------


def test_live_sim_bridge_node_publishes_rgbd_tf_and_switches_camera() -> None:
    sim_node = SimBridgeNode(
        parameter_overrides=[
            Parameter("seed", Parameter.Type.INTEGER, 42),
            Parameter("num_objects", Parameter.Type.INTEGER, 3),
            Parameter("include_novel", Parameter.Type.BOOL, False),
            Parameter("image_width", Parameter.Type.INTEGER, 320),
            Parameter("image_height", Parameter.Type.INTEGER, 240),
            Parameter("publish_rate_hz", Parameter.Type.DOUBLE, 10.0),
            Parameter("init_controller", Parameter.Type.BOOL, False),
        ]
    )

    probe = Node("sim_probe_node")
    received: dict[str, Any] = {"rgb": None, "depth": None, "info": None}
    sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
    probe.create_subscription(
        Image, "/camera/color/image_raw", lambda m: received.__setitem__("rgb", m), sensor_qos
    )
    probe.create_subscription(
        Image, "/camera/depth/image_raw", lambda m: received.__setitem__("depth", m), sensor_qos
    )
    probe.create_subscription(
        CameraInfo, "/camera/camera_info", lambda m: received.__setitem__("info", m), sensor_qos
    )

    tf_buffer = Buffer()
    _tf_listener = TransformListener(tf_buffer, probe)
    switch_client = probe.create_client(SetBool, "/sim/switch_camera")

    executor = SingleThreadedExecutor()
    executor.add_node(sim_node)
    executor.add_node(probe)

    try:
        # 1. Verify RGB, Depth, CameraInfo stream from ManiSkill 3
        ok_msgs = _spin_until(
            executor,
            lambda: received["rgb"] is not None
            and received["depth"] is not None
            and received["info"] is not None,
            timeout_sec=8.0,
        )
        assert ok_msgs, "Timed out waiting for RGB-D + CameraInfo from SimBridgeNode"

        rgb = image_msg_to_numpy(received["rgb"])
        depth = image_msg_to_numpy(received["depth"])
        assert rgb.shape == (240, 320, 3)
        assert rgb.dtype == np.uint8
        assert int(rgb.max()) > 0
        assert depth.shape == (240, 320)
        assert depth.dtype == np.float32
        valid_depth = depth[depth > 0.05]
        assert valid_depth.size > 1000
        assert float(np.median(valid_depth)) < 3.0
        assert received["info"].width == 320
        assert received["info"].height == 240
        assert received["info"].k[0] > 0.0

        # 2. Verify TF2 frames world -> camera_color_optical_frame & world -> panda_link0
        def _has_both_tfs() -> bool:
            return tf_buffer.can_transform(
                "world", "camera_color_optical_frame", Time()
            ) and tf_buffer.can_transform("world", "panda_link0", Time())

        assert _spin_until(executor, _has_both_tfs, timeout_sec=5.0), "Missing TF2 transforms"
        tf_base = tf_buffer.lookup_transform(
            "world", "camera_color_optical_frame", Time(), timeout=Duration(seconds=0.1)
        )
        tf_robot = tf_buffer.lookup_transform(
            "world", "panda_link0", Time(), timeout=Duration(seconds=0.1)
        )
        base_pos = transform_to_matrix(tf_base)[:3, 3]
        robot_pos = transform_to_matrix(tf_robot)[:3, 3]
        assert np.all(np.isfinite(base_pos))
        assert np.all(np.isfinite(robot_pos))

        # 3. Call /sim/switch_camera -> True (nbv_camera) and verify viewpoint pose changes
        assert switch_client.wait_for_service(timeout_sec=3.0)
        fut = switch_client.call_async(SetBool.Request(data=True))
        assert _spin_until(executor, fut.done, timeout_sec=5.0)
        resp = fut.result()
        assert resp is not None and resp.success
        assert sim_node.active_camera == "nbv_camera"

        def _tf_moved() -> bool:
            tf_now = tf_buffer.lookup_transform(
                "world", "camera_color_optical_frame", Time(), timeout=Duration(seconds=0.1)
            )
            now_pos = transform_to_matrix(tf_now)[:3, 3]
            return float(np.linalg.norm(now_pos - base_pos)) > 0.15

        assert _spin_until(executor, _tf_moved, timeout_sec=5.0), "Camera TF did not move to NBV"

        # 4. Toggle back -> False (base_camera)
        fut_back = switch_client.call_async(SetBool.Request(data=False))
        assert _spin_until(executor, fut_back.done, timeout_sec=5.0)
        resp_back = fut_back.result()
        assert resp_back is not None and resp_back.success
        assert sim_node.active_camera == "base_camera"
    finally:
        executor.shutdown()
        probe.destroy_node()
        sim_node.destroy_node()


# ---------------------------------------------------------------------------
# 3. Live rclpy integration: Registry + Perception3D + GraspPlanner + Coordinator
# ---------------------------------------------------------------------------


def test_live_registry_perception3d_grasp_planner_and_coordinator(tmp_path: Path) -> None:
    memory_dir = tmp_path / "omnigda_mem"
    memory_dir.mkdir(parents=True, exist_ok=True)

    registry_node = RegistryServiceNode(
        parameter_overrides=[
            Parameter("memory_dir", Parameter.Type.STRING, str(memory_dir)),
            Parameter("enable_2d", Parameter.Type.BOOL, False),
            Parameter("enable_clip", Parameter.Type.BOOL, False),
        ]
    )
    perception_3d_node = Perception3DNode(
        parameter_overrides=[
            Parameter("memory_dir", Parameter.Type.STRING, str(memory_dir)),
            Parameter("enable_clip", Parameter.Type.BOOL, False),
            Parameter("min_area_px", Parameter.Type.INTEGER, 50),
        ]
    )
    grasp_node = GraspPlannerNode(
        parameter_overrides=[
            Parameter("top_k_per_object", Parameter.Type.INTEGER, 4),
            Parameter("table_z", Parameter.Type.DOUBLE, 0.0),
        ]
    )
    coordinator_node = CoordinatorNode()

    driver = Node("pipeline_driver_node")
    sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
    rgb_pub = driver.create_publisher(Image, "/camera/color/image_raw", sensor_qos)
    depth_pub = driver.create_publisher(Image, "/camera/depth/image_raw", sensor_qos)
    info_pub = driver.create_publisher(CameraInfo, "/camera/camera_info", sensor_qos)
    mask_pub = driver.create_publisher(Image, "/perception/seg_mask", sensor_qos)
    conf_pub = driver.create_publisher(Image, "/perception/confidence_map", sensor_qos)
    lat_pub = driver.create_publisher(Float32, "/perception/seg2d_latency_ms", 10)
    tf_broadcaster = TransformBroadcaster(driver)

    reg_client = driver.create_client(RegisterObject, "/gda/register_object")
    list_client = driver.create_client(ListObjects, "/gda/list_objects")
    query_client = driver.create_client(QueryScene, "/agent/query_scene")

    outputs: dict[str, Any] = {
        "mem_updates": [],
        "objects_3d": None,
        "scene_graph": None,
        "metrics": None,
        "markers_3d": None,
        "grasps": None,
        "grasp_markers": None,
    }
    driver.create_subscription(
        String, "/gda/memory_updated", lambda m: outputs["mem_updates"].append(m.data), 10
    )
    driver.create_subscription(
        Object3DArray, "/perception/objects_3d", lambda m: outputs.__setitem__("objects_3d", m), 10
    )
    driver.create_subscription(
        SceneGraph, "/perception/scene_graph", lambda m: outputs.__setitem__("scene_graph", m), 10
    )
    driver.create_subscription(
        PerceptionMetrics, "/perception/metrics", lambda m: outputs.__setitem__("metrics", m), 10
    )
    driver.create_subscription(
        MarkerArray, "/perception/markers_3d", lambda m: outputs.__setitem__("markers_3d", m), 10
    )
    driver.create_subscription(
        GraspPoseArray, "/grasp/candidates", lambda m: outputs.__setitem__("grasps", m), 10
    )
    driver.create_subscription(
        MarkerArray, "/grasp/markers", lambda m: outputs.__setitem__("grasp_markers", m), 10
    )

    executor = SingleThreadedExecutor()
    for n in (registry_node, perception_3d_node, grasp_node, coordinator_node, driver):
        executor.add_node(n)

    try:
        # 1. Verify /gda/list_objects starts empty
        assert list_client.wait_for_service(timeout_sec=3.0)
        fut_list0 = list_client.call_async(ListObjects.Request())
        assert _spin_until(executor, fut_list0.done, timeout_sec=5.0)
        assert list(fut_list0.result().class_names) == []

        # 2. Construct a clean synthetic tabletop RGB-D frame with 2 tabletop boxes
        h, w = 120, 160
        intr = CameraIntrinsics(fx=180.0, fy=180.0, cx=80.0, cy=60.0, width=w, height=h)
        # Camera directly above tabletop at z=0.60 looking straight down (-Z world = +Z optical)
        cam_to_world = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.60],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rgb = np.full((h, w, 3), 180, dtype=np.uint8)
        depth = np.full((h, w), 0.60, dtype=np.float32)

        # Object 1 ("red_cube"): top surface at z=0.06 -> depth=0.54, pixel patch left side
        mask1 = np.zeros((h, w), dtype=np.uint8)
        mask1[44:76, 30:58] = 1
        rgb[mask1 > 0] = [220, 30, 30]
        depth[mask1 > 0] = 0.54
        # Slight vertical relief so 3D OBB has realistic thickness
        depth[48:72, 34:54] = 0.52

        # Object 2 ("blue_box"): top surface at z=0.08 -> depth=0.52, pixel patch right side
        mask2 = np.zeros((h, w), dtype=np.uint8)
        mask2[42:78, 102:132] = 1
        rgb[mask2 > 0] = [30, 60, 220]
        depth[mask2 > 0] = 0.52
        depth[46:74, 106:128] = 0.49

        frame = RGBDFrame(
            rgb=rgb,
            depth=depth,
            intrinsics=intr,
            camera_to_world=cam_to_world,
            timestamp_sec=1.0,
            frame_id="camera_color_optical_frame",
        )
        rgb_msg, depth_msg, info_msg = rgbd_frame_to_ros_msgs(frame)

        # 3. Register both classes via /gda/register_object
        assert reg_client.wait_for_service(timeout_sec=3.0)
        for cname, m_u8 in (("red_cube", mask1), ("blue_box", mask2)):
            req = RegisterObject.Request()
            req.class_name = cname
            req.text_prompt = ""
            req.rgb_image = rgb_msg
            req.depth_image = depth_msg
            req.mask_image = numpy_to_image_msg(m_u8, "mono8", header=rgb_msg.header)
            req.camera_info = info_msg
            fut = reg_client.call_async(req)
            assert _spin_until(executor, fut.done, timeout_sec=5.0)
            resp = fut.result()
            assert resp is not None and resp.success, f"Register failed: {resp}"

        assert _spin_until(executor, lambda: len(outputs["mem_updates"]) >= 2, timeout_sec=5.0)
        payload = json.loads(outputs["mem_updates"][-1])
        assert payload["total_classes"] == 2
        assert (memory_dir / "gmda3d_memory.npz").exists()

        # Verify /gda/list_objects returns both registered classes
        fut_list1 = list_client.call_async(ListObjects.Request())
        assert _spin_until(executor, fut_list1.done, timeout_sec=5.0)
        list_resp = fut_list1.result()
        assert list(list_resp.class_names) == ["red_cube", "blue_box"]
        assert list(list_resp.sample_counts) == [1, 1]

        # 4. Publish TF + RGB-D + SegMask + ConfidenceMap and verify full downstream pipeline
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        combined_mask[mask1 > 0] = 1
        combined_mask[mask2 > 0] = 2
        conf_map = np.full((h, w), 0.92, dtype=np.float32)

        stamp = driver.get_clock().now().to_msg()
        header = make_header("camera_color_optical_frame", stamp=stamp)
        rgb_msg.header = header
        depth_msg.header = header
        info_msg.header = header
        tf_msg = matrix_to_transform_stamped(
            cam_to_world, "world", "camera_color_optical_frame", stamp=stamp
        )

        # Broadcast TF + RGB-D first so TF buffer and caches warm up
        for _ in range(4):
            tf_broadcaster.sendTransform(tf_msg)
            rgb_pub.publish(rgb_msg)
            depth_pub.publish(depth_msg)
            info_pub.publish(info_msg)
            conf_pub.publish(numpy_to_image_msg(conf_map, "32FC1", header=header))
            lat_pub.publish(Float32(data=14.5))
            executor.spin_once(timeout_sec=0.05)

        mask_pub.publish(numpy_to_image_msg(combined_mask, "mono8", header=header))

        ok_pipe = _spin_until(
            executor,
            lambda: outputs["objects_3d"] is not None
            and outputs["scene_graph"] is not None
            and outputs["metrics"] is not None
            and outputs["markers_3d"] is not None
            and outputs["grasps"] is not None
            and outputs["grasp_markers"] is not None,
            timeout_sec=10.0,
        )
        assert ok_pipe, f"Incomplete pipeline outputs: { {k: v is not None for k, v in outputs.items()} }"

        assert len(outputs["objects_3d"].objects) == 2
        assert len(outputs["scene_graph"].objects) == 2
        assert outputs["metrics"].num_registered_classes == 2
        assert outputs["metrics"].seg2d_latency_ms == pytest.approx(14.5, abs=1e-4)
        assert outputs["metrics"].gmda3d_latency_ms > 0.0
        assert len(outputs["markers_3d"].markers) >= 5  # DELETEALL + 2*(CUBE + TEXT)

        # GraspPlannerNode produced valid collision-free grasps + RViz markers
        assert len(outputs["grasps"].grasps) > 0
        assert all(g.collision_free for g in outputs["grasps"].grasps)
        assert all(g.score > 0.0 for g in outputs["grasps"].grasps)
        assert len(outputs["grasp_markers"].markers) >= 3

        # 5. Query CoordinatorNode via /agent/query_scene
        assert query_client.wait_for_service(timeout_sec=3.0)
        first_class = outputs["scene_graph"].objects[0].class_name
        fut_q = query_client.call_async(QueryScene.Request(instruction=f"pick the {first_class}"))
        assert _spin_until(executor, fut_q.done, timeout_sec=5.0)
        q_resp = fut_q.result()
        assert q_resp is not None and q_resp.found
        assert q_resp.target_class_name == first_class
        assert q_resp.target_instance_id > 0
    finally:
        executor.shutdown()
        for n in (driver, coordinator_node, grasp_node, perception_3d_node, registry_node):
            n.destroy_node()


# ---------------------------------------------------------------------------
# 4. Launch files & RViz2 config structural validation
# ---------------------------------------------------------------------------


def test_bringup_launch_files_and_rviz_config() -> None:
    bringup_share = get_package_share_directory("omnigda_bringup")
    sim_share = get_package_share_directory("omnigda_sim")

    expected_args = {"backend", "seed", "num_objects", "include_novel", "use_rviz"}
    for fname, min_entities in (("perception_only.launch.py", 10), ("omnigda_full.launch.py", 12)):
        fpath = os.path.join(bringup_share, "launch", fname)
        assert os.path.isfile(fpath), f"Missing launch file {fpath}"
        spec = importlib.util.spec_from_file_location("launch_test_mod", fpath)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ld = mod.generate_launch_description()
        assert len(ld.entities) >= min_entities
        declared = {getattr(e, "name", "") for e in ld.entities}
        assert expected_args.issubset(declared)

    rviz_path = os.path.join(sim_share, "rviz", "omnigda.rviz")
    assert os.path.isfile(rviz_path)
    rviz_cfg = yaml.safe_load(Path(rviz_path).read_text(encoding="utf-8"))
    displays = rviz_cfg["Visualization Manager"]["Displays"]
    topics = {
        d["Topic"]["Value"]
        for d in displays
        if isinstance(d.get("Topic"), dict) and "Value" in d["Topic"]
    }
    assert "/camera/color/image_raw" in topics
    assert "/perception/markers_3d" in topics
    assert "/grasp/markers" in topics


def test_live_perception_2d_node_and_instruction_parser(tmp_path: Path) -> None:
    from omnigda_manipulation.coordinator_node import parse_instruction_fallback
    from omnigda_perception.perception_2d_node import Perception2DNode

    # Verify fallback instruction parser handles spatial relations & superlatives
    p1 = parse_instruction_fallback("pick the red mug to the left of the blue box")
    assert p1.action == "pick"
    assert p1.target_query == "red mug"
    assert p1.spatial_relation == "left_of"
    assert p1.reference_query == "blue box"

    p2 = parse_instruction_fallback("grasp the leftmost bottle")
    assert p2.spatial_relation == "leftmost"
    assert p2.target_query == "bottle"

    # Spin Perception2DNode live on /camera/color/image_raw
    p2d_node = Perception2DNode(
        parameter_overrides=[
            Parameter("memory_dir", Parameter.Type.STRING, str(tmp_path / "empty_mem")),
            Parameter("max_rate_hz", Parameter.Type.DOUBLE, 0.0),
        ]
    )
    probe = Node("p2d_probe_node")
    sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
    rgb_pub = probe.create_publisher(Image, "/camera/color/image_raw", sensor_qos)

    got: dict[str, Any] = {"mask": None, "conf": None, "lat": None}
    probe.create_subscription(
        Image, "/perception/seg_mask", lambda m: got.__setitem__("mask", m), sensor_qos
    )
    probe.create_subscription(
        Image, "/perception/confidence_map", lambda m: got.__setitem__("conf", m), sensor_qos
    )
    probe.create_subscription(
        Float32, "/perception/seg2d_latency_ms", lambda m: got.__setitem__("lat", m), 10
    )

    executor = SingleThreadedExecutor()
    executor.add_node(p2d_node)
    executor.add_node(probe)

    try:
        img = np.full((60, 80, 3), 120, dtype=np.uint8)
        hdr = make_header("camera_color_optical_frame", timestamp_sec=2.0)
        msg = numpy_to_image_msg(img, "rgb8", header=hdr)
        for _ in range(3):
            rgb_pub.publish(msg)
            executor.spin_once(timeout_sec=0.05)

        ok = _spin_until(
            executor,
            lambda: got["mask"] is not None and got["conf"] is not None and got["lat"] is not None,
            timeout_sec=5.0,
        )
        assert ok, "Perception2DNode did not publish mask/confidence/latency"
        assert got["mask"].encoding == "mono8"
        assert got["conf"].encoding == "32FC1"
        assert image_msg_to_numpy(got["mask"]).shape == (60, 80)
        assert image_msg_to_numpy(got["conf"]).shape == (60, 80)
        assert got["lat"].data >= 0.0
    finally:
        executor.shutdown()
        probe.destroy_node()
        p2d_node.destroy_node()
