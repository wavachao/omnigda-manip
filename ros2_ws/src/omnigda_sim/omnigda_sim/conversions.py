"""Fast, ABI-safe ROS 2 message conversions for OmniGDA-Manip.

Uses direct NumPy buffer conversions (no cv_bridge dependency) for zero ABI quirks
between NumPy 1.26 and system ROS 2 packages.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import numpy as np
from builtin_interfaces.msg import Duration, Time
from geometry_msgs.msg import Point, Pose, Quaternion, TransformStamped, Vector3
from omnigda_interfaces.msg import (
    GraspPose,
    Object3D,
    SceneGraph,
)
from omnigda_interfaces.msg import (
    SceneRelation as RosSceneRelation,
)
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from omnigda.core.schemas import (
    CameraIntrinsics,
    GraspCandidate,
    Object3DDetection,
    RGBDFrame,
    SceneGraph3D,
    SceneRelation,
)

_ENCODING_SPECS: dict[str, tuple[np.dtype, int]] = {
    "rgb8": (np.dtype(np.uint8), 3),
    "bgr8": (np.dtype(np.uint8), 3),
    "mono8": (np.dtype(np.uint8), 1),
    "8UC1": (np.dtype(np.uint8), 1),
    "8UC3": (np.dtype(np.uint8), 3),
    "16UC1": (np.dtype(np.uint16), 1),
    "32FC1": (np.dtype(np.float32), 1),
    "32SC1": (np.dtype(np.int32), 1),
}


def stamp_from_sec(timestamp_sec: float) -> Time:
    """Convert floating-point seconds into a ROS 2 builtin_interfaces/Time."""
    t = max(0.0, float(timestamp_sec))
    sec = int(math.floor(t))
    nanosec = int(round((t - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    msg = Time()
    msg.sec = sec
    msg.nanosec = nanosec
    return msg


def stamp_to_sec(stamp: Time) -> float:
    """Convert ROS 2 builtin_interfaces/Time to floating-point seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def make_header(frame_id: str, stamp: Time | None = None, timestamp_sec: float = 0.0) -> Header:
    """Create a populated std_msgs/Header."""
    header = Header()
    header.frame_id = str(frame_id)
    header.stamp = stamp if stamp is not None else stamp_from_sec(timestamp_sec)
    return header


def _orthonormalize_rotation(rot: np.ndarray) -> np.ndarray:
    """Project a near-orthogonal 3x3 matrix onto SO(3) via SVD."""
    u, _, vt = np.linalg.svd(np.asarray(rot, dtype=np.float64).reshape(3, 3))
    r = u @ vt
    if np.linalg.det(r) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def matrix_to_pose(mat_4x4: np.ndarray) -> Pose:
    """Convert a (4, 4) SE(3) homogeneous matrix into geometry_msgs/Pose."""
    mat = np.asarray(mat_4x4, dtype=np.float64).reshape(4, 4)
    rot = _orthonormalize_rotation(mat[:3, :3])
    quat = Rotation.from_matrix(rot).as_quat()  # (x, y, z, w)

    pose = Pose()
    pose.position = Point(x=float(mat[0, 3]), y=float(mat[1, 3]), z=float(mat[2, 3]))
    pose.orientation = Quaternion(
        x=float(quat[0]),
        y=float(quat[1]),
        z=float(quat[2]),
        w=float(quat[3]),
    )
    return pose


def pose_to_matrix(pose: Pose) -> np.ndarray:
    """Convert a geometry_msgs/Pose into a (4, 4) float64 SE(3) matrix."""
    quat = np.array(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        quat /= norm

    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = Rotation.from_quat(quat).as_matrix()
    mat[:3, 3] = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)
    return mat


def matrix_to_transform_stamped(
    mat_4x4: np.ndarray,
    parent_frame_id: str,
    child_frame_id: str,
    stamp: Time | None = None,
    timestamp_sec: float = 0.0,
) -> TransformStamped:
    """Convert a (4, 4) SE(3) matrix into a geometry_msgs/TransformStamped."""
    pose = matrix_to_pose(mat_4x4)
    tf_msg = TransformStamped()
    tf_msg.header = make_header(parent_frame_id, stamp=stamp, timestamp_sec=timestamp_sec)
    tf_msg.child_frame_id = str(child_frame_id)
    tf_msg.transform.translation = Vector3(
        x=pose.position.x,
        y=pose.position.y,
        z=pose.position.z,
    )
    tf_msg.transform.rotation = pose.orientation
    return tf_msg


def transform_to_matrix(tf_msg: TransformStamped) -> np.ndarray:
    """Convert a geometry_msgs/TransformStamped into a (4, 4) float64 SE(3) matrix."""
    pose = Pose()
    pose.position = Point(
        x=float(tf_msg.transform.translation.x),
        y=float(tf_msg.transform.translation.y),
        z=float(tf_msg.transform.translation.z),
    )
    pose.orientation = tf_msg.transform.rotation
    return pose_to_matrix(pose)


def numpy_to_image_msg(
    array: np.ndarray,
    encoding: str,
    header: Header | None = None,
) -> Image:
    """Serialize a NumPy image array directly into a sensor_msgs/Image without cv_bridge."""
    if encoding not in _ENCODING_SPECS:
        raise ValueError(f"Unsupported image encoding {encoding!r}. Supported: {sorted(_ENCODING_SPECS)}")
    dtype, channels = _ENCODING_SPECS[encoding]

    arr = np.asarray(array, dtype=dtype)
    if channels == 1:
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        if arr.ndim != 2:
            raise ValueError(f"Encoding {encoding!r} expects 2D array (H, W), got shape {arr.shape}")
        height, width = int(arr.shape[0]), int(arr.shape[1])
    else:
        if arr.ndim != 3 or arr.shape[2] != channels:
            raise ValueError(
                f"Encoding {encoding!r} expects shape (H, W, {channels}), got {arr.shape}"
            )
        height, width = int(arr.shape[0]), int(arr.shape[1])

    contiguous = np.ascontiguousarray(arr)
    msg = Image()
    if header is not None:
        msg.header = header
    msg.height = height
    msg.width = width
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(width * channels * dtype.itemsize)
    msg.data = contiguous.tobytes()
    return msg


def image_msg_to_numpy(msg: Image, copy: bool = True) -> np.ndarray:
    """Deserialize a sensor_msgs/Image directly into a NumPy array without cv_bridge."""
    encoding = msg.encoding
    if encoding not in _ENCODING_SPECS:
        raise ValueError(f"Unsupported image encoding {encoding!r}. Supported: {sorted(_ENCODING_SPECS)}")
    dtype, channels = _ENCODING_SPECS[encoding]

    if msg.is_bigendian:
        dtype = dtype.newbyteorder(">")

    h = int(msg.height)
    w = int(msg.width)
    if h == 0 or w == 0:
        shape = (h, w) if channels == 1 else (h, w, channels)
        return np.empty(shape, dtype=dtype.newbyteorder("="))

    row_bytes = w * channels * dtype.itemsize
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.step == row_bytes or msg.step == 0:
        arr = np.frombuffer(msg.data, dtype=dtype, count=h * w * channels)
        shape = (h, w) if channels == 1 else (h, w, channels)
        arr = arr.reshape(shape)
    else:
        # Handle padded row strides cleanly
        rows = buf[: h * msg.step].reshape(h, msg.step)[:, :row_bytes]
        arr = np.frombuffer(np.ascontiguousarray(rows), dtype=dtype)
        shape = (h, w) if channels == 1 else (h, w, channels)
        arr = arr.reshape(shape)

    if msg.is_bigendian:
        arr = arr.astype(dtype.newbyteorder("="), copy=True)
    elif copy:
        arr = arr.copy()
    return arr


def intrinsics_to_camera_info(
    intrinsics: CameraIntrinsics,
    header: Header | None = None,
) -> CameraInfo:
    """Convert CameraIntrinsics into a pinhole sensor_msgs/CameraInfo."""
    msg = CameraInfo()
    if header is not None:
        msg.header = header
    msg.width = int(intrinsics.width)
    msg.height = int(intrinsics.height)
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.k = [
        float(intrinsics.fx), 0.0, float(intrinsics.cx),
        0.0, float(intrinsics.fy), float(intrinsics.cy),
        0.0, 0.0, 1.0,
    ]
    msg.r = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]
    msg.p = [
        float(intrinsics.fx), 0.0, float(intrinsics.cx), 0.0,
        0.0, float(intrinsics.fy), float(intrinsics.cy), 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]
    return msg


def camera_info_to_intrinsics(msg: CameraInfo) -> CameraIntrinsics:
    """Convert a sensor_msgs/CameraInfo back into CameraIntrinsics."""
    k = list(msg.k)
    fx = float(k[0]) if len(k) == 9 and k[0] > 0.0 else float(msg.p[0])
    fy = float(k[4]) if len(k) == 9 and k[4] > 0.0 else float(msg.p[5])
    cx = float(k[2]) if len(k) == 9 else float(msg.p[2])
    cy = float(k[5]) if len(k) == 9 else float(msg.p[6])
    return CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=int(msg.width),
        height=int(msg.height),
    )


def rgbd_frame_to_ros_msgs(
    frame: RGBDFrame,
    frame_id: str | None = None,
    stamp: Time | None = None,
) -> tuple[Image, Image, CameraInfo]:
    """Convert an RGBDFrame into (rgb_image, depth_image, camera_info) ROS 2 messages."""
    resolved_frame_id = frame.frame_id if frame_id is None else str(frame_id)
    header = make_header(resolved_frame_id, stamp=stamp, timestamp_sec=frame.timestamp_sec)

    rgb_msg = numpy_to_image_msg(frame.rgb, encoding="rgb8", header=header)
    depth_msg = numpy_to_image_msg(frame.depth.astype(np.float32, copy=False), encoding="32FC1", header=header)
    info_msg = intrinsics_to_camera_info(frame.intrinsics, header=header)
    return rgb_msg, depth_msg, info_msg


def ros_msgs_to_rgbd_frame(
    rgb_msg: Image,
    depth_msg: Image,
    camera_info_msg: CameraInfo,
    camera_to_world: np.ndarray | None = None,
) -> RGBDFrame:
    """Reconstruct an RGBDFrame from (rgb, depth, camera_info) ROS 2 messages."""
    rgb = image_msg_to_numpy(rgb_msg, copy=True)
    if rgb_msg.encoding == "bgr8":
        rgb = np.ascontiguousarray(rgb[..., ::-1])

    depth = image_msg_to_numpy(depth_msg, copy=True)
    if depth_msg.encoding == "16UC1":
        depth = depth.astype(np.float32) / 1000.0
    else:
        depth = depth.astype(np.float32, copy=False)

    intrinsics = camera_info_to_intrinsics(camera_info_msg)
    cam_to_world = (
        np.eye(4, dtype=np.float64)
        if camera_to_world is None
        else np.asarray(camera_to_world, dtype=np.float64).reshape(4, 4).copy()
    )
    frame_id = rgb_msg.header.frame_id or camera_info_msg.header.frame_id or "camera_color_optical_frame"
    timestamp_sec = stamp_to_sec(rgb_msg.header.stamp)

    return RGBDFrame(
        rgb=rgb,
        depth=depth,
        intrinsics=intrinsics,
        camera_to_world=cam_to_world,
        timestamp_sec=timestamp_sec,
        frame_id=frame_id,
    )


def _extract_summary_weights(modality_weights: dict[str, float]) -> tuple[float, float, float]:
    """Aggregate fine-grained modality weights into (weight_2d, weight_3d, weight_text)."""
    if not modality_weights:
        return 0.0, 0.0, 0.0

    w_2d = (
        float(modality_weights["2d"])
        if "2d" in modality_weights
        else float(modality_weights.get("seg2d", 0.0)) + float(modality_weights.get("visual_clip", 0.0))
    )
    w_3d = (
        float(modality_weights["3d"])
        if "3d" in modality_weights
        else float(modality_weights.get("geom_3d", 0.0)) + float(modality_weights.get("size_3d", 0.0))
    )
    w_text = (
        float(modality_weights["text"])
        if "text" in modality_weights
        else float(modality_weights.get("text_clip", 0.0))
    )
    return w_2d, w_3d, w_text


def object3d_to_ros_msg(obj: Object3DDetection) -> Object3D:
    """Convert an Object3DDetection into omnigda_interfaces/Object3D."""
    msg = Object3D()
    msg.instance_id = int(obj.instance_id)
    msg.class_name = str(obj.class_name)
    msg.confidence = float(obj.confidence)
    msg.entropy = float(obj.entropy)
    msg.ood_score = float(obj.ood_score)
    msg.is_ood = bool(obj.is_ood)

    centroid = np.asarray(obj.centroid, dtype=np.float64).reshape(3)
    msg.centroid = Point(x=float(centroid[0]), y=float(centroid[1]), z=float(centroid[2]))

    obb_mat = np.eye(4, dtype=np.float64)
    obb_mat[:3, :3] = _orthonormalize_rotation(obj.obb_rotation)
    obb_mat[:3, 3] = np.asarray(obj.obb_center, dtype=np.float64).reshape(3)
    msg.obb_pose = matrix_to_pose(obb_mat)

    extent = np.asarray(obj.obb_extent, dtype=np.float64).reshape(3)
    msg.obb_extent = Vector3(x=float(extent[0]), y=float(extent[1]), z=float(extent[2]))

    bbox = [int(v) for v in obj.bbox_2d]
    if len(bbox) != 4:
        bbox = [0, 0, 0, 0]
    msg.bbox_2d = bbox

    w_2d, w_3d, w_text = _extract_summary_weights(obj.modality_weights)
    msg.weight_2d = float(w_2d)
    msg.weight_3d = float(w_3d)
    msg.weight_text = float(w_text)
    return msg


def ros_msg_to_object3d(
    msg: Object3D,
    point_cloud_xyz: np.ndarray | None = None,
) -> Object3DDetection:
    """Convert omnigda_interfaces/Object3D back into Object3DDetection."""
    obb_mat = pose_to_matrix(msg.obb_pose)
    centroid = np.array([msg.centroid.x, msg.centroid.y, msg.centroid.z], dtype=np.float64)
    obb_center = obb_mat[:3, 3].copy()
    obb_rotation = obb_mat[:3, :3].copy()
    obb_extent = np.array([msg.obb_extent.x, msg.obb_extent.y, msg.obb_extent.z], dtype=np.float64)
    bbox_tuple = (int(msg.bbox_2d[0]), int(msg.bbox_2d[1]), int(msg.bbox_2d[2]), int(msg.bbox_2d[3]))
    pts = (
        np.empty((0, 3), dtype=np.float32)
        if point_cloud_xyz is None
        else np.asarray(point_cloud_xyz, dtype=np.float32).reshape(-1, 3)
    )

    return Object3DDetection(
        instance_id=int(msg.instance_id),
        class_name=str(msg.class_name),
        confidence=float(msg.confidence),
        entropy=float(msg.entropy),
        ood_score=float(msg.ood_score),
        is_ood=bool(msg.is_ood),
        centroid=centroid,
        obb_center=obb_center,
        obb_extent=obb_extent,
        obb_rotation=obb_rotation,
        point_cloud_xyz=pts,
        bbox_2d=bbox_tuple,
        modality_weights={
            "2d": float(msg.weight_2d),
            "3d": float(msg.weight_3d),
            "text": float(msg.weight_text),
        },
    )


def grasp_to_ros_msg(grasp: GraspCandidate) -> GraspPose:
    """Convert a GraspCandidate into omnigda_interfaces/GraspPose."""
    msg = GraspPose()
    msg.target_instance_id = int(grasp.target_instance_id)
    msg.target_class_name = str(grasp.target_class_name)

    quat = np.asarray(grasp.quaternion_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm > 1e-12:
        quat = quat / norm
    else:
        quat = Rotation.from_matrix(_orthonormalize_rotation(grasp.pose_4x4[:3, :3])).as_quat()

    pos = np.asarray(grasp.position, dtype=np.float64).reshape(3)
    msg.pose = Pose(
        position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
        orientation=Quaternion(
            x=float(quat[0]),
            y=float(quat[1]),
            z=float(quat[2]),
            w=float(quat[3]),
        ),
    )
    msg.width_m = float(grasp.width_m)
    msg.score = float(grasp.score)
    msg.force_closure_score = float(grasp.force_closure_score)
    msg.antipodal_score = float(grasp.antipodal_score)
    msg.collision_free = bool(grasp.collision_free)
    return msg


def ros_msg_to_grasp(msg: GraspPose) -> GraspCandidate:
    """Convert omnigda_interfaces/GraspPose back into GraspCandidate."""
    pose_4x4 = pose_to_matrix(msg.pose)
    position = pose_4x4[:3, 3].copy()
    quat = np.array(
        [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(quat))
    quat = quat / norm if norm > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    # Gripper TCP convention: local +Z is approach vector (wrist -> fingers)
    approach_vector = pose_4x4[:3, 2].copy()

    return GraspCandidate(
        pose_4x4=pose_4x4,
        position=position,
        quaternion_xyzw=quat,
        width_m=float(msg.width_m),
        score=float(msg.score),
        force_closure_score=float(msg.force_closure_score),
        antipodal_score=float(msg.antipodal_score),
        collision_free=bool(msg.collision_free),
        approach_vector=approach_vector,
        target_instance_id=int(msg.target_instance_id),
        target_class_name=str(msg.target_class_name),
    )


def scene_relation_to_ros_msg(rel: SceneRelation) -> RosSceneRelation:
    """Convert SceneRelation dataclass into omnigda_interfaces/SceneRelation."""
    msg = RosSceneRelation()
    msg.subject_id = int(rel.subject_id)
    msg.relation = str(rel.relation)
    msg.object_id = int(rel.object_id)
    msg.confidence = float(rel.confidence)
    msg.distance_m = float(rel.distance_m)
    return msg


def ros_msg_to_scene_relation(msg: RosSceneRelation) -> SceneRelation:
    """Convert omnigda_interfaces/SceneRelation into SceneRelation dataclass."""
    return SceneRelation(
        subject_id=int(msg.subject_id),
        relation=str(msg.relation),
        object_id=int(msg.object_id),
        confidence=float(msg.confidence),
        distance_m=float(msg.distance_m),
    )


def scene_graph_to_ros_msg(
    scene_graph: SceneGraph3D,
    frame_id: str = "world",
    stamp: Time | None = None,
) -> SceneGraph:
    """Convert SceneGraph3D into omnigda_interfaces/SceneGraph."""
    msg = SceneGraph()
    msg.header = make_header(frame_id, stamp=stamp, timestamp_sec=scene_graph.timestamp_sec)
    msg.objects = [object3d_to_ros_msg(obj) for obj in scene_graph.objects]
    msg.relations = [scene_relation_to_ros_msg(rel) for rel in scene_graph.relations]
    return msg


def ros_msg_to_scene_graph(msg: SceneGraph) -> SceneGraph3D:
    """Convert omnigda_interfaces/SceneGraph back into SceneGraph3D."""
    return SceneGraph3D(
        objects=[ros_msg_to_object3d(obj_msg) for obj_msg in msg.objects],
        relations=[ros_msg_to_scene_relation(rel_msg) for rel_msg in msg.relations],
        timestamp_sec=stamp_to_sec(msg.header.stamp),
    )


def _class_color(class_name: str, is_ood: bool = False, alpha: float = 0.45) -> ColorRGBA:
    """Generate a deterministic, visually distinct RGBA color per class."""
    if is_ood:
        return ColorRGBA(r=0.95, g=0.18, b=0.18, a=float(alpha))
    digest = hashlib.md5(class_name.encode("utf-8")).digest()
    r = 0.25 + 0.70 * (digest[0] / 255.0)
    g = 0.25 + 0.70 * (digest[1] / 255.0)
    b = 0.25 + 0.70 * (digest[2] / 255.0)
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(alpha))


def objects_to_marker_array(
    objects: Sequence[Object3DDetection],
    frame_id: str = "world",
    stamp: Time | None = None,
    ns: str = "omnigda_objects",
) -> MarkerArray:
    """Build RViz2 MarkerArray (3D OBB cubes + class labels) for detected objects."""
    header = make_header(frame_id, stamp=stamp)
    markers: list[Marker] = []

    clear = Marker()
    clear.header = header
    clear.ns = ns
    clear.id = 0
    clear.action = Marker.DELETEALL
    markers.append(clear)

    lifetime = Duration(sec=1, nanosec=0)

    for idx, obj in enumerate(objects):
        obb_mat = np.eye(4, dtype=np.float64)
        obb_mat[:3, :3] = _orthonormalize_rotation(obj.obb_rotation)
        obb_mat[:3, 3] = np.asarray(obj.obb_center, dtype=np.float64).reshape(3)
        extent = np.maximum(np.asarray(obj.obb_extent, dtype=np.float64).reshape(3), 0.005)

        cube = Marker()
        cube.header = header
        cube.ns = ns
        cube.id = idx * 2 + 1
        cube.type = Marker.CUBE
        cube.action = Marker.ADD
        cube.pose = matrix_to_pose(obb_mat)
        cube.scale = Vector3(x=float(extent[0]), y=float(extent[1]), z=float(extent[2]))
        cube.color = _class_color(obj.class_name, is_ood=obj.is_ood, alpha=0.45)
        cube.lifetime = lifetime
        markers.append(cube)

        label = Marker()
        label.header = header
        label.ns = ns
        label.id = idx * 2 + 2
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position = Point(
            x=float(obb_mat[0, 3]),
            y=float(obb_mat[1, 3]),
            z=float(obb_mat[2, 3] + 0.5 * extent[2] + 0.04),
        )
        label.pose.orientation.w = 1.0
        label.scale.z = 0.032
        label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
        tag = "[OOD] " if obj.is_ood else ""
        label.text = f"{tag}{obj.class_name} #{obj.instance_id} ({obj.confidence:.2f})"
        label.lifetime = lifetime
        markers.append(label)

    return MarkerArray(markers=markers)


def grasps_to_marker_array(
    grasps: Sequence[GraspCandidate],
    frame_id: str = "world",
    stamp: Time | None = None,
    ns: str = "omnigda_grasps",
    arrow_length_m: float = 0.08,
    finger_depth_m: float = 0.045,
) -> MarkerArray:
    """Build RViz2 MarkerArray (approach arrows + parallel-jaw wireframes) for 6DoF grasps."""
    header = make_header(frame_id, stamp=stamp)
    markers: list[Marker] = []

    clear = Marker()
    clear.header = header
    clear.ns = ns
    clear.id = 0
    clear.action = Marker.DELETEALL
    markers.append(clear)

    lifetime = Duration(sec=1, nanosec=0)

    for idx, grasp in enumerate(grasps):
        pose_4x4 = np.asarray(grasp.pose_4x4, dtype=np.float64).reshape(4, 4)
        rot = _orthonormalize_rotation(pose_4x4[:3, :3])
        tcp = pose_4x4[:3, 3]
        approach = rot[:, 2]  # wrist -> fingers
        closing = rot[:, 1]  # jaw axis

        score = float(np.clip(grasp.score, 0.0, 1.0))
        if grasp.collision_free:
            color = ColorRGBA(r=float(1.0 - score), g=float(0.35 + 0.65 * score), b=0.15, a=0.92)
        else:
            color = ColorRGBA(r=0.95, g=0.15, b=0.15, a=0.55)

        # 1. Approach arrow from pre-grasp tail to TCP
        tail = tcp - approach * float(arrow_length_m)
        arrow = Marker()
        arrow.header = header
        arrow.ns = ns
        arrow.id = idx * 2 + 1
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.points = [
            Point(x=float(tail[0]), y=float(tail[1]), z=float(tail[2])),
            Point(x=float(tcp[0]), y=float(tcp[1]), z=float(tcp[2])),
        ]
        arrow.scale = Vector3(x=0.006, y=0.012, z=0.018)
        arrow.color = color
        arrow.lifetime = lifetime
        markers.append(arrow)

        # 2. Parallel-jaw U-frame wireframe at TCP
        half_w = 0.5 * max(float(grasp.width_m), 0.01)
        base_center = tcp - approach * float(finger_depth_m)
        left_base = base_center + closing * half_w
        right_base = base_center - closing * half_w
        left_tip = tcp + closing * half_w
        right_tip = tcp - closing * half_w

        jaw = Marker()
        jaw.header = header
        jaw.ns = ns
        jaw.id = idx * 2 + 2
        jaw.type = Marker.LINE_LIST
        jaw.action = Marker.ADD
        jaw.pose.orientation.w = 1.0
        jaw.scale.x = 0.004
        jaw.color = color
        jaw.lifetime = lifetime

        def _pt(v: np.ndarray) -> Point:
            return Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))

        jaw.points = [
            _pt(left_base), _pt(right_base),
            _pt(left_base), _pt(left_tip),
            _pt(right_base), _pt(right_tip),
        ]
        markers.append(jaw)

    return MarkerArray(markers=markers)
