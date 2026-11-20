"""Interactive Flask Web Visualization & Manipulation Studio for OmniGDA-Manip."""

from __future__ import annotations

import argparse
import base64
import threading
import time
from typing import Any

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, request

from omnigda.agent import OmniGDAAgent
from omnigda.core.config import load_config
from omnigda.core.schemas import GraspCandidate, Object3DDetection, RGBDFrame, SegmentationResult
from omnigda.sim import (
    DEFAULT_BASE_OBJECTS,
    HELD_OUT_NOVEL_OBJECTS,
    ClutteredTabletopSim,
    FrankaGraspController,
    capture_few_shot_templates,
)
from omnigda.web.ui import INDEX_HTML

_PALETTE: list[tuple[int, int, int]] = [
    (230, 68, 68),
    (59, 130, 246),
    (34, 197, 94),
    (234, 179, 8),
    (249, 115, 22),
    (168, 85, 247),
    (148, 163, 184),
    (6, 182, 212),
    (236, 72, 153),
]


def _encode_rgb_jpeg_b64(rgb: np.ndarray, quality: int = 85) -> str:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode JPEG frame")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _project_world_point(frame: RGBDFrame, pt_world: np.ndarray) -> tuple[int, int] | None:
    w2c = np.linalg.inv(frame.camera_to_world)
    pt_cam = w2c[:3, :3] @ pt_world + w2c[:3, 3]
    if pt_cam[2] <= 1e-3:
        return None
    intr = frame.intrinsics
    u = int(round(intr.fx * pt_cam[0] / pt_cam[2] + intr.cx))
    v = int(round(intr.fy * pt_cam[1] / pt_cam[2] + intr.cy))
    if 0 <= u < intr.width and 0 <= v < intr.height:
        return (u, v)
    return None


def render_annotated_view(
    frame: RGBDFrame,
    seg: SegmentationResult | None = None,
    detections: list[Object3DDetection] | None = None,
    grasp: GraspCandidate | None = None,
) -> np.ndarray:
    """Overlay 2D SegmentGDA masks, 3D detection badges, and 6DoF grasp axes on RGB."""
    vis = frame.rgb.copy()
    if seg is not None and seg.class_map is not None:
        overlay = vis.copy()
        for idx, cname in enumerate(seg.class_names, start=1):
            mask = seg.masks.get(cname)
            if mask is None or not np.any(mask):
                continue
            overlay[mask] = _PALETTE[(idx - 1) % len(_PALETTE)]
        vis = cv2.addWeighted(vis, 0.62, overlay, 0.38, 0.0)

    if detections:
        for det in detections:
            x1, y1, x2, y2 = det.bbox_2d
            color = (248, 81, 73) if det.is_ood else (63, 185, 80)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"#{det.instance_id} {det.class_name} {det.confidence * 100:.0f}%"
            if det.is_ood:
                label = f"[OOD] {label}"
            y_txt = max(16, y1 - 6)
            cv2.putText(vis, label, (x1, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, label, (x1, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

    if grasp is not None:
        origin = grasp.position
        rot = grasp.pose_4x4[:3, :3]
        p0 = _project_world_point(frame, origin)
        if p0 is not None:
            for axis_idx, col in ((0, (255, 60, 60)), (1, (60, 255, 60)), (2, (60, 160, 255))):
                p1 = _project_world_point(frame, origin + rot[:, axis_idx] * 0.06)
                if p1 is not None:
                    cv2.arrowedLine(vis, p0, p1, col, 2, tipLength=0.25)
            cv2.circle(vis, p0, 5, (255, 255, 0), -1)
    return vis


def render_heatmap_view(seg: SegmentationResult | None, rgb_shape: tuple[int, int]) -> np.ndarray:
    """Render side-by-side confidence & normalized Shannon entropy colormaps."""
    h, w = rgb_shape
    if seg is None:
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(
            blank, "Run Perception to view Confidence & Entropy Maps",
            (max(10, w // 6), h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 190, 200), 1, cv2.LINE_AA,
        )
        return blank
    conf_u8 = np.clip(seg.confidence_map * 255.0, 0, 255).astype(np.uint8)
    ent_u8 = np.clip(seg.entropy_map * 255.0, 0, 255).astype(np.uint8)
    conf_cm = cv2.cvtColor(cv2.applyColorMap(conf_u8, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    ent_cm = cv2.cvtColor(cv2.applyColorMap(ent_u8, cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)
    cv2.putText(conf_cm, "SegmentGDA+ Confidence [0..1]", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(ent_cm, "Normalized Shannon Entropy [0..1]", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return np.concatenate([conf_cm, ent_cm], axis=1)


class OmniGDAStudioBackend:
    """Thread-safe controller holding ManiSkill 3 sim + OmniGDAAgent for the Web UI."""

    def __init__(self, backend: str = "tensorrt", seed: int = 42, num_objects: int = 6, include_novel: bool = True) -> None:
        self._lock = threading.RLock()
        self.cfg = load_config()
        self.cfg.setdefault("perception_2d", {})["backend"] = backend
        self.backend_name = backend
        self.seed = seed
        self.num_objects = num_objects
        self.include_novel = include_novel
        self.active_camera = "base_camera"

        self.sim = ClutteredTabletopSim(num_objects=num_objects, include_novel=include_novel, seed=seed)
        self.controller = FrankaGraspController(self.sim)
        self.agent = OmniGDAAgent.from_config(self.cfg)
        self.registered_classes: list[str] = []

        self._last_frame: RGBDFrame | None = None
        self._last_seg: SegmentationResult | None = None
        self._last_dets: list[Object3DDetection] = []
        self._last_relations: list[str] = []
        self._last_grasp: GraspCandidate | None = None
        self._last_metrics: dict[str, Any] = {}
        self._refresh_frame()

    def _refresh_frame(self) -> RGBDFrame:
        frame, _ = self.sim.capture_rgbd(self.active_camera)
        self._last_frame = frame
        return frame

    def _serialize_detections(self, dets: list[Object3DDetection]) -> list[dict[str, Any]]:
        out = []
        for d in dets:
            mw = d.modality_weights or {}
            out.append({
                "instance_id": int(d.instance_id),
                "class_name": d.class_name,
                "confidence": float(d.confidence),
                "entropy": float(d.entropy),
                "ood_score": float(d.ood_score),
                "is_ood": bool(d.is_ood),
                "centroid": [float(v) for v in d.centroid],
                "w_2d": float(mw.get("seg2d", 0.0) + mw.get("visual_clip", 0.0)),
                "w_3d": float(mw.get("geom_3d", 0.0) + mw.get("size_3d", 0.0)),
                "w_text": float(mw.get("text_clip", 0.0)),
            })
        return out

    def _vram_mb(self) -> float:
        return float(torch.cuda.memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            frame = self._last_frame or self._refresh_frame()
            ann = render_annotated_view(frame, self._last_seg, self._last_dets, self._last_grasp)
            hm = render_heatmap_view(self._last_seg, frame.rgb.shape[:2])
            return {
                "backend": self.backend_name,
                "seed": self.seed,
                "camera": self.active_camera,
                "scene_objects": list(self.sim.get_object_names()),
                "registered_classes": list(self.registered_classes),
                "annotated_jpg": _encode_rgb_jpeg_b64(ann),
                "heatmap_jpg": _encode_rgb_jpeg_b64(hm),
                "detections": self._serialize_detections(self._last_dets),
                "relations_summary": ", ".join(self._last_relations[:8]),
                "metrics": {**self._last_metrics, "vram_mb": self._vram_mb()},
            }

    def reset_scene(self, seed: int, num_objects: int, include_novel: bool) -> dict[str, Any]:
        with self._lock:
            need_rebuild = (num_objects != self.num_objects) or (include_novel != self.include_novel)
            self.seed, self.num_objects, self.include_novel = int(seed), int(num_objects), bool(include_novel)
            if need_rebuild:
                self.sim.close()
                self.sim = ClutteredTabletopSim(num_objects=self.num_objects, include_novel=self.include_novel, seed=self.seed)
                self.controller = FrankaGraspController(self.sim)
            else:
                self.sim.reset(seed=self.seed)
            self.active_camera = "base_camera"
            self._last_seg, self._last_dets, self._last_relations, self._last_grasp = None, [], [], None
            self._refresh_frame()
            return self.get_state()

    def switch_camera(self) -> dict[str, Any]:
        with self._lock:
            self.active_camera = "nbv_camera" if self.active_camera == "base_camera" else "base_camera"
            self._last_seg, self._last_dets, self._last_grasp = None, [], None
            self._refresh_frame()
            return self.get_state()

    def register_preset(self, group: str = "base", num_shots: int = 2) -> dict[str, Any]:
        with self._lock:
            pool = HELD_OUT_NOVEL_OBJECTS if group == "novel" else DEFAULT_BASE_OBJECTS
            targets = [n for n in pool if n in self.sim.get_object_names()] or list(pool[:4])
            templates = capture_few_shot_templates(targets, num_shots=num_shots, seed=self.seed)
            t0 = time.perf_counter()
            newly: list[str] = []
            for cname, samples in templates.items():
                prompt = samples[0].text_prompt if samples else cname.replace("_", " ")
                self.agent.register_object_few_shot(cname, samples, text_prompt=prompt)
                if cname not in self.registered_classes:
                    self.registered_classes.append(cname)
                newly.append(cname)
            reg_ms = (time.perf_counter() - t0) * 1000.0
            state = self.get_state()
            state["newly_registered"] = newly
            state["registration_ms"] = reg_ms
            return state

    def register_zero_shot(self, class_name: str, text_prompt: str) -> dict[str, Any]:
        with self._lock:
            t0 = time.perf_counter()
            self.agent.register_object_zero_shot(class_name, text_prompt)
            reg_ms = (time.perf_counter() - t0) * 1000.0
            if class_name not in self.registered_classes:
                self.registered_classes.append(class_name)
            state = self.get_state()
            state["registration_ms"] = reg_ms
            return state

    def perceive(self) -> dict[str, Any]:
        with self._lock:
            if not self.registered_classes:
                raise ValueError("No object classes registered yet. Click '+ Register Base Objects' first.")
            frame = self._refresh_frame()
            out = self.agent.perceive_frame(frame)
            self._last_seg = out.segmentation
            self._last_dets = list(out.scene_graph.objects)
            self._last_relations = [f"{r.subject_id} {r.relation} {r.object_id}" for r in out.scene_graph.relations]
            self._last_grasp = None
            grasp_ms = 0.0
            valid_dets = [d for d in self._last_dets if not d.is_ood]
            if valid_dets:
                best_det = max(valid_dets, key=lambda d: d.confidence)
                obstacles = [d.point_cloud_xyz for d in self._last_dets if d.instance_id != best_det.instance_id]
                obs_xyz = np.concatenate(obstacles, axis=0) if obstacles else None
                plan = self.agent.grasp_planner.plan(best_det, obstacle_xyz=obs_xyz, table_z=0.0, top_k=3)
                grasp_ms = float(plan.planning_latency_ms)
                if plan.candidates:
                    self._last_grasp = plan.candidates[0]
            self._last_metrics = {"seg2d_ms": float(out.seg2d_ms), "gmda3d_ms": float(out.gmda3d_ms), "grasp_ms": grasp_ms, "num_views": 1}
            return self.get_state()

    def execute_instruction(self, instruction: str, allow_nbv: bool = True, force_nbv: bool = False) -> dict[str, Any]:
        with self._lock:
            if not self.registered_classes:
                raise ValueError("No object classes registered yet. Click '+ Register Base/Novel Objects' first.")
            old_ent, old_conf = self.agent.entropy_nbv_threshold, self.agent.confidence_accept_threshold
            if force_nbv:
                self.agent.entropy_nbv_threshold = -1.0
                self.agent.confidence_accept_threshold = 2.0
            try:
                res = self.agent.execute_instruction(
                    self.sim, self.controller, instruction, allow_nbv=allow_nbv, record_frames=True
                )
            finally:
                self.agent.entropy_nbv_threshold = old_ent
                self.agent.confidence_accept_threshold = old_conf

            if self.agent.last_perception is not None:
                self._last_seg = self.agent.last_perception.segmentation
                self._last_dets = list(self.agent.last_perception.scene_graph.objects)
                self._last_relations = [f"{r.subject_id} {r.relation} {r.object_id}" for r in self.agent.last_perception.scene_graph.relations]
            self._last_grasp = res.grasp_candidate
            self._refresh_frame()
            self._last_metrics = {
                "seg2d_ms": float(self.agent.last_perception.seg2d_ms) if self.agent.last_perception else 0.0,
                "gmda3d_ms": float(self.agent.last_perception.gmda3d_ms) if self.agent.last_perception else 0.0,
                "grasp_ms": float(res.planning_latency_ms),
                "num_views": int(res.num_views_used),
            }
            raw_frames = self.controller.get_recorded_frames()
            stride = max(1, len(raw_frames) // 40)
            pb_b64 = [_encode_rgb_jpeg_b64(f, quality=78) for f in raw_frames[::stride]]
            state = self.get_state()
            state.update({
                "success": bool(res.success),
                "message": res.message,
                "target_class_name": res.target_class_name,
                "num_views_used": int(res.num_views_used),
                "grounding_reasoning": self.agent.last_grounding.reasoning if self.agent.last_grounding else "",
                "playback_frames": pb_b64,
            })
            return state


def create_app(backend_obj: OmniGDAStudioBackend) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index() -> Response:
        return Response(INDEX_HTML, mimetype="text/html")

    @app.route("/api/state", methods=["GET"])
    def api_state() -> Response:
        return jsonify(backend_obj.get_state())

    @app.route("/api/reset", methods=["POST"])
    def api_reset() -> Response:
        p = request.get_json(silent=True) or {}
        try:
            return jsonify(backend_obj.reset_scene(int(p.get("seed", 42)), int(p.get("num_objects", 6)), bool(p.get("include_novel", True))))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/switch_camera", methods=["POST"])
    def api_switch_camera() -> Response:
        return jsonify(backend_obj.switch_camera())

    @app.route("/api/register_preset", methods=["POST"])
    def api_register_preset() -> Response:
        p = request.get_json(silent=True) or {}
        try:
            return jsonify(backend_obj.register_preset(str(p.get("group", "base")), int(p.get("num_shots", 2))))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/register_zero_shot", methods=["POST"])
    def api_register_zero_shot() -> Response:
        p = request.get_json(silent=True) or {}
        try:
            return jsonify(backend_obj.register_zero_shot(str(p["class_name"]), str(p.get("text_prompt", p["class_name"]))))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/perceive", methods=["POST"])
    def api_perceive() -> Response:
        try:
            return jsonify(backend_obj.perceive())
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/execute", methods=["POST"])
    def api_execute() -> Response:
        p = request.get_json(silent=True) or {}
        try:
            return jsonify(backend_obj.execute_instruction(str(p["instruction"]), bool(p.get("allow_nbv", True)), bool(p.get("force_nbv", False))))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniGDA-Manip Interactive Web Studio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--backend", default="tensorrt", choices=["tensorrt", "pytorch", "onnx"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-objects", type=int, default=6)
    parser.add_argument("--include-novel", action="store_true", default=True)
    args = parser.parse_args()

    studio = OmniGDAStudioBackend(backend=args.backend, seed=args.seed, num_objects=args.num_objects, include_novel=args.include_novel)
    app = create_app(studio)
    print(f"OmniGDA-Manip Interactive Studio running at http://{args.host}:{args.port} (backend={args.backend})", flush=True)
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
