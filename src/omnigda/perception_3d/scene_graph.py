"""3D Spatial Scene Graph Builder and Spatial Relation Query Engine.

Builds ``SceneGraph3D`` instances from ``list[Object3DDetection]`` with smooth
distance/angle confidence scores for pairwise 3D relations (``left_of``, ``right_of``,
``front_of``, ``behind``, ``above``, ``near``) in robot/world frame, and supports both
pairwise relation queries and spatial superlatives (``leftmost``, ``rightmost``,
``nearest``, ``farthest``, ``highest``, ``frontmost``, ``backmost``).
"""

from __future__ import annotations

import numpy as np

from omnigda.core.schemas import Object3DDetection, SceneGraph3D, SceneRelation


class SceneGraphBuilder3D:
    """Builds and queries 3D pairwise spatial scene graphs in robot/world coordinates.

    Default coordinate convention follows ROS REP-103 / ManiSkill 3 tabletop world frame
    (robot facing ``+X``):
    - ``+X``: forward (away from robot -> ``behind`` has larger ``+X``, ``front_of`` smaller ``+X``)
    - ``+Y``: left (from robot's view -> ``left_of`` has larger ``+Y``, ``right_of`` smaller ``+Y``)
    - ``+Z``: up (``above`` has larger ``+Z``)

    Custom axis vectors can be supplied if operating in a camera-aligned or custom world frame.
    """

    def __init__(
        self,
        right_axis: tuple[float, float, float] = (0.0, -1.0, 0.0),
        forward_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
        up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
        min_directional_dist_m: float = 0.03,
        min_cos_angle: float = 0.55,
        min_vertical_dist_m: float = 0.03,
        max_above_horiz_dist_m: float = 0.22,
        near_threshold_m: float = 0.25,
        dist_scale_m: float = 0.08,
    ) -> None:
        self.right_axis = self._unit(right_axis)
        self.forward_axis = self._unit(forward_axis)
        self.up_axis = self._unit(up_axis)
        self.min_directional_dist_m = float(min_directional_dist_m)
        self.min_cos_angle = float(min_cos_angle)
        self.min_vertical_dist_m = float(min_vertical_dist_m)
        self.max_above_horiz_dist_m = float(max_above_horiz_dist_m)
        self.near_threshold_m = float(near_threshold_m)
        self.dist_scale_m = float(dist_scale_m)

    @staticmethod
    def _unit(vec: tuple[float, float, float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float64)
        norm = float(np.linalg.norm(arr))
        if norm < 1e-12:
            raise ValueError("Axis vector cannot be zero")
        return arr / norm

    def _directional_confidence(self, component: float, horiz_dist: float) -> float:
        cos_ang = abs(component) / max(horiz_dist, 1e-8)
        angle_score = float(np.clip((cos_ang - 0.5) / 0.5, 0.0, 1.0))
        dist_score = float(1.0 - np.exp(-abs(component) / max(self.dist_scale_m, 1e-4)))
        conf = 0.55 * angle_score + 0.45 * dist_score
        return float(np.clip(conf, 0.05, 1.0))

    def build(
        self,
        detections: list[Object3DDetection],
        timestamp_sec: float = 0.0,
    ) -> SceneGraph3D:
        """Compute all pairwise 3D spatial relations between detected objects."""
        relations: list[SceneRelation] = []
        n = len(detections)

        for i in range(n):
            subj = detections[i]
            c_subj = np.asarray(subj.centroid, dtype=np.float64)
            ext_subj = np.asarray(subj.obb_extent, dtype=np.float64)
            r_subj = 0.5 * float(max(ext_subj[0], ext_subj[1]))

            for j in range(n):
                if i == j:
                    continue
                ref = detections[j]
                c_ref = np.asarray(ref.centroid, dtype=np.float64)
                ext_ref = np.asarray(ref.obb_extent, dtype=np.float64)
                r_ref = 0.5 * float(max(ext_ref[0], ext_ref[1]))

                delta = c_subj - c_ref
                dist = float(np.linalg.norm(delta))

                d_right = float(np.dot(delta, self.right_axis))
                d_forward = float(np.dot(delta, self.forward_axis))
                d_up = float(np.dot(delta, self.up_axis))
                horiz_dist = float(np.hypot(d_right, d_forward))

                # Horizontal directional relations: left_of / right_of / front_of / behind
                if horiz_dist >= self.min_directional_dist_m:
                    cos_right = d_right / horiz_dist
                    if cos_right >= self.min_cos_angle:
                        relations.append(
                            SceneRelation(
                                subject_id=subj.instance_id,
                                relation="right_of",
                                object_id=ref.instance_id,
                                confidence=self._directional_confidence(d_right, horiz_dist),
                                distance_m=dist,
                            )
                        )
                    elif -cos_right >= self.min_cos_angle:
                        relations.append(
                            SceneRelation(
                                subject_id=subj.instance_id,
                                relation="left_of",
                                object_id=ref.instance_id,
                                confidence=self._directional_confidence(-d_right, horiz_dist),
                                distance_m=dist,
                            )
                        )

                    cos_fwd = d_forward / horiz_dist
                    if cos_fwd >= self.min_cos_angle:
                        relations.append(
                            SceneRelation(
                                subject_id=subj.instance_id,
                                relation="behind",
                                object_id=ref.instance_id,
                                confidence=self._directional_confidence(d_forward, horiz_dist),
                                distance_m=dist,
                            )
                        )
                    elif -cos_fwd >= self.min_cos_angle:
                        relations.append(
                            SceneRelation(
                                subject_id=subj.instance_id,
                                relation="front_of",
                                object_id=ref.instance_id,
                                confidence=self._directional_confidence(-d_forward, horiz_dist),
                                distance_m=dist,
                            )
                        )

                # Vertical relation: above (requires vertical separation + horizontal proximity)
                horiz_limit = max(self.max_above_horiz_dist_m, r_subj + r_ref + 0.08)
                if d_up >= self.min_vertical_dist_m and horiz_dist <= horiz_limit:
                    vert_align = d_up / max(dist, 1e-8)
                    if vert_align >= 0.45:
                        prox = float(np.exp(-((horiz_dist / max(horiz_limit, 1e-4)) ** 2)))
                        sep = float(1.0 - np.exp(-d_up / max(self.dist_scale_m, 1e-4)))
                        conf = float(np.clip(0.5 * vert_align + 0.25 * prox + 0.25 * sep, 0.05, 1.0))
                        relations.append(
                            SceneRelation(
                                subject_id=subj.instance_id,
                                relation="above",
                                object_id=ref.instance_id,
                                confidence=conf,
                                distance_m=dist,
                            )
                        )

                # Proximity relation: near
                if dist <= self.near_threshold_m:
                    sigma = max(0.6 * self.near_threshold_m, 1e-4)
                    conf = float(np.clip(np.exp(-0.5 * (dist / sigma) ** 2), 0.05, 1.0))
                    relations.append(
                        SceneRelation(
                            subject_id=subj.instance_id,
                            relation="near",
                            object_id=ref.instance_id,
                            confidence=conf,
                            distance_m=dist,
                        )
                    )

        return SceneGraph3D(
            objects=list(detections),
            relations=relations,
            timestamp_sec=float(timestamp_sec),
        )

    @staticmethod
    def _class_matches(class_name: str, query: str | None) -> bool:
        if query is None or not query.strip():
            return True
        c_norm = class_name.strip().lower().replace("_", " ")
        q_norm = query.strip().lower().replace("_", " ")
        return c_norm == q_norm or q_norm in c_norm or c_norm in q_norm

    def query_relation(
        self,
        scene_graph: SceneGraph3D,
        subject_class: str | None = None,
        relation: str | None = None,
        reference_class: str | None = None,
        min_confidence: float = 0.25,
        reference_point: np.ndarray | None = None,
    ) -> list[Object3DDetection]:
        """Query objects satisfying a pairwise spatial relation or spatial superlative.

        Supports:
        - Pairwise relations: ``left_of``, ``right_of``, ``front_of``, ``behind``, ``above``, ``near``.
        - Superlatives: ``leftmost``, ``rightmost``, ``nearest``, ``farthest``, ``highest``,
          ``frontmost``, ``backmost``.
        - Plain class filtering when ``relation is None``.
        """
        candidates = [
            obj for obj in scene_graph.objects if self._class_matches(obj.class_name, subject_class)
        ]
        if not candidates:
            return []

        if relation is None or not relation.strip():
            return sorted(candidates, key=lambda o: o.confidence, reverse=True)

        rel_norm = relation.strip().lower()

        # Superlative queries
        if rel_norm == "leftmost":
            return sorted(
                candidates,
                key=lambda o: float(np.dot(o.centroid, -self.right_axis)),
                reverse=True,
            )
        if rel_norm == "rightmost":
            return sorted(
                candidates,
                key=lambda o: float(np.dot(o.centroid, self.right_axis)),
                reverse=True,
            )
        if rel_norm == "highest":
            return sorted(
                candidates,
                key=lambda o: float(np.dot(o.centroid, self.up_axis)),
                reverse=True,
            )
        if rel_norm == "frontmost":
            return sorted(
                candidates,
                key=lambda o: float(np.dot(o.centroid, -self.forward_axis)),
                reverse=True,
            )
        if rel_norm == "backmost":
            return sorted(
                candidates,
                key=lambda o: float(np.dot(o.centroid, self.forward_axis)),
                reverse=True,
            )
        if rel_norm in {"nearest", "farthest"}:
            reverse = rel_norm == "farthest"
            if reference_class is not None and reference_class.strip():
                refs = [
                    obj
                    for obj in scene_graph.objects
                    if self._class_matches(obj.class_name, reference_class)
                ]
                if not refs:
                    return []
                scored: list[tuple[float, Object3DDetection]] = []
                for cand in candidates:
                    dists = [
                        float(np.linalg.norm(cand.centroid - r.centroid))
                        for r in refs
                        if r.instance_id != cand.instance_id
                    ]
                    if dists:
                        scored.append((min(dists), cand))
                scored.sort(key=lambda pair: pair[0], reverse=reverse)
                return [obj for _, obj in scored]

            origin = (
                np.asarray(reference_point, dtype=np.float64)
                if reference_point is not None
                else np.zeros(3, dtype=np.float64)
            )
            return sorted(
                candidates,
                key=lambda o: float(np.linalg.norm(o.centroid - origin)),
                reverse=reverse,
            )

        # Pairwise relation queries
        cand_ids = {obj.instance_id for obj in candidates}
        ref_ids = {
            obj.instance_id
            for obj in scene_graph.objects
            if self._class_matches(obj.class_name, reference_class)
        }
        best_score_by_id: dict[int, float] = {}
        for edge in scene_graph.relations:
            if (
                edge.relation == rel_norm
                and edge.subject_id in cand_ids
                and edge.object_id in ref_ids
                and edge.confidence >= float(min_confidence)
            ):
                prev = best_score_by_id.get(edge.subject_id, -1.0)
                if edge.confidence > prev:
                    best_score_by_id[edge.subject_id] = edge.confidence

        matched = [obj for obj in candidates if obj.instance_id in best_score_by_id]
        matched.sort(key=lambda o: (best_score_by_id[o.instance_id], o.confidence), reverse=True)
        return matched

    def resolve_query(
        self,
        scene_graph: SceneGraph3D,
        target_query: str | None = None,
        spatial_relation: str | None = None,
        reference_query: str | None = None,
        min_confidence: float = 0.25,
    ) -> Object3DDetection | None:
        """Return the single highest-scoring object matching a spatial query, or ``None``."""
        matches = self.query_relation(
            scene_graph=scene_graph,
            subject_class=target_query,
            relation=spatial_relation,
            reference_class=reference_query,
            min_confidence=min_confidence,
        )
        return matches[0] if matches else None

