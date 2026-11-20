"""3D Open-Vocabulary Semantic & Spatial Scene Graph Grounding.

Resolves structured ``ManipulationInstruction`` queries against a 3D ``SceneGraph3D`` by
fusing:
1. Open-vocabulary OpenCLIP text-embedding matching (with bilingual Chinese-to-English color
   and object alias normalization).
2. 3D Scene Graph pairwise spatial relation confidence (``left_of``, ``right_of``,
   ``front_of``, ``behind``, ``above``, ``near``) and spatial superlative ranking
   (``leftmost``, ``rightmost``, ``nearest``, ``farthest``, ``highest``, ``frontmost``,
   ``backmost``).
3. Active-perception uncertainty flagging (``needs_nbv``) based on target entropy,
   classification confidence, OOD status, or missing target/reference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from omnigda.core.gda_math import softmax
from omnigda.core.schemas import (
    ManipulationInstruction,
    Object3DDetection,
    SceneGraph3D,
)
from omnigda.perception_3d.encoders import OpenCLIPMultimodalEncoder
from omnigda.perception_3d.scene_graph import SceneGraphBuilder3D


@dataclass(slots=True)
class GroundingDecision:
    """Result of grounding a ``ManipulationInstruction`` in a 3D ``SceneGraph3D``."""

    found: bool
    target_object: Object3DDetection | None
    score: float
    confidence: float
    needs_nbv: bool
    destination_position: np.ndarray | None = None
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Bilingual (ZH -> EN) & English lexical alias tables
# ---------------------------------------------------------------------------

# Ordered longest-first at runtime to avoid partial substring corruption (e.g., 洋红 before 红)
_ZH_TO_EN_PHRASES: dict[str, str] = {
    # Multi-character colors
    "洋红色": "magenta",
    "品红色": "magenta",
    "玫红色": "magenta",
    "粉红色": "magenta",
    "洋红": "magenta",
    "品红": "magenta",
    "玫红": "magenta",
    "深红色": "red",
    "朱红色": "red",
    "大红色": "red",
    "红色": "red",
    "深蓝色": "blue",
    "天蓝色": "cyan",
    "青蓝色": "cyan",
    "蓝色": "blue",
    "绿色": "green",
    "黄色": "yellow",
    "橙色": "orange",
    "橘红色": "orange",
    "橘色": "orange",
    "紫色": "purple",
    "灰色": "gray",
    "银灰色": "gray",
    "银色": "gray",
    "青色": "cyan",
    "白色": "white",
    "黑色": "black",
    # Single-character colors
    "红": "red",
    "蓝": "blue",
    "绿": "green",
    "黄": "yellow",
    "橙": "orange",
    "橘": "orange",
    "紫": "purple",
    "灰": "gray",
    "青": "cyan",
    "白": "white",
    "黑": "black",
    # Objects / shapes (multi-character first)
    "马克杯": "mug",
    "水杯": "mug",
    "茶杯": "mug",
    "杯子": "mug",
    "杯": "mug",
    "水瓶": "bottle",
    "瓶子": "bottle",
    "瓶": "bottle",
    "盒子": "box",
    "箱子": "box",
    "盒": "box",
    "箱": "box",
    "立方体": "cube",
    "正方形": "cube",
    "正方体": "cube",
    "木块": "block",
    "方块": "block",
    "块": "block",
    "圆柱体": "cylinder",
    "圆柱": "cylinder",
    "易拉罐": "cylinder",
    "罐子": "cylinder",
    "柱": "cylinder",
    "罐": "cylinder",
    "长条": "bar",
    "条状": "bar",
    "木条": "bar",
    "棒": "bar",
    "条": "bar",
    "T形工具": "t tool",
    "T型工具": "t tool",
    "t形工具": "t tool",
    "t型工具": "t tool",
    "工具": "tool",
    "扳手": "tool",
    "托盘": "tray",
    "盘子": "tray",
    "盘": "tray",
    "物体": "object",
    "物品": "object",
    "东西": "object",
}

_ZH_SORTED_KEYS: tuple[str, ...] = tuple(
    sorted(_ZH_TO_EN_PHRASES.keys(), key=len, reverse=True)
)

_EN_CANONICAL_SYNONYMS: dict[str, str] = {
    # Color canonicals
    "crimson": "red",
    "scarlet": "red",
    "ruby": "red",
    "maroon": "red",
    "azure": "blue",
    "navy": "blue",
    "cobalt": "blue",
    "sapphire": "blue",
    "emerald": "green",
    "lime": "green",
    "olive": "green",
    "gold": "yellow",
    "golden": "yellow",
    "amber": "yellow",
    "tangerine": "orange",
    "violet": "purple",
    "lavender": "purple",
    "plum": "purple",
    "grey": "gray",
    "silver": "gray",
    "metallic": "gray",
    "turquoise": "cyan",
    "teal": "cyan",
    "aqua": "cyan",
    "fuchsia": "magenta",
    "pink": "magenta",
    # Object / shape canonicals
    "cup": "mug",
    "tumbler": "mug",
    "flask": "bottle",
    "jug": "bottle",
    "carton": "box",
    "case": "box",
    "cube": "block",
    "brick": "block",
    "can": "cylinder",
    "tin": "cylinder",
    "tube": "cylinder",
    "rod": "bar",
    "stick": "bar",
    "beam": "bar",
    "wrench": "tool",
    "spanner": "tool",
    "plate": "tray",
    "dish": "tray",
    "platter": "tray",
}

_COLOR_CANONICALS: frozenset[str] = frozenset(
    {"red", "blue", "green", "yellow", "orange", "purple", "gray", "cyan", "magenta", "white", "black"}
)

_GENERIC_QUERIES: frozenset[str] = frozenset(
    {"", "object", "item", "thing", "anything", "one", "it", "target", "物体", "物品", "东西"}
)

_STOP_WORDS: frozenset[str] = frozenset(
    {"a", "an", "the", "of", "with", "shaped", "on", "in", "for", "and", "to", "color", "colored"}
)

_PAIRWISE_RELATIONS: frozenset[str] = frozenset(
    {"left_of", "right_of", "front_of", "behind", "above", "near"}
)

_SUPERLATIVE_RELATIONS: frozenset[str] = frozenset(
    {"leftmost", "rightmost", "nearest", "farthest", "highest", "frontmost", "backmost"}
)


def normalize_bilingual_query(query: str) -> str:
    """Translate Chinese color/object terms into clean English while preserving English text."""
    text = str(query).strip()
    if not text:
        return ""

    # Replace underscores / hyphens with spaces for uniform matching
    out = re.sub(r"[_-]+", " ", text)

    # Only run Chinese phrase substitution when CJK characters are present
    if re.search(r"[\u4e00-\u9fff]", out):
        for zh_key in _ZH_SORTED_KEYS:
            if zh_key in out:
                out = out.replace(zh_key, f" {_ZH_TO_EN_PHRASES[zh_key]} ")
        # Drop remaining Chinese filler particles like 的 / 色 / 个
        out = re.sub(r"[\u4e00-\u9fff]+", " ", out)

    out = re.sub(r"[^a-zA-Z0-9\s]", " ", out)
    out = re.sub(r"\s+", " ", out).strip().lower()
    return out


def _canonical_tokens(text: str) -> list[str]:
    norm = normalize_bilingual_query(text)
    if not norm:
        return []
    tokens: list[str] = []
    for tok in norm.split():
        if tok in _STOP_WORDS:
            continue
        tokens.append(_EN_CANONICAL_SYNONYMS.get(tok, tok))
    return tokens


class SpatialGrounder3D:
    """Open-vocabulary semantic + 3D spatial scene graph grounder."""

    def __init__(
        self,
        clip_encoder: OpenCLIPMultimodalEncoder | None = None,
        scene_graph_builder: SceneGraphBuilder3D | None = None,
        class_text_prompts: dict[str, str] | None = None,
        entropy_nbv_threshold: float = 0.85,
        confidence_accept_threshold: float = 0.60,
        min_grounding_score: float = 0.25,
        semantic_weight: float = 0.60,
        spatial_weight: float = 0.40,
        place_z_offset_m: float = 0.03,
        reference_origin: tuple[float, float, float] | None = None,
    ) -> None:
        self.clip_encoder = clip_encoder
        self.scene_graph_builder = (
            scene_graph_builder if scene_graph_builder is not None else SceneGraphBuilder3D()
        )
        self.class_text_prompts: dict[str, str] = dict(class_text_prompts or {})
        self.entropy_nbv_threshold = float(entropy_nbv_threshold)
        self.confidence_accept_threshold = float(confidence_accept_threshold)
        self.min_grounding_score = float(min_grounding_score)
        self.semantic_weight = float(semantic_weight)
        self.spatial_weight = float(spatial_weight)
        self.place_z_offset_m = float(place_z_offset_m)
        self.reference_origin = (
            np.asarray(reference_origin, dtype=np.float64)
            if reference_origin is not None
            else None
        )

    def register_class_prompt(self, class_name: str, text_prompt: str | None = None) -> None:
        """Register or update a descriptive text prompt for a known object class."""
        prompt = (text_prompt or class_name.replace("_", " ")).strip()
        self.class_text_prompts[class_name] = prompt

    def _lexical_score(self, query: str, class_name: str) -> float:
        """Compute bilingual-normalized lexical + color-aware similarity in ``[0, 1]``."""
        norm_q = normalize_bilingual_query(query)
        norm_c = normalize_bilingual_query(class_name)
        if not norm_q:
            return 1.0
        if norm_q == norm_c:
            return 1.0

        q_toks = _canonical_tokens(query)
        if not q_toks:
            return 1.0

        c_name_toks = _canonical_tokens(class_name)
        prompt = self.class_text_prompts.get(class_name, "")
        c_prompt_toks = _canonical_tokens(prompt) if prompt else []
        c_all_set = set(c_name_toks) | set(c_prompt_toks)
        c_name_set = set(c_name_toks)

        if not c_all_set:
            return 0.0

        q_set = set(q_toks)
        if q_set == c_name_set:
            return 1.0

        # Recall of query tokens in class name (stronger) and full prompt
        inter_name = len(q_set & c_name_set)
        inter_all = len(q_set & c_all_set)
        recall_name = inter_name / float(len(q_set))
        recall_all = inter_all / float(len(q_set))
        precision_name = inter_name / float(max(len(c_name_set), 1))

        score = 0.65 * recall_all + 0.25 * recall_name + 0.10 * precision_name

        # Check explicit color agreement/conflict
        q_colors = q_set & _COLOR_CANONICALS
        c_colors = c_all_set & _COLOR_CANONICALS
        if q_colors and c_colors:
            if q_colors & c_colors:
                score = min(1.0, score + 0.08)
            else:
                score *= 0.25

        return float(np.clip(score, 0.0, 1.0))

    def score_semantics(
        self,
        query: str,
        objects: list[Object3DDetection],
    ) -> dict[int, float]:
        """Score every object in ``objects`` against free-form ``query`` in ``[0, 1]``."""
        if not objects:
            return {}

        norm_q = normalize_bilingual_query(query)
        if query.strip().lower() in _GENERIC_QUERIES or norm_q in _GENERIC_QUERIES:
            return {obj.instance_id: (0.2 if obj.is_ood else 1.0) for obj in objects}

        unique_classes = list(dict.fromkeys(obj.class_name for obj in objects))
        lex_by_class = {cname: self._lexical_score(query, cname) for cname in unique_classes}

        clip_by_class: dict[str, float] | None = None
        if self.clip_encoder is not None and norm_q:
            q_emb = self.clip_encoder.encode_text(norm_q, use_templates=True).astype(np.float64)
            raw_sims: list[float] = []
            for cname in unique_classes:
                human_name = cname.replace("_", " ").strip()
                c_emb = self.clip_encoder.encode_text(human_name, use_templates=True).astype(
                    np.float64
                )
                sim = float(np.dot(q_emb, c_emb))
                prompt = self.class_text_prompts.get(cname, "").strip()
                if prompt and prompt.lower() != human_name.lower():
                    p_emb = self.clip_encoder.encode_text(prompt, use_templates=True).astype(
                        np.float64
                    )
                    sim = max(sim, float(np.dot(q_emb, p_emb)))
                raw_sims.append(sim)

            sims_arr = np.asarray(raw_sims, dtype=np.float64)
            abs_scores = np.clip((sims_arr - 0.58) / 0.36, 0.0, 1.0)

            if len(unique_classes) >= 2:
                rel_probs = softmax(sims_arr * 18.0)
                rel_ratio = rel_probs / max(float(np.max(rel_probs)), 1e-8)
                cal_scores = abs_scores * (0.45 + 0.55 * rel_ratio)
            else:
                cal_scores = abs_scores

            clip_by_class = {
                cname: float(np.clip(cal_scores[idx], 0.0, 1.0))
                for idx, cname in enumerate(unique_classes)
            }

        # Combine CLIP + lexical scores per unique class (same-class instances get identical scores)
        q_colors = set(_canonical_tokens(query)) & _COLOR_CANONICALS
        class_scores: dict[str, float] = {}
        for cname in unique_classes:
            lex = lex_by_class[cname]
            if clip_by_class is not None:
                clip_s = clip_by_class[cname]
                combined = 0.65 * clip_s + 0.35 * lex
                if lex >= 0.99:
                    combined = max(combined, 0.98)
            else:
                combined = lex

            # Soft guard against explicit color contradiction
            c_colors = (
                set(_canonical_tokens(cname))
                | set(_canonical_tokens(self.class_text_prompts.get(cname, "")))
            ) & _COLOR_CANONICALS
            if q_colors and c_colors and not (q_colors & c_colors):
                combined *= 0.45

            class_scores[cname] = float(np.clip(combined, 0.0, 1.0))

        return {obj.instance_id: class_scores[obj.class_name] for obj in objects}

    def _soft_directional_score(
        self,
        subj: Object3DDetection,
        ref: Object3DDetection,
        relation: str,
    ) -> float:
        """Conservative directional fallback when a valid side alignment misses strict SG angle."""
        delta = subj.centroid - ref.centroid
        dist = float(np.linalg.norm(delta))
        if dist < 0.02:
            return 0.0
        b = self.scene_graph_builder
        if relation == "left_of":
            proj = float(np.dot(delta, -b.right_axis))
        elif relation == "right_of":
            proj = float(np.dot(delta, b.right_axis))
        elif relation == "front_of":
            proj = float(np.dot(delta, -b.forward_axis))
        elif relation == "behind":
            proj = float(np.dot(delta, b.forward_axis))
        elif relation == "above":
            proj = float(np.dot(delta, b.up_axis))
        elif relation == "near":
            return float(0.40 * np.exp(-dist / max(b.near_threshold_m, 0.1)))
        else:
            return 0.0

        if proj <= 0.015:
            return 0.0
        cos_a = float(np.clip(proj / dist, 0.0, 1.0))
        return float(0.45 * cos_a)

    def _score_pairwise_spatial(
        self,
        scene_graph: SceneGraph3D,
        relation: str,
        reference_query: str | None,
    ) -> tuple[dict[int, float], list[Object3DDetection], str]:
        """Score every object for a pairwise spatial relation against grounded reference(s)."""
        objects = scene_graph.objects
        if not reference_query:
            return {}, [], "missing reference query"

        ref_sem = self.score_semantics(reference_query, objects)
        if not ref_sem:
            return {}, [], f"no candidates for reference {reference_query!r}"

        best_ref_score = max(ref_sem.values())
        if best_ref_score < self.min_grounding_score:
            return {}, [], f"reference {reference_query!r} not found (best_sem={best_ref_score:.2f})"

        ref_thresh = max(self.min_grounding_score, best_ref_score - 0.15)
        refs = [o for o in objects if ref_sem[o.instance_id] >= ref_thresh]
        ref_ids = {r.instance_id for r in refs}

        # 1. Exact SceneGraph3D relation edges
        edge_scores: dict[int, float] = {o.instance_id: 0.0 for o in objects}
        any_edge = False
        for edge in scene_graph.relations:
            if edge.relation == relation and edge.object_id in ref_ids:
                weighted = float(edge.confidence) * ref_sem[edge.object_id]
                if weighted > edge_scores.get(edge.subject_id, 0.0):
                    edge_scores[edge.subject_id] = weighted
                    any_edge = True

        # 2. If no strict edge existed (e.g., wide angle), apply conservative directional fallback
        if not any_edge:
            for obj in objects:
                best_fb = 0.0
                for ref in refs:
                    if ref.instance_id == obj.instance_id:
                        continue
                    fb = self._soft_directional_score(obj, ref, relation) * ref_sem[ref.instance_id]
                    if fb > best_fb:
                        best_fb = fb
                edge_scores[obj.instance_id] = best_fb

        ref_desc = ", ".join(f"#{r.instance_id}:{r.class_name}" for r in refs)
        return edge_scores, refs, f"relation={relation} wrt [{ref_desc}]"

    def _score_superlative_spatial(
        self,
        scene_graph: SceneGraph3D,
        relation: str,
        plausible_ids: set[int],
        reference_query: str | None = None,
    ) -> tuple[dict[int, float], str]:
        """Rank semantically plausible objects along SceneGraphBuilder3D's spatial axes."""
        b = self.scene_graph_builder
        plausible_objs = [o for o in scene_graph.objects if o.instance_id in plausible_ids]
        scores: dict[int, float] = {o.instance_id: 0.0 for o in scene_graph.objects}
        if not plausible_objs:
            return scores, f"no plausible candidates for {relation}"

        raw_vals: dict[int, float] = {}
        if relation == "leftmost":
            for o in plausible_objs:
                raw_vals[o.instance_id] = float(np.dot(o.centroid, -b.right_axis))
        elif relation == "rightmost":
            for o in plausible_objs:
                raw_vals[o.instance_id] = float(np.dot(o.centroid, b.right_axis))
        elif relation == "highest":
            for o in plausible_objs:
                raw_vals[o.instance_id] = float(np.dot(o.centroid, b.up_axis))
        elif relation == "frontmost":
            for o in plausible_objs:
                raw_vals[o.instance_id] = float(np.dot(o.centroid, -b.forward_axis))
        elif relation == "backmost":
            for o in plausible_objs:
                raw_vals[o.instance_id] = float(np.dot(o.centroid, b.forward_axis))
        elif relation in {"nearest", "farthest"}:
            sign = -1.0 if relation == "nearest" else 1.0
            ref_objs: list[Object3DDetection] = []
            if reference_query:
                ref_sem = self.score_semantics(reference_query, scene_graph.objects)
                best_rs = max(ref_sem.values(), default=0.0)
                if best_rs >= self.min_grounding_score:
                    ref_objs = [
                        o for o in scene_graph.objects if ref_sem[o.instance_id] >= best_rs - 0.15
                    ]
            origin = (
                self.reference_origin
                if self.reference_origin is not None
                else np.zeros(3, dtype=np.float64)
            )
            for o in plausible_objs:
                if ref_objs:
                    dists = [
                        float(np.linalg.norm(o.centroid - r.centroid))
                        for r in ref_objs
                        if r.instance_id != o.instance_id
                    ]
                    dist = min(dists) if dists else float(np.linalg.norm(o.centroid - origin))
                else:
                    dist = float(np.linalg.norm(o.centroid - origin))
                raw_vals[o.instance_id] = sign * dist
        else:
            for o in plausible_objs:
                raw_vals[o.instance_id] = 1.0

        vals = list(raw_vals.values())
        v_min = min(vals)
        v_max = max(vals)
        span = v_max - v_min
        for oid, val in raw_vals.items():
            scores[oid] = 1.0 if span < 1e-6 else float(0.20 + 0.80 * (val - v_min) / span)

        return scores, f"superlative={relation} over {len(plausible_objs)} candidate(s)"

    def _resolve_destination_position(
        self,
        instruction: ManipulationInstruction,
        scene_graph: SceneGraph3D,
        target: Object3DDetection,
    ) -> tuple[np.ndarray | None, str]:
        """Resolve 3D placement coordinates for ``pick_and_place`` instructions."""
        if instruction.action != "pick_and_place" or not instruction.destination_query:
            return None, ""

        others = [o for o in scene_graph.objects if o.instance_id != target.instance_id]
        if not others:
            return None, "no destination object in scene"

        dest_sem = self.score_semantics(instruction.destination_query, others)
        best_dest = max(others, key=lambda o: (dest_sem[o.instance_id], o.confidence))
        best_score = dest_sem[best_dest.instance_id]
        if best_score < self.min_grounding_score:
            return (
                None,
                f"destination {instruction.destination_query!r} not found (best={best_score:.2f})",
            )

        raw_lower = instruction.raw_text.lower()
        b = self.scene_graph_builder
        lateral_offset = (
            0.5 * float(np.max(best_dest.obb_extent[:2]))
            + 0.5 * float(np.max(target.obb_extent[:2]))
            + 0.045
        )
        place_z_table = max(float(target.obb_center[2]), 0.025)

        if any(tok in raw_lower for tok in ("to the left of", "左边", "左侧")) and (
            instruction.spatial_relation != "left_of"
        ):
            pos = best_dest.obb_center + lateral_offset * (-b.right_axis)
            pos = np.array([pos[0], pos[1], place_z_table], dtype=np.float64)
            return pos, f"dest=#{best_dest.instance_id}:{best_dest.class_name} (left_of)"

        if any(tok in raw_lower for tok in ("to the right of", "右边", "右侧")) and (
            instruction.spatial_relation != "right_of"
        ):
            pos = best_dest.obb_center + lateral_offset * b.right_axis
            pos = np.array([pos[0], pos[1], place_z_table], dtype=np.float64)
            return pos, f"dest=#{best_dest.instance_id}:{best_dest.class_name} (right_of)"

        if any(tok in raw_lower for tok in ("next to", "beside", "near", "旁边", "附近", "边上")):
            # Place to the side that keeps Y closer to tabletop center (0.0)
            side_axis = -b.right_axis if best_dest.obb_center[1] <= 0.0 else b.right_axis
            pos = best_dest.obb_center + lateral_offset * side_axis
            pos = np.array([pos[0], pos[1], place_z_table], dtype=np.float64)
            return pos, f"dest=#{best_dest.instance_id}:{best_dest.class_name} (near)"

        # Default: place directly on top / inside destination object
        top_z = float(best_dest.obb_center[2] + 0.5 * best_dest.obb_extent[2] + self.place_z_offset_m)
        pos = np.array(
            [float(best_dest.obb_center[0]), float(best_dest.obb_center[1]), top_z],
            dtype=np.float64,
        )
        return pos, f"dest=#{best_dest.instance_id}:{best_dest.class_name} (on_top)"

    def ground(
        self,
        instruction: ManipulationInstruction,
        scene_graph: SceneGraph3D,
    ) -> GroundingDecision:
        """Ground ``instruction`` against ``scene_graph`` and decide whether NBV is needed."""
        objects = scene_graph.objects
        if not objects:
            return GroundingDecision(
                found=False,
                target_object=None,
                score=0.0,
                confidence=0.0,
                needs_nbv=True,
                destination_position=None,
                reasoning="3D scene graph is empty; requesting NBV view.",
            )

        sem_scores = self.score_semantics(instruction.target_query, objects)
        best_sem = max(sem_scores.values(), default=0.0)
        rel = instruction.spatial_relation.strip().lower() if instruction.spatial_relation else None

        spatial_scores: dict[int, float] | None = None
        spatial_note = "no spatial constraint"
        excluded_ids: set[int] = set()

        if rel in _PAIRWISE_RELATIONS:
            spatial_scores, refs, spatial_note = self._score_pairwise_spatial(
                scene_graph=scene_graph,
                relation=rel,
                reference_query=instruction.reference_query,
            )
            if not refs:
                return GroundingDecision(
                    found=False,
                    target_object=None,
                    score=0.0,
                    confidence=0.0,
                    needs_nbv=True,
                    destination_position=None,
                    reasoning=f"Pairwise grounding failed: {spatial_note}.",
                )
            if len(refs) == 1:
                excluded_ids.add(refs[0].instance_id)
        elif rel in _SUPERLATIVE_RELATIONS:
            plausible_thresh = max(self.min_grounding_score, best_sem - 0.22)
            plausible_ids = {
                o.instance_id for o in objects if sem_scores[o.instance_id] >= plausible_thresh
            }
            spatial_scores, spatial_note = self._score_superlative_spatial(
                scene_graph=scene_graph,
                relation=rel,
                plausible_ids=plausible_ids,
                reference_query=instruction.reference_query,
            )

        w_sum = max(self.semantic_weight + self.spatial_weight, 1e-6)
        w_sem = self.semantic_weight / w_sum
        w_spa = self.spatial_weight / w_sum

        best_obj: Object3DDetection | None = None
        best_total = -1.0
        best_sem_val = 0.0
        best_spa_val = 0.0

        for obj in objects:
            if obj.instance_id in excluded_ids:
                continue
            s_val = sem_scores[obj.instance_id]
            if spatial_scores is not None:
                p_val = spatial_scores.get(obj.instance_id, 0.0)
                if rel in _PAIRWISE_RELATIONS and p_val <= 0.0:
                    # Strictly penalize candidates violating a pairwise spatial relation
                    total = 0.30 * w_sem * s_val
                else:
                    total = w_sem * s_val + w_spa * p_val
            else:
                p_val = 1.0
                total = s_val

            if (total > best_total + 1e-9) or (
                abs(total - best_total) <= 1e-9
                and best_obj is not None
                and obj.confidence > best_obj.confidence
            ):
                best_total = total
                best_obj = obj
                best_sem_val = s_val
                best_spa_val = p_val

        # Validate grounding acceptance
        if best_obj is None or best_sem_val < self.min_grounding_score:
            return GroundingDecision(
                found=False,
                target_object=None,
                score=max(best_total, 0.0),
                confidence=0.0,
                needs_nbv=True,
                destination_position=None,
                reasoning=(
                    f"Target {instruction.target_query!r} not confidently matched "
                    f"(best_sem={best_sem_val:.2f} < {self.min_grounding_score:.2f}); requesting NBV."
                ),
            )

        if rel in _PAIRWISE_RELATIONS and best_spa_val <= 0.0:
            return GroundingDecision(
                found=False,
                target_object=None,
                score=float(best_total),
                confidence=float(best_obj.confidence),
                needs_nbv=True,
                destination_position=None,
                reasoning=(
                    f"No {instruction.target_query!r} satisfies {spatial_note}; requesting NBV."
                ),
            )

        dest_pos, dest_note = self._resolve_destination_position(
            instruction=instruction,
            scene_graph=scene_graph,
            target=best_obj,
        )

        nbv_reasons: list[str] = []
        if best_obj.is_ood:
            nbv_reasons.append(f"is_ood(d2={best_obj.ood_score:.1f})")
        if best_obj.entropy > self.entropy_nbv_threshold:
            nbv_reasons.append(
                f"high_entropy({best_obj.entropy:.2f}>{self.entropy_nbv_threshold:.2f})"
            )
        if best_obj.confidence < self.confidence_accept_threshold:
            nbv_reasons.append(
                f"low_conf({best_obj.confidence:.2f}<{self.confidence_accept_threshold:.2f})"
            )
        if instruction.action == "pick_and_place" and instruction.destination_query and dest_pos is None:
            nbv_reasons.append("missing_destination")

        needs_nbv = len(nbv_reasons) > 0
        reasoning_parts = [
            f"target=#{best_obj.instance_id}:{best_obj.class_name}",
            f"sem={best_sem_val:.2f}",
            f"spa={best_spa_val:.2f} ({spatial_note})",
            f"conf={best_obj.confidence:.2f}",
            f"H={best_obj.entropy:.2f}",
        ]
        if dest_note:
            reasoning_parts.append(dest_note)
        if nbv_reasons:
            reasoning_parts.append("NBV=[" + ", ".join(nbv_reasons) + "]")

        return GroundingDecision(
            found=True,
            target_object=best_obj,
            score=float(np.clip(best_total, 0.0, 1.0)),
            confidence=float(best_obj.confidence),
            needs_nbv=needs_nbv,
            destination_position=dest_pos,
            reasoning="; ".join(reasoning_parts),
        )
