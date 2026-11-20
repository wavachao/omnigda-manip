"""Unit, OpenCLIP 3D grounding, NBV fusion, and real ManiSkill 3 closed-loop tests for OmniGDAAgent."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from omnigda.agent import (
    InstructionParserProtocol,
    OmniGDAAgent,
    OpenAICompatibleVLMParser,
    RuleBasedMultimodalParser,
    SpatialGrounder3D,
    normalize_bilingual_query,
)
from omnigda.core.config import load_config
from omnigda.core.schemas import Object3DDetection
from omnigda.perception_2d.backbone import DINOv2MultiScaleExtractor
from omnigda.perception_2d.segment_gda import SegmentGDAPlus
from omnigda.perception_3d.encoders import OpenCLIPMultimodalEncoder
from omnigda.perception_3d.multimodal_gmda import PartialAwareMultimodalGMDA
from omnigda.perception_3d.scene_graph import SceneGraphBuilder3D
from omnigda.sim import (
    OBJECT_CATALOG,
    ClutteredTabletopSim,
    FrankaGraspController,
    capture_few_shot_templates,
)

# ---------------------------------------------------------------------------
# Helpers & shared GPU fixtures (RTX 3060)
# ---------------------------------------------------------------------------


def _make_det(
    instance_id: int,
    class_name: str,
    centroid: tuple[float, float, float],
    extent: tuple[float, float, float] = (0.06, 0.06, 0.06),
    confidence: float = 0.92,
    entropy: float = 0.12,
    is_ood: bool = False,
    posterior_probs: dict[str, float] | None = None,
    num_pts: int = 80,
    seed: int = 0,
) -> Object3DDetection:
    rng = np.random.default_rng(seed + instance_id * 17)
    c = np.asarray(centroid, dtype=np.float64)
    ext = np.asarray(extent, dtype=np.float64)
    pts = (c[None, :] + rng.uniform(-0.5, 0.5, size=(num_pts, 3)) * ext[None, :]).astype(
        np.float32
    )
    probs = posterior_probs or {class_name: confidence}
    return Object3DDetection(
        instance_id=instance_id,
        class_name=class_name,
        confidence=confidence,
        entropy=entropy,
        ood_score=6.0 if not is_ood else 80.0,
        is_ood=is_ood,
        centroid=c,
        obb_center=c.copy(),
        obb_extent=ext,
        obb_rotation=np.eye(3, dtype=np.float64),
        point_cloud_xyz=pts,
        posterior_probs=probs,
    )


@pytest.fixture(scope="module")
def clip_encoder() -> OpenCLIPMultimodalEncoder:
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return OpenCLIPMultimodalEncoder(
        model_name=cfg["perception_3d"]["clip_model"],
        pretrained=cfg["perception_3d"]["clip_pretrained"],
        device=device,
    )


@pytest.fixture(scope="module")
def dinov2_extractor() -> DINOv2MultiScaleExtractor:
    cfg = load_config()
    p2d = cfg["perception_2d"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return DINOv2MultiScaleExtractor(
        model_name=p2d["backbone"],
        input_size=int(p2d["input_size"]),
        feature_layers=tuple(p2d["feature_layers"]),
        device=device,
        fp16=bool(p2d.get("fp16", True)) and device == "cuda",
    )


# ---------------------------------------------------------------------------
# 1. Bilingual EN + ZH Instruction Parsing & VLM Fallback Tests
# ---------------------------------------------------------------------------


def test_rule_based_parser_english_commands() -> None:
    parser = RuleBasedMultimodalParser()
    assert isinstance(parser, InstructionParserProtocol)

    # Pick + novel object
    inst = parser.parse("Please pick up the cyan t tool.")
    assert inst.action == "pick"
    assert inst.target_query == "cyan t tool"
    assert inst.spatial_relation is None
    assert inst.reference_query is None
    assert inst.destination_query is None

    # Pairwise spatial relations
    cases = [
        ("grab the red mug to the left of the blue bottle", "pick", "red mug", "left_of", "blue bottle"),
        ("grasp the yellow block right of the green box", "pick", "yellow block", "right_of", "green box"),
        ("pick the orange cylinder in front of the purple bar", "pick", "orange cylinder", "front_of", "purple bar"),
        ("find the blue bottle behind the red mug", "inspect", "blue bottle", "behind", "red mug"),
        ("locate the cube on top of the gray tray", "inspect", "cube", "above", "gray tray"),
        ("grab the mug next to the green box", "pick", "mug", "near", "green box"),
    ]
    for text, exp_act, exp_tgt, exp_rel, exp_ref in cases:
        out = parser.parse(text)
        assert out.action == exp_act, text
        assert out.target_query == exp_tgt, text
        assert out.spatial_relation == exp_rel, text
        assert out.reference_query == exp_ref, text

    # Superlatives
    assert parser.parse("pick up the leftmost red mug").spatial_relation == "leftmost"
    assert parser.parse("pick up the leftmost red mug").target_query == "red mug"
    assert parser.parse("grab the rightmost bottle").spatial_relation == "rightmost"
    assert parser.parse("grasp the nearest yellow block").spatial_relation == "nearest"

    # Pick and place (including combined with spatial relation)
    pap1 = parser.parse("put the red mug on top of the gray tray")
    assert pap1.action == "pick_and_place"
    assert pap1.target_query == "red mug"
    assert pap1.destination_query == "gray tray"

    pap2 = parser.parse("place the green box next to the blue bottle")
    assert pap2.action == "pick_and_place"
    assert pap2.target_query == "green box"
    assert pap2.destination_query == "blue bottle"

    pap3 = parser.parse("put the mug to the left of the bottle onto the gray tray")
    assert pap3.action == "pick_and_place"
    assert pap3.target_query == "mug"
    assert pap3.spatial_relation == "left_of"
    assert pap3.reference_query == "bottle"
    assert pap3.destination_query == "gray tray"

    pap4 = parser.parse("pick up the leftmost yellow block and put it into the gray tray")
    assert pap4.action == "pick_and_place"
    assert pap4.target_query == "yellow block"
    assert pap4.spatial_relation == "leftmost"
    assert pap4.destination_query == "gray tray"


def test_rule_based_parser_chinese_commands() -> None:
    parser = RuleBasedMultimodalParser()

    # Basic pick & inspect
    p1 = parser.parse("请抓起蓝色的瓶子")
    assert p1.action == "pick"
    assert p1.target_query == "蓝色瓶子"

    p2 = parser.parse("拿起青色工具")
    assert p2.action == "pick"
    assert p2.target_query == "青色工具"

    i1 = parser.parse("找到绿色盒子")
    assert i1.action == "inspect"
    assert i1.target_query == "绿色盒子"

    i2 = parser.parse("检查红色杯子")
    assert i2.action == "inspect"
    assert i2.target_query == "红色杯子"

    # Pairwise spatial relations in Chinese
    zh_spatial = [
        ("抓起蓝色瓶子左边的红色杯子", "pick", "红色杯子", "left_of", "蓝色瓶子"),
        ("拿起绿色盒子右边的黄色方块", "pick", "黄色方块", "right_of", "绿色盒子"),
        ("找到紫色长条前面的橙色圆柱", "inspect", "橙色圆柱", "front_of", "紫色长条"),
        ("找到黄色方块后面的蓝色瓶子", "inspect", "蓝色瓶子", "behind", "黄色方块"),
        ("抓起灰色托盘上面的绿色盒子", "pick", "绿色盒子", "above", "灰色托盘"),
        ("拿起蓝色瓶子旁边的红色杯子", "pick", "红色杯子", "near", "蓝色瓶子"),
    ]
    for text, exp_act, exp_tgt, exp_rel, exp_ref in zh_spatial:
        out = parser.parse(text)
        assert out.action == exp_act, text
        assert out.target_query == exp_tgt, text
        assert out.spatial_relation == exp_rel, text
        assert out.reference_query == exp_ref, text

    # Superlatives in Chinese
    s1 = parser.parse("抓起最左边的红色杯子")
    assert s1.action == "pick"
    assert s1.spatial_relation == "leftmost"
    assert s1.target_query == "红色杯子"

    s2 = parser.parse("拿起最右边的杯子")
    assert s2.spatial_relation == "rightmost"
    assert s2.target_query == "杯子"

    s3 = parser.parse("抓起最近的蓝色瓶子")
    assert s3.spatial_relation == "nearest"
    assert s3.target_query == "蓝色瓶子"

    # Pick and place in Chinese
    pap_zh1 = parser.parse("把红色杯子放到灰色托盘上")
    assert pap_zh1.action == "pick_and_place"
    assert pap_zh1.target_query == "红色杯子"
    assert pap_zh1.destination_query == "灰色托盘"

    pap_zh2 = parser.parse("把最左边的绿色盒子放进灰色托盘里")
    assert pap_zh2.action == "pick_and_place"
    assert pap_zh2.target_query == "绿色盒子"
    assert pap_zh2.spatial_relation == "leftmost"
    assert pap_zh2.destination_query == "灰色托盘"


def test_openai_compatible_vlm_parser_offline_fallback_and_structured_json() -> None:
    # 1. Unconfigured -> falls back cleanly without network
    vlm_offline = OpenAICompatibleVLMParser(api_base="", api_key="")
    assert isinstance(vlm_offline, InstructionParserProtocol)
    assert not vlm_offline.is_configured
    res = vlm_offline.parse("把蓝色瓶子放到灰色托盘上")
    assert vlm_offline.last_used_fallback is True
    assert res.action == "pick_and_place"
    assert res.target_query == "蓝色瓶子"
    assert res.destination_query == "灰色托盘"

    # 2. Configured endpoint returning structured JSON -> uses VLM output
    def fake_ok_post(url: str, headers: dict[str, str], payload: dict, timeout_s: float) -> dict:
        assert url.endswith("/chat/completions")
        assert headers["Authorization"] == "Bearer test-key"
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "action": "pick_and_place",
                                "target_query": "cyan t tool",
                                "spatial_relation": "left_of",
                                "reference_query": "blue bottle",
                                "destination_query": "gray tray",
                            }
                        )
                    }
                }
            ]
        }

    vlm_ok = OpenAICompatibleVLMParser(
        api_base="http://localhost:9999/v1",
        api_key="test-key",
        model="local-vlm",
        http_post_fn=fake_ok_post,
    )
    res_ok = vlm_ok.parse("complex natural language request")
    assert vlm_ok.last_used_fallback is False
    assert res_ok.action == "pick_and_place"
    assert res_ok.target_query == "cyan t tool"
    assert res_ok.spatial_relation == "left_of"
    assert res_ok.reference_query == "blue bottle"
    assert res_ok.destination_query == "gray tray"

    # 3. Failing HTTP transport -> gracefully falls back to RuleBasedMultimodalParser
    def fake_fail_post(*args, **kwargs) -> dict:
        raise TimeoutError("simulated timeout")

    vlm_fail = OpenAICompatibleVLMParser(
        api_base="http://localhost:9999/v1",
        api_key="test-key",
        http_post_fn=fake_fail_post,
    )
    res_fb = vlm_fail.parse("pick up the leftmost red mug")
    assert vlm_fail.last_used_fallback is True
    assert res_fb.action == "pick"
    assert res_fb.target_query == "red mug"
    assert res_fb.spatial_relation == "leftmost"


# ---------------------------------------------------------------------------
# 2. SpatialGrounder3D with Real OpenCLIP on RTX 3060
# ---------------------------------------------------------------------------


def test_spatial_grounder_disambiguation_and_open_vocab_clip(
    clip_encoder: OpenCLIPMultimodalEncoder,
) -> None:
    assert normalize_bilingual_query("蓝色瓶子") == "blue bottle"
    assert normalize_bilingual_query("青色T形工具") == "cyan t tool"
    assert normalize_bilingual_query("洋红方块") == "magenta block"

    # Scene with TWO red_mug instances flanking a blue_bottle, plus gray_tray and cyan_t_tool
    # ManiSkill / ROS REP-103: +X forward, +Y left, +Z up
    mug_left = _make_det(1, "red_mug", centroid=(0.00, 0.22, 0.03))
    bottle_mid = _make_det(2, "blue_bottle", centroid=(0.00, 0.00, 0.045))
    mug_right = _make_det(3, "red_mug", centroid=(0.00, -0.22, 0.03))
    tray_back = _make_det(4, "gray_tray", centroid=(0.16, 0.00, 0.01), extent=(0.11, 0.11, 0.02))
    tool_front = _make_det(5, "cyan_t_tool", centroid=(-0.14, 0.06, 0.015))

    sg_builder = SceneGraphBuilder3D()
    sg = sg_builder.build([mug_left, bottle_mid, mug_right, tray_back, tool_front])

    prompts = {name: OBJECT_CATALOG[name].description for name in OBJECT_CATALOG}
    grounder = SpatialGrounder3D(
        clip_encoder=clip_encoder,
        scene_graph_builder=sg_builder,
        class_text_prompts=prompts,
        entropy_nbv_threshold=0.85,
        confidence_accept_threshold=0.60,
    )
    parser = RuleBasedMultimodalParser()

    # 1. Pairwise spatial disambiguation between two same-class red_mugs
    dec_left = grounder.ground(
        parser.parse("pick up the red mug to the left of the blue bottle"), sg
    )
    assert dec_left.found and dec_left.target_object is not None
    assert dec_left.target_object.instance_id == 1
    assert not dec_left.needs_nbv

    dec_right = grounder.ground(
        parser.parse("grab the rightmost red mug"), sg
    )
    assert dec_right.found and dec_right.target_object is not None
    assert dec_right.target_object.instance_id == 3

    # 2. Open-vocabulary synonym matching via real OpenCLIP ("crimson cup" -> red_mug)
    dec_crimson = grounder.ground(
        parser.parse("pick up the crimson cup to the left of the azure bottle"), sg
    )
    assert dec_crimson.found and dec_crimson.target_object is not None
    assert dec_crimson.target_object.instance_id == 1
    assert dec_crimson.target_object.class_name == "red_mug"

    # 3. Chinese open-vocabulary query ("蓝色瓶子" -> blue_bottle, "青色工具" -> cyan_t_tool)
    dec_zh_bottle = grounder.ground(parser.parse("抓起蓝色瓶子"), sg)
    assert dec_zh_bottle.found and dec_zh_bottle.target_object is not None
    assert dec_zh_bottle.target_object.instance_id == 2
    assert dec_zh_bottle.target_object.class_name == "blue_bottle"

    dec_zh_tool = grounder.ground(parser.parse("拿起青色工具"), sg)
    assert dec_zh_tool.found and dec_zh_tool.target_object is not None
    assert dec_zh_tool.target_object.instance_id == 5
    assert dec_zh_tool.target_object.class_name == "cyan_t_tool"

    # 4. Pick-and-place destination resolution
    dec_pap = grounder.ground(parser.parse("把蓝色瓶子左边的红色杯子放到灰色托盘上"), sg)
    assert dec_pap.found and dec_pap.target_object is not None
    assert dec_pap.target_object.instance_id == 1
    assert dec_pap.destination_position is not None
    assert np.allclose(dec_pap.destination_position[:2], tray_back.obb_center[:2], atol=1e-4)
    assert dec_pap.destination_position[2] > tray_back.obb_center[2]

    # 5. High-entropy / low-confidence object triggers needs_nbv=True
    uncertain_tool = _make_det(
        5, "cyan_t_tool", centroid=(-0.14, 0.06, 0.015), confidence=0.42, entropy=0.91
    )
    sg_uncertain = sg_builder.build([mug_left, uncertain_tool])
    dec_nbv = grounder.ground(parser.parse("pick up the cyan t tool"), sg_uncertain)
    assert dec_nbv.found
    assert dec_nbv.needs_nbv is True


# ---------------------------------------------------------------------------
# 3. Multi-View Bayesian Log-Posterior Fusion Unit Verification
# ---------------------------------------------------------------------------


def test_multiview_bayesian_fusion_inverse_entropy_and_density(
    dinov2_extractor: DINOv2MultiScaleExtractor,
    clip_encoder: OpenCLIPMultimodalEncoder,
) -> None:
    cfg = load_config()
    agent = OmniGDAAgent(
        segmenter=SegmentGDAPlus.from_config(cfg, extractor=dinov2_extractor),
        classifier_3d=PartialAwareMultimodalGMDA(config=cfg, clip_encoder=clip_encoder),
        config=cfg,
    )

    # View 1: uncertain (high entropy 0.88) slightly favoring wrong class
    d1 = _make_det(
        instance_id=1,
        class_name="green_box",
        centroid=(0.02, 0.05, 0.025),
        extent=(0.05, 0.05, 0.04),
        confidence=0.52,
        entropy=0.88,
        posterior_probs={"green_box": 0.52, "cyan_t_tool": 0.48},
        num_pts=120,
        seed=1,
    )
    # View 2: sharp (low entropy 0.14) confidently identifying cyan_t_tool on the opposite side
    d2 = _make_det(
        instance_id=7,
        class_name="cyan_t_tool",
        centroid=(0.03, 0.055, 0.025),  # within 0.08m match radius
        extent=(0.05, 0.05, 0.04),
        confidence=0.95,
        entropy=0.14,
        posterior_probs={"green_box": 0.05, "cyan_t_tool": 0.95},
        num_pts=130,
        seed=2,
    )
    # Unmatched newly discovered object in View 2
    d2_new = _make_det(
        instance_id=8,
        class_name="red_mug",
        centroid=(-0.15, -0.18, 0.03),
        confidence=0.91,
        entropy=0.18,
        posterior_probs={"red_mug": 0.91, "green_box": 0.09},
        num_pts=90,
        seed=3,
    )

    fused = agent.fuse_multiview_detections([d1], [d2, d2_new])
    assert len(fused) == 2
    merged = fused[0]
    assert merged.instance_id == 1
    assert merged.class_name == "cyan_t_tool"
    assert merged.confidence > 0.92
    assert merged.entropy < d1.entropy
    assert merged.point_cloud_xyz.shape[0] > d1.point_cloud_xyz.shape[0]
    assert fused[1].class_name == "red_mug"


def test_zero_shot_text_registration_and_grounding(
    dinov2_extractor: DINOv2MultiScaleExtractor,
    clip_encoder: OpenCLIPMultimodalEncoder,
) -> None:
    cfg = load_config()
    agent = OmniGDAAgent(
        segmenter=SegmentGDAPlus.from_config(cfg, extractor=dinov2_extractor),
        classifier_3d=PartialAwareMultimodalGMDA(config=cfg, clip_encoder=clip_encoder),
        config=cfg,
    )
    ms1 = agent.register_object_zero_shot("ceramic_vase", "white decorative ceramic vase")
    ms2 = agent.register_object_zero_shot("metal_wrench", "silver metallic wrench tool")
    assert ms1 > 0.0 and ms2 > 0.0
    assert "ceramic_vase" in agent.classifier_3d.registered_classes
    assert "metal_wrench" in agent.classifier_3d.registered_classes
    assert agent.spatial_grounder.class_text_prompts["ceramic_vase"] == "white decorative ceramic vase"

    sg = agent.scene_graph_builder.build(
        [
            _make_det(1, "ceramic_vase", centroid=(0.0, 0.15, 0.05)),
            _make_det(2, "metal_wrench", centroid=(0.0, -0.15, 0.02)),
        ]
    )
    dec = agent.spatial_grounder.ground(agent.instruction_parser.parse("find the white vase"), sg)
    assert dec.found and dec.target_object is not None
    assert dec.target_object.class_name == "ceramic_vase"


# ---------------------------------------------------------------------------
# 4. REAL CLOSED-LOOP END-TO-END MANISKILL 3 EXECUTION ON RTX 3060
# ---------------------------------------------------------------------------


def test_maniskill_closed_loop_novel_object_spatial_and_nbv(
    dinov2_extractor: DINOv2MultiScaleExtractor,
    clip_encoder: OpenCLIPMultimodalEncoder,
) -> None:
    """End-to-end closed-loop ManiSkill 3 test on RTX 3060 covering:
    1. 2-shot online registration of 5 objects (including held-out novel ``cyan_t_tool``)
       + zero-shot text prototype registration.
    2. Natural-language grasping of held-out novel ``cyan_t_tool`` -> physical lift!
    3. Spatial-relation / bilingual pick-and-place execution!
    4. Active Perception NBV trigger -> switches to ``nbv_camera``, fuses multi-view
       point clouds (``num_views_used == 2``), increases cloud density, and completes grasp!
    """
    cfg = load_config()
    with ClutteredTabletopSim(
        num_objects=5,
        include_novel=True,
        seed=42,
    ) as sim:
        scene_objects = sim.get_object_names()
        assert "cyan_t_tool" in scene_objects, f"Expected novel cyan_t_tool in {scene_objects}"

        # 1. Capture 2-shot isolated templates and register into OmniGDAAgent
        templates = capture_few_shot_templates(scene_objects, num_shots=2, seed=42)

        agent = OmniGDAAgent(
            segmenter=SegmentGDAPlus.from_config(cfg, extractor=dinov2_extractor),
            classifier_3d=PartialAwareMultimodalGMDA(config=cfg, clip_encoder=clip_encoder),
            config=cfg,
        )

        reg_times_ms: dict[str, float] = {}
        for cname in scene_objects:
            ms = agent.register_object_few_shot(
                class_name=cname,
                samples=templates[cname],
                text_prompt=OBJECT_CATALOG[cname].description,
            )
            reg_times_ms[cname] = ms
            assert ms > 0.0
        print(f"\n[Metrics] 2-shot registration ms: {reg_times_ms}")

        controller = FrankaGraspController(sim=sim)

        # ------------------------------------------------------------------
        # Scenario A: Pick up held-out novel object "cyan_t_tool"
        # ------------------------------------------------------------------
        sim.reset(seed=42)
        res_novel = agent.execute_instruction(
            sim=sim,
            controller=controller,
            instruction_text="pick up the cyan t tool",
            allow_nbv=True,
        )
        print(
            f"\n[Scenario A - Novel Pick] success={res_novel.success} "
            f"target={res_novel.target_class_name} views={res_novel.num_views_used} "
            f"percep_ms={res_novel.perception_latency_ms:.1f} "
            f"plan_ms={res_novel.planning_latency_ms:.1f} msg={res_novel.message}"
        )
        assert res_novel.success is True, res_novel.message
        assert res_novel.target_class_name == "cyan_t_tool"
        assert res_novel.grasp_candidate is not None
        assert res_novel.perception_latency_ms > 0.0

        # ------------------------------------------------------------------
        # Scenario B: Spatial superlative / bilingual pick & place execution
        # ------------------------------------------------------------------
        sim.reset(seed=42)
        # Determine actual leftmost object in scene ground-truth for verification
        gt_positions = sim.get_object_positions()
        # ROS / ManiSkill +Y is left
        gt_leftmost_name = max(gt_positions, key=lambda k: float(gt_positions[k][1]))

        res_spatial = agent.execute_instruction(
            sim=sim,
            controller=controller,
            instruction_text="抓起最左边的物体",
            allow_nbv=True,
        )
        print(
            f"\n[Scenario B - ZH Spatial Superlative] gt_leftmost={gt_leftmost_name} "
            f"picked={res_spatial.target_class_name} success={res_spatial.success} "
            f"msg={res_spatial.message}"
        )
        assert res_spatial.success is True, res_spatial.message
        assert res_spatial.target_class_name == gt_leftmost_name

        # ------------------------------------------------------------------
        # Scenario C: Active Perception NBV Trigger (forced high-entropy / low-conf threshold)
        # ------------------------------------------------------------------
        sim.reset(seed=42)
        orig_ent = agent.entropy_nbv_threshold
        orig_conf = agent.confidence_accept_threshold
        try:
            # Force view-1 to trigger NBV active perception
            agent.entropy_nbv_threshold = 0.0
            agent.confidence_accept_threshold = 0.9999

            res_nbv = agent.execute_instruction(
                sim=sim,
                controller=controller,
                instruction_text="pick up the cyan t tool",
                allow_nbv=True,
            )
        finally:
            agent.entropy_nbv_threshold = orig_ent
            agent.confidence_accept_threshold = orig_conf

        assert agent.last_nbv_pre_perception is not None
        assert agent.last_perception is not None
        assert agent.last_grounding is not None and agent.last_grounding.target_object is not None

        fused_target = agent.last_grounding.target_object
        pre_matches = [
            d
            for d in agent.last_nbv_pre_perception.detections
            if np.linalg.norm(d.centroid - fused_target.centroid) < 0.08
        ]
        assert len(pre_matches) >= 1, "Expected matched view-1 detection for cyan_t_tool"
        v1_pts = pre_matches[0].point_cloud_xyz.shape[0]
        fused_pts = fused_target.point_cloud_xyz.shape[0]

        print(
            f"\n[Scenario C - NBV Active Perception] success={res_nbv.success} "
            f"views={res_nbv.num_views_used} v1_pts={v1_pts} -> fused_pts={fused_pts} "
            f"msg={res_nbv.message}"
        )
        assert res_nbv.num_views_used == 2
        assert fused_pts > v1_pts, f"Expected NBV fusion to increase cloud density ({fused_pts} <= {v1_pts})"
        assert res_nbv.success is True, res_nbv.message
        assert res_nbv.target_class_name == "cyan_t_tool"

        # ------------------------------------------------------------------
        # Scenario D: Inspect + Bilingual Closed-Loop Pick-and-Place
        # ------------------------------------------------------------------
        sim.reset(seed=42)
        res_inspect = agent.execute_instruction(
            sim=sim,
            controller=controller,
            instruction_text="找到洋红方块",
            allow_nbv=True,
        )
        print(
            f"\n[Scenario D1 - ZH Inspect] success={res_inspect.success} "
            f"target={res_inspect.target_class_name} msg={res_inspect.message}"
        )
        assert res_inspect.success is True
        assert res_inspect.target_class_name == "magenta_cube"

        res_pap = agent.execute_instruction(
            sim=sim,
            controller=controller,
            instruction_text="put the green box next to the blue bottle",
            allow_nbv=True,
        )
        print(
            f"\n[Scenario D2 - Pick-and-Place] success={res_pap.success} "
            f"target={res_pap.target_class_name} msg={res_pap.message}"
        )
        assert res_pap.success is True, res_pap.message
        assert res_pap.target_class_name == "green_box"
