"""Bilingual (English + Chinese) manipulation instruction parsing with optional VLM adapter.

Provides:
- ``InstructionParserProtocol``: Runtime-checkable protocol for pluggable instruction parsers.
- ``RuleBasedMultimodalParser``: Deterministic EN/ZH parser handling ``pick``,
  ``pick_and_place``, and ``inspect`` intents with pairwise 3D spatial relations
  (``left_of``, ``right_of``, ``front_of``, ``behind``, ``above``, ``near``) and spatial
  superlatives (``leftmost``, ``rightmost``, ``nearest``, ``farthest``, ``highest``).
- ``OpenAICompatibleVLMParser``: Pluggable OpenAI-compatible HTTP VLM/LLM adapter that
  gracefully falls back to ``RuleBasedMultimodalParser`` when offline or unconfigured.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from omnigda.core.schemas import ManipulationInstruction

logger = logging.getLogger(__name__)

VALID_ACTIONS: frozenset[str] = frozenset({"pick", "pick_and_place", "inspect"})

VALID_SPATIAL_RELATIONS: frozenset[str] = frozenset(
    {
        "left_of",
        "right_of",
        "front_of",
        "behind",
        "above",
        "near",
        "leftmost",
        "rightmost",
        "nearest",
        "farthest",
        "highest",
        "frontmost",
        "backmost",
    }
)


@runtime_checkable
class InstructionParserProtocol(Protocol):
    """Protocol implemented by all OmniGDA manipulation instruction parsers."""

    def parse(self, text: str) -> ManipulationInstruction:
        """Parse a natural-language command into a structured ``ManipulationInstruction``."""
        ...


# ---------------------------------------------------------------------------
# Lexical patterns for RuleBasedMultimodalParser (English & Chinese)
# ---------------------------------------------------------------------------

_EN_POLITE_PREFIX_RE = re.compile(
    r"^(?:please|kindly|robot|hey\s+robot|can\s+you|could\s+you|would\s+you|"
    r"i\s+want\s+you\s+to|go\s+ahead\s+and)\b[\s,]*",
    flags=re.IGNORECASE,
)

_ZH_POLITE_PREFIX_RE = re.compile(
    r"^(?:请|麻烦|帮我|帮忙|机器人|请帮我|能不能|可以|帮我一下)\s*"
)

_EN_LEADING_ARTICLES_RE = re.compile(
    r"^(?:the|a|an|this|that|these|those|one|some|my|target)\b\s*",
    flags=re.IGNORECASE,
)

_EN_TRAILING_FILLER_RE = re.compile(
    r"\s*\b(?:please|now|thanks|thank\s+you)\b\s*$",
    flags=re.IGNORECASE,
)

_ZH_FILLER_RE = re.compile(r"^(?:一下|那个|这个|一个|那只|这只|那块|这块|那瓶|这瓶|那杯|这杯)+")

# Pick-and-place patterns (checked BEFORE plain pick)
_EN_PICK_AND_PUT_RE = re.compile(
    r"^(?:pick\s+up|pick|grab|grasp|take|lift(?:\s+up)?)\s+(.+?)\s+"
    r"(?:and\s+)?(?:put|place|move|set|drop|stack)\s+(?:it|them|that|the\s+object)?\s*"
    r"(on\s+top\s+of|onto|into|inside\s+of|inside|in\s+front\s+of|to\s+the\s+left\s+of|"
    r"to\s+the\s+right\s+of|next\s+to|close\s+to|beside|near|behind|above|over|on|in|to)\s+(.+)$",
    flags=re.IGNORECASE,
)

_EN_PLACE_VERB_RE = re.compile(
    r"^(?:put|place|move|set|drop|stack|transfer)\s+(.+)$",
    flags=re.IGNORECASE,
)

# Strong placement prepositions take priority over spatial prepositions inside target clauses
_EN_STRONG_PLACE_PREPS: tuple[str, ...] = (
    "on top of",
    "onto",
    "into",
    "inside of",
    "inside",
    "on",
    "in",
)

_EN_SPATIAL_PLACE_PREPS: tuple[str, ...] = (
    "to the left of",
    "to the right of",
    "in front of",
    "next to",
    "close to",
    "beside",
    "near",
    "behind",
    "above",
    "over",
)

_EN_FALLBACK_PLACE_PREPS: tuple[str, ...] = ("to",)

_ZH_PICK_AND_PLACE_RE = re.compile(
    r"^(?:把|将|拿起|抓起|拿取|抓取|夹起|拿|抓)\s*(.+?)\s*"
    r"(?:然后|并|再)?(?:放到|放在|放进|放入|移到|移动到|摆到|摆在|搁到|搁在|置于|置入|放到)\s*(.+)$"
)

_ZH_DEST_LOCATIVE_SUFFIX_RE = re.compile(
    r"(?:的)?(?:上面|上方|顶上|上边|上|里面|里边|里|内部|中|旁边|附近|边上|左侧|左边|右侧|右边|前面|后面)$"
)

# Inspect patterns
_EN_INSPECT_RE = re.compile(
    r"^(?:find|locate|search(?:\s+for)?|inspect|check(?:\s+for)?|look\s+for|identify|"
    r"where\s+is|show(?:\s+me)?|detect)\b\s*(.*)$",
    flags=re.IGNORECASE,
)

_ZH_INSPECT_RE = re.compile(
    r"^(?:找到|寻找|查找|定位|检查|查看|看看|找一下|识别|检测|找)\s*(.*)$"
)

_ZH_WHERE_RE = re.compile(r"^(.+?)(?:在哪里|在哪儿|在哪|在什么位置)$")

# Pick patterns
_EN_PICK_RE = re.compile(
    r"^(?:pick\s+up|pick|grab|grasp|lift(?:\s+up)?|take|get|fetch|hold|retrieve)\b\s*(.*)$",
    flags=re.IGNORECASE,
)

_ZH_PICK_RE = re.compile(
    r"^(?:抓起|抓取|拿起|拿取|夹起|夹取|拾起|提起|抓一下|拿一下|抓|拿|夹)\s*(.*)$"
)

# Superlative patterns (checked BEFORE pairwise patterns so "最左边" isn't split by "左边")
_SUPERLATIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "leftmost",
        re.compile(
            r"(?:\b(?:the\s+)?(?:leftmost|left-most|far(?:thest)?\s+left|furthest\s+left|"
            r"on\s+the\s+far\s+left)\b|最左边(?:的)?|最左侧(?:的)?|最左面(?:的)?|最左(?:的)?)",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "rightmost",
        re.compile(
            r"(?:\b(?:the\s+)?(?:rightmost|right-most|far(?:thest)?\s+right|furthest\s+right|"
            r"on\s+the\s+far\s+right)\b|最右边(?:的)?|最右侧(?:的)?|最右面(?:的)?|最右(?:的)?)",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "nearest",
        re.compile(
            r"(?:\b(?:the\s+)?(?:nearest|closest)\b|离我最近(?:的)?|最近(?:的)?)",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "farthest",
        re.compile(
            r"(?:\b(?:the\s+)?(?:farthest|furthest)\b|离我最远(?:的)?|最远(?:的)?)",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "highest",
        re.compile(
            r"(?:\b(?:the\s+)?(?:highest|topmost|tallest)\b|最高(?:的)?|最上面(?:的)?)",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "frontmost",
        re.compile(
            r"(?:\b(?:the\s+)?(?:frontmost|front-most)\b|最前面(?:的)?|最前方(?:的)?)",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "backmost",
        re.compile(
            r"(?:\b(?:the\s+)?(?:backmost|back-most|rearmost)\b|最后面(?:的)?|最后方(?:的)?)",
            flags=re.IGNORECASE,
        ),
    ),
)

# Pairwise spatial patterns (English)
_EN_PAIRWISE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "left_of",
        re.compile(
            r"^(.+?)\s+(?:(?:that|which)\s+is\s+)?(?:located\s+)?"
            r"(?:to\s+the\s+left\s+of|on\s+the\s+left(?:\s+side)?\s+of|left\s+of)\s+(.+)$",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "right_of",
        re.compile(
            r"^(.+?)\s+(?:(?:that|which)\s+is\s+)?(?:located\s+)?"
            r"(?:to\s+the\s+right\s+of|on\s+the\s+right(?:\s+side)?\s+of|right\s+of)\s+(.+)$",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "front_of",
        re.compile(
            r"^(.+?)\s+(?:(?:that|which)\s+is\s+)?(?:located\s+)?"
            r"(?:in\s+front\s+of|front\s+of|ahead\s+of)\s+(.+)$",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "behind",
        re.compile(
            r"^(.+?)\s+(?:(?:that|which)\s+is\s+)?(?:located\s+)?"
            r"(?:behind|in\s+back\s+of|at\s+the\s+back\s+of)\s+(.+)$",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "above",
        re.compile(
            r"^(.+?)\s+(?:(?:that|which)\s+is\s+)?(?:located\s+)?"
            r"(?:on\s+top\s+of|above|over)\s+(.+)$",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "near",
        re.compile(
            r"^(.+?)\s+(?:(?:that|which)\s+is\s+)?(?:located\s+)?"
            r"(?:next\s+to|near(?:\s+to)?|beside|close\s+to|adjacent\s+to)\s+(.+)$",
            flags=re.IGNORECASE,
        ),
    ),
)

_ZH_REL_TOKEN_MAP: dict[str, str] = {
    "左边": "left_of",
    "左侧": "left_of",
    "左面": "left_of",
    "右边": "right_of",
    "右侧": "right_of",
    "右面": "right_of",
    "前面": "front_of",
    "前方": "front_of",
    "前边": "front_of",
    "后面": "behind",
    "后方": "behind",
    "后边": "behind",
    "上面": "above",
    "上方": "above",
    "顶上": "above",
    "旁边": "near",
    "附近": "near",
    "边上": "near",
    "靠着": "near",
    "挨着": "near",
}

_ZH_REL_ALT = "|".join(sorted(_ZH_REL_TOKEN_MAP.keys(), key=len, reverse=True))

# Chinese modifier order: "[在] <ref> [的] <rel> 的 <target>"
_ZH_PAIRWISE_MODIFIER_RE = re.compile(
    rf"^(?:在|位于)?\s*(.+?)(?:的)?({_ZH_REL_ALT})(?:的|那个|这个)\s*(.+)$"
)

# Chinese predicative order: "<target> 在 <ref> [的] <rel>"
_ZH_PAIRWISE_PREDICATE_RE = re.compile(
    rf"^(.+?)\s*(?:在|位于)\s*(.+?)(?:的)?({_ZH_REL_ALT})$"
)

# Nearest/farthest with explicit reference: "nearest <target> to <ref>"
_EN_NEAREST_REF_RE = re.compile(r"^(.+?)\s+(?:to|from)\s+(.+)$", flags=re.IGNORECASE)


def _strip_punctuation(text: str) -> str:
    return text.strip(" \t\r\n.,!?;:。，！？；：\"'“”‘’()（）")


def _clean_phrase(phrase: str) -> str:
    """Clean articles, polite fillers, and redundant particles from an entity phrase."""
    s = _strip_punctuation(phrase)
    # Iteratively remove leading English articles / Chinese filler particles
    for _ in range(3):
        prev = s
        s = _EN_LEADING_ARTICLES_RE.sub("", s).strip()
        s = _EN_TRAILING_FILLER_RE.sub("", s).strip()
        s = _ZH_FILLER_RE.sub("", s).strip()
        s = s.strip("的").strip()
        if s == prev:
            break
    # Collapse internal whitespace and clean Chinese "色 的" -> "色"
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"([\u4e00-\u9fff])的([\u4e00-\u9fff])", r"\1\2", s)
    return s


def _split_en_place_clause(body: str) -> tuple[str, str] | None:
    """Split `<target_clause> <prep> <dest_clause>` using the rightmost valid preposition."""
    for prep_group in (_EN_STRONG_PLACE_PREPS, _EN_SPATIAL_PLACE_PREPS, _EN_FALLBACK_PLACE_PREPS):
        best_pos = -1
        best_prep = ""
        for prep in prep_group:
            pattern = re.compile(rf"\s+{re.escape(prep)}\s+", flags=re.IGNORECASE)
            matches = list(pattern.finditer(body))
            if matches:
                m = matches[-1]
                if m.start() > best_pos:
                    best_pos = m.start()
                    best_prep = m.group(0)
        if best_pos > 0:
            left = body[:best_pos].strip()
            right = body[best_pos + len(best_prep) :].strip()
            if left and right:
                return left, right
    return None


def _extract_spatial_from_clause(
    clause: str,
) -> tuple[str, str | None, str | None]:
    """Extract ``(target_query, spatial_relation, reference_query)`` from a noun phrase."""
    text = _strip_punctuation(clause)
    if not text:
        return "", None, None

    # 1. Check superlatives first so "最左边" is never hijacked by "左边"
    for rel_name, pattern in _SUPERLATIVE_PATTERNS:
        m = pattern.search(text)
        if m is not None:
            remainder = (text[: m.start()] + " " + text[m.end() :]).strip()
            remainder = re.sub(r"\s+", " ", remainder)
            # Support "nearest mug to the bottle"
            if rel_name in {"nearest", "farthest"}:
                m_ref = _EN_NEAREST_REF_RE.match(remainder)
                if m_ref is not None:
                    tgt = _clean_phrase(m_ref.group(1))
                    ref = _clean_phrase(m_ref.group(2))
                    if tgt and ref:
                        return tgt, rel_name, ref
            tgt = _clean_phrase(remainder)
            return tgt, rel_name, None

    # 2. Check Chinese pairwise spatial clauses
    m_zh_mod = _ZH_PAIRWISE_MODIFIER_RE.match(text)
    if m_zh_mod is not None:
        ref = _clean_phrase(m_zh_mod.group(1))
        rel = _ZH_REL_TOKEN_MAP[m_zh_mod.group(2)]
        tgt = _clean_phrase(m_zh_mod.group(3))
        if tgt and ref:
            return tgt, rel, ref

    m_zh_pred = _ZH_PAIRWISE_PREDICATE_RE.match(text)
    if m_zh_pred is not None:
        tgt = _clean_phrase(m_zh_pred.group(1))
        ref = _clean_phrase(m_zh_pred.group(2))
        rel = _ZH_REL_TOKEN_MAP[m_zh_pred.group(3)]
        if tgt and ref:
            return tgt, rel, ref

    # 3. Check English pairwise spatial clauses
    for rel_name, pattern in _EN_PAIRWISE_PATTERNS:
        m_en = pattern.match(text)
        if m_en is not None:
            tgt = _clean_phrase(m_en.group(1))
            ref = _clean_phrase(m_en.group(2))
            if tgt and ref:
                return tgt, rel_name, ref

    return _clean_phrase(text), None, None


class RuleBasedMultimodalParser:
    """Fast, deterministic English & Chinese manipulation instruction parser."""

    def parse(self, text: str) -> ManipulationInstruction:
        """Parse an English or Chinese manipulation command into ``ManipulationInstruction``."""
        raw = str(text).strip()
        if not raw:
            return ManipulationInstruction(raw_text="", action="pick", target_query="")

        cleaned = _strip_punctuation(raw)
        # Remove polite prefixes iteratively (supports mixed or repeated prefixes)
        for _ in range(3):
            prev = cleaned
            cleaned = _EN_POLITE_PREFIX_RE.sub("", cleaned).strip()
            cleaned = _ZH_POLITE_PREFIX_RE.sub("", cleaned).strip()
            cleaned = _EN_TRAILING_FILLER_RE.sub("", cleaned).strip()
            if cleaned == prev:
                break

        # 1. Check pick_and_place intents first (before plain pick)
        m_en_pap = _EN_PICK_AND_PUT_RE.match(cleaned)
        if m_en_pap is not None:
            target_clause = m_en_pap.group(1).strip()
            dest_clause = _clean_phrase(m_en_pap.group(3))
            tgt, rel, ref = _extract_spatial_from_clause(target_clause)
            return ManipulationInstruction(
                raw_text=raw,
                action="pick_and_place",
                target_query=tgt,
                spatial_relation=rel,
                reference_query=ref,
                destination_query=dest_clause or None,
            )

        m_zh_pap = _ZH_PICK_AND_PLACE_RE.match(cleaned)
        if m_zh_pap is not None:
            target_clause = m_zh_pap.group(1).strip()
            raw_dest = _ZH_DEST_LOCATIVE_SUFFIX_RE.sub("", m_zh_pap.group(2).strip())
            dest_clause = _clean_phrase(raw_dest)
            tgt, rel, ref = _extract_spatial_from_clause(target_clause)
            return ManipulationInstruction(
                raw_text=raw,
                action="pick_and_place",
                target_query=tgt,
                spatial_relation=rel,
                reference_query=ref,
                destination_query=dest_clause or None,
            )

        m_en_place = _EN_PLACE_VERB_RE.match(cleaned)
        if m_en_place is not None:
            split = _split_en_place_clause(m_en_place.group(1).strip())
            if split is not None:
                target_clause, dest_raw = split
                tgt, rel, ref = _extract_spatial_from_clause(target_clause)
                dest_clause = _clean_phrase(dest_raw)
                return ManipulationInstruction(
                    raw_text=raw,
                    action="pick_and_place",
                    target_query=tgt,
                    spatial_relation=rel,
                    reference_query=ref,
                    destination_query=dest_clause or None,
                )

        # 2. Check inspect intents
        m_en_insp = _EN_INSPECT_RE.match(cleaned)
        if m_en_insp is not None:
            tgt, rel, ref = _extract_spatial_from_clause(m_en_insp.group(1))
            return ManipulationInstruction(
                raw_text=raw,
                action="inspect",
                target_query=tgt,
                spatial_relation=rel,
                reference_query=ref,
                destination_query=None,
            )

        m_zh_insp = _ZH_INSPECT_RE.match(cleaned)
        if m_zh_insp is not None:
            tgt, rel, ref = _extract_spatial_from_clause(m_zh_insp.group(1))
            return ManipulationInstruction(
                raw_text=raw,
                action="inspect",
                target_query=tgt,
                spatial_relation=rel,
                reference_query=ref,
                destination_query=None,
            )

        m_zh_where = _ZH_WHERE_RE.match(cleaned)
        if m_zh_where is not None:
            tgt, rel, ref = _extract_spatial_from_clause(m_zh_where.group(1))
            return ManipulationInstruction(
                raw_text=raw,
                action="inspect",
                target_query=tgt,
                spatial_relation=rel,
                reference_query=ref,
                destination_query=None,
            )

        # 3. Check pick intents
        m_en_pick = _EN_PICK_RE.match(cleaned)
        if m_en_pick is not None:
            tgt, rel, ref = _extract_spatial_from_clause(m_en_pick.group(1))
            return ManipulationInstruction(
                raw_text=raw,
                action="pick",
                target_query=tgt,
                spatial_relation=rel,
                reference_query=ref,
                destination_query=None,
            )

        m_zh_pick = _ZH_PICK_RE.match(cleaned)
        if m_zh_pick is not None:
            tgt, rel, ref = _extract_spatial_from_clause(m_zh_pick.group(1))
            return ManipulationInstruction(
                raw_text=raw,
                action="pick",
                target_query=tgt,
                spatial_relation=rel,
                reference_query=ref,
                destination_query=None,
            )

        # 4. Bare noun/spatial phrase defaults to "pick"
        tgt, rel, ref = _extract_spatial_from_clause(cleaned)
        return ManipulationInstruction(
            raw_text=raw,
            action="pick",
            target_query=tgt,
            spatial_relation=rel,
            reference_query=ref,
            destination_query=None,
        )


_VLM_SYSTEM_PROMPT = """You are a robotic manipulation instruction parser.
Convert the user's English or Chinese manipulation command into a single JSON object with keys:
- "action": one of "pick", "pick_and_place", "inspect"
- "target_query": concise object description (string)
- "spatial_relation": one of null, "left_of", "right_of", "front_of", "behind", "above", "near", "leftmost", "rightmost", "nearest", "farthest", "highest"
- "reference_query": reference object description if spatial_relation is pairwise, else null
- "destination_query": destination object/surface description if action is "pick_and_place", else null
Return ONLY valid JSON without markdown formatting."""


class OpenAICompatibleVLMParser:
    """Pluggable OpenAI-compatible HTTP VLM/LLM instruction parser with offline fallback.

    When ``api_base`` is unconfigured or the remote endpoint is unreachable/invalid,
    automatically falls back to ``RuleBasedMultimodalParser`` without raising.
    """

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 8.0,
        fallback_parser: InstructionParserProtocol | None = None,
        http_post_fn: Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]
        | None = None,
    ) -> None:
        self.api_base = (
            api_base if api_base is not None else os.environ.get("OMNIGDA_VLM_API_BASE", "")
        ).strip().rstrip("/")
        self.api_key = (
            api_key if api_key is not None else os.environ.get("OMNIGDA_VLM_API_KEY", "")
        ).strip()
        self.model = (
            model if model is not None else os.environ.get("OMNIGDA_VLM_MODEL", "gpt-4o-mini")
        ).strip()
        self.timeout_s = float(timeout_s)
        self.fallback_parser: InstructionParserProtocol = (
            fallback_parser if fallback_parser is not None else RuleBasedMultimodalParser()
        )
        self._http_post_fn = http_post_fn
        self.last_used_fallback: bool = False
        self.last_error: str | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_base)

    @staticmethod
    def _default_http_post(
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw_bytes = resp.read()
        return json.loads(raw_bytes.decode("utf-8"))

    @staticmethod
    def _extract_json_object(content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in VLM response")
        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Parsed VLM JSON is not an object")
        return parsed

    def parse(self, text: str) -> ManipulationInstruction:
        """Parse instruction via OpenAI-compatible API, falling back cleanly when offline."""
        self.last_error = None
        if not self.is_configured:
            self.last_used_fallback = True
            return self.fallback_parser.parse(text)

        endpoint = (
            self.api_base
            if self.api_base.endswith("/chat/completions")
            else f"{self.api_base}/chat/completions"
        )
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": _VLM_SYSTEM_PROMPT},
                {"role": "user", "content": str(text)},
            ],
        }

        post_fn = self._http_post_fn or self._default_http_post
        try:
            resp = post_fn(endpoint, headers, payload, self.timeout_s)
            choices = resp.get("choices") or []
            if not choices:
                raise ValueError("VLM response missing choices")
            message = choices[0].get("message") or {}
            content = str(message.get("content", ""))
            data = self._extract_json_object(content)

            action = str(data.get("action", "pick")).strip().lower()
            if action not in VALID_ACTIONS:
                action = "pick"

            raw_rel = data.get("spatial_relation")
            spatial_relation: str | None = None
            if raw_rel is not None and str(raw_rel).strip().lower() not in {"", "none", "null"}:
                norm_rel = str(raw_rel).strip().lower()
                if norm_rel in VALID_SPATIAL_RELATIONS:
                    spatial_relation = norm_rel

            def _opt_str(val: Any) -> str | None:
                if val is None:
                    return None
                s = str(val).strip()
                return s if s and s.lower() not in {"none", "null"} else None

            target_query = str(data.get("target_query") or "").strip()
            reference_query = _opt_str(data.get("reference_query"))
            destination_query = _opt_str(data.get("destination_query"))

            self.last_used_fallback = False
            return ManipulationInstruction(
                raw_text=str(text).strip(),
                action=action,
                target_query=target_query,
                spatial_relation=spatial_relation,
                reference_query=reference_query,
                destination_query=destination_query,
            )
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            self.last_used_fallback = True
            self.last_error = str(exc)
            logger.debug("VLM parser unavailable (%s); using RuleBasedMultimodalParser.", exc)
            return self.fallback_parser.parse(text)
