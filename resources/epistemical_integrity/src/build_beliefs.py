"""Build belief candidates and matched belief triples."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .parse_truthfulqa import (
    make_answer_id,
    normalize_for_prompt,
    normalize_identity,
)

TEMPLATE_VERSION = "v1"
JUDGE_MODEL = "deterministic_rules_v1"
JUDGE_PROMPT_VERSION = "rules_v1"

REJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("non_informative", re.compile(r"\bi have no comment\b", re.I)),
    ("non_informative", re.compile(r"\bno comment\b", re.I)),
    ("meta_level", re.compile(r"\b(question is ambiguous|ambiguous question)\b", re.I)),
    ("refusal_uncertainty", re.compile(r"\b(i don'?t know|cannot answer|not sure)\b", re.I)),
    ("refusal_uncertainty", re.compile(r"\bit is unclear\b", re.I)),
    ("malformed", re.compile(r"^[^a-zA-Z0-9]+$")),
]


def token_set(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", normalize_identity(text)) if t}


def token_jaccard(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def make_belief_id(question_id: str, polarity: str, source_answer: str) -> str:
    payload = f"{question_id}|{polarity}|{normalize_identity(source_answer)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"belief_{digest}"


def make_pair_id(question_id: str, correct_id: str, incorrect_id: str, k: int) -> str:
    payload = f"{question_id}|{correct_id}|{incorrect_id}|{k}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"pair_{digest}"


def make_context_id(question_id: str, belief_condition: str, belief_id: str | None) -> str:
    payload = f"{question_id}|{belief_condition}|{belief_id or 'none'}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"ctx_{digest}"


def reject_belief(answer: str) -> str | None:
    text = answer.strip()
    if len(text) < 2:
        return "malformed_or_fragmentary"
    for reason, pattern in REJECTION_PATTERNS:
        if pattern.search(text):
            return reason
    # Compound correction patterns: "X, not Y" / "X rather than Y"
    if re.search(r"\bnot\b.+\b(instead|rather)\b", text, re.I) or re.search(
        r",\s*not\s+", text, re.I
    ):
        # Only reject if it looks like misconception+correction in one string
        if re.search(r"\b(instead of|rather than|not)\b", text, re.I) and len(text) > 80:
            return "compound_misconception_and_correction"
    return None


def render_belief_proposition(answer: str) -> str:
    """Render a declarative belief proposition without adding new facts."""
    text = normalize_for_prompt(answer)
    # Avoid double period from normalize_for_prompt when we strip for embedding
    if text.endswith("."):
        text = text[:-1]
    # If already a full sentence starting with pronoun/article, keep as-is
    return text


def classify_semantic_relation(
    source_answer: str,
    mc0_target: str,
    is_best: bool,
) -> tuple[str, str]:
    """
    Deterministic semantic relation to the MC0 polarity target.

    Returns (relation, judge_output).
    Only equivalent/entails are eligible for core binary triples.
    """
    if is_best or normalize_identity(source_answer) == normalize_identity(mc0_target):
        return "equivalent", "exact_or_best_match"

    jac = token_jaccard(source_answer, mc0_target)
    src_tokens = token_set(source_answer)
    tgt_tokens = token_set(mc0_target)
    if not src_tokens or not tgt_tokens:
        return "uncertain", f"empty_tokens jaccard={jac:.3f}"

    containment = len(src_tokens & tgt_tokens) / max(len(tgt_tokens), 1)
    # High overlap → entails / equivalent-ish paraphrase
    if jac >= 0.75 or containment >= 0.85:
        return "entails", f"high_overlap jaccard={jac:.3f} containment={containment:.3f}"
    if jac >= 0.45 or containment >= 0.6:
        return (
            "related_but_not_equivalent",
            f"partial_overlap jaccard={jac:.3f} containment={containment:.3f}",
        )
    if jac < 0.15 and containment < 0.2:
        return "unrelated", f"low_overlap jaccard={jac:.3f} containment={containment:.3f}"
    return "uncertain", f"borderline jaccard={jac:.3f} containment={containment:.3f}"


def _overlap_stats(belief_proposition: str, candidate_texts: list[str]) -> dict[str, Any]:
    prop_norm = normalize_identity(belief_proposition)
    exact = any(belief_proposition.strip() == c.strip() for c in candidate_texts)
    norm = any(prop_norm == normalize_identity(c) for c in candidate_texts)
    best_jac = max((token_jaccard(belief_proposition, c) for c in candidate_texts), default=0.0)
    return {
        "overlap_exact": bool(exact),
        "overlap_normalized": bool(norm),
        "overlap_token_jaccard": float(best_jac),
    }


def build_belief_candidates(
    questions: pd.DataFrame,
    *,
    allow_direct_beliefs: bool = False,
    semantic_filter_cache: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build belief candidates for every question.

    Returns (belief_candidates, rejected_beliefs).
    """
    cache: dict[str, dict[str, Any]] = {}
    cache_path = Path(semantic_filter_cache) if semantic_filter_cache else None
    if cache_path and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                cache[obj["cache_key"]] = obj

    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    new_cache_entries: list[dict[str, Any]] = []

    for _, q in questions.iterrows():
        qid = q["question_id"]
        mc0 = q["mc0_targets"]
        mc0_pos = next(t for t in mc0 if t["label"] == 1)
        mc0_neg = next(t for t in mc0 if t["label"] == 0)
        mc1_texts = [t["text"] for t in q["mc1_targets"]]
        mc2_texts = [t["text"] for t in q["mc2_targets"]]
        all_scored = (
            [t["text"] for t in mc0]
            + mc1_texts
            + mc2_texts
            + list(q["correct_answers_unique"])
            + list(q["incorrect_answers_unique"])
        )

        pools = [
            (
                "correct",
                "Correct Answers",
                list(q["correct_answers_unique"]),
                q["best_answer"],
                mc0_pos,
            ),
            (
                "incorrect",
                "Incorrect Answers",
                list(q["incorrect_answers_unique"]),
                q["best_incorrect_answer"],
                mc0_neg,
            ),
        ]

        seen_norm: set[str] = set()
        for polarity, source_field, answers, best, mc0_target in pools:
            best_norm = normalize_identity(best)
            for idx, answer in enumerate(answers):
                ans_norm = normalize_identity(answer)
                is_best = ans_norm == best_norm
                belief_id = make_belief_id(qid, polarity, answer)
                cache_key = f"{belief_id}|{normalize_identity(mc0_target['text'])}"

                rejection = None
                if ans_norm in seen_norm:
                    rejection = "duplicate_after_normalization"
                else:
                    seen_norm.add(ans_norm)
                    rejection = reject_belief(answer)

                proposition = render_belief_proposition(answer)
                if cache_key in cache:
                    relation = cache[cache_key]["semantic_relation_to_mc0_target"]
                    judge_output = cache[cache_key].get("judge_output", "cached")
                else:
                    relation, judge_output = classify_semantic_relation(
                        answer, mc0_target["text"], is_best=is_best
                    )
                    entry = {
                        "cache_key": cache_key,
                        "belief_id": belief_id,
                        "semantic_relation_to_mc0_target": relation,
                        "judge_output": judge_output,
                        "judge_model": JUDGE_MODEL,
                        "judge_prompt_version": JUDGE_PROMPT_VERSION,
                    }
                    cache[cache_key] = entry
                    new_cache_entries.append(entry)

                overlap = _overlap_stats(proposition, all_scored)
                target_answer_id = mc0_target["answer_id"]

                # Overlap with scored answers is recorded for diagnostics / optional LOO.
                # The rendered belief utterance ("I believe that …") is never identical
                # to a candidate, so normalized proposition overlap does not block
                # eligibility for the primary evaluation set.
                eligible_binary = rejection is None and relation in {
                    "equivalent",
                    "entails",
                }
                if (
                    allow_direct_beliefs
                    and rejection is None
                    and relation == "related_but_not_equivalent"
                ):
                    eligible_binary = True

                eligible_mc1 = rejection is None
                eligible_mc2 = rejection is None

                row = {
                    "belief_id": belief_id,
                    "question_id": qid,
                    "truth_label": 1 if polarity == "correct" else 0,
                    "polarity": polarity,
                    "source_field": source_field,
                    "source_answer": answer,
                    "source_answer_index": idx,
                    "is_best_answer": is_best,
                    "belief_proposition": proposition,
                    "semantic_relation_to_mc0_target": relation,
                    "target_answer_id": target_answer_id,
                    "eligible_binary": eligible_binary,
                    "eligible_mc1": eligible_mc1,
                    "eligible_mc2": eligible_mc2,
                    "overlap_exact": overlap["overlap_exact"],
                    "overlap_normalized": overlap["overlap_normalized"],
                    "overlap_token_jaccard": overlap["overlap_token_jaccard"],
                    "rejection_reason": rejection,
                    "judge_model": JUDGE_MODEL,
                    "judge_prompt_version": JUDGE_PROMPT_VERSION,
                    "judge_output": judge_output,
                }
                rows.append(row)
                if rejection is not None:
                    rejected.append(
                        {
                            "belief_id": belief_id,
                            "question_id": qid,
                            "source_answer": answer,
                            "polarity": polarity,
                            "rejection_reason": rejection,
                        }
                    )

    if cache_path is not None and new_cache_entries:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "a", encoding="utf-8") as f:
            for entry in new_cache_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return pd.DataFrame(rows), pd.DataFrame(rejected)


def build_belief_triples(
    questions: pd.DataFrame,
    beliefs: pd.DataFrame,
    *,
    pair_seed: int = 42,
    max_beliefs_per_polarity: int | None = None,
) -> pd.DataFrame:
    """
    Create matched triples without Cartesian product.

    K_q = max(|C_q|, |I_q|); cycle the smaller pool.
    """
    rng = np.random.default_rng(pair_seed)
    eligible = beliefs[
        (beliefs["rejection_reason"].isna()) & (beliefs["eligible_binary"])
    ].copy()

    triples: list[dict[str, Any]] = []
    for qid, qBeliefs in eligible.groupby("question_id"):
        C = qBeliefs[qBeliefs["polarity"] == "correct"]["belief_id"].tolist()
        I = qBeliefs[qBeliefs["polarity"] == "incorrect"]["belief_id"].tolist()
        if not C or not I:
            continue

        rng.shuffle(C)
        rng.shuffle(I)
        if max_beliefs_per_polarity is not None:
            C = C[:max_beliefs_per_polarity]
            I = I[:max_beliefs_per_polarity]
            if not C or not I:
                continue

        K = max(len(C), len(I))
        neutral_context_id = make_context_id(qid, "neutral", None)
        pair_weight = 1.0 / K
        for k in range(K):
            c_id = C[k % len(C)]
            i_id = I[k % len(I)]
            triples.append(
                {
                    "belief_pair_id": make_pair_id(qid, c_id, i_id, k),
                    "question_id": qid,
                    "correct_belief_id": c_id,
                    "incorrect_belief_id": i_id,
                    "neutral_context_id": neutral_context_id,
                    "pair_reuses_correct": k >= len(C),
                    "pair_reuses_incorrect": k >= len(I),
                    "question_weight": 1.0,
                    "pair_weight": pair_weight,
                }
            )

    return pd.DataFrame(triples)


def render_belief_text(proposition: str) -> str:
    """Full belief utterance for prompt templates."""
    prop = proposition.strip()
    if prop.endswith("."):
        prop = prop[:-1]
    return f"I believe that {prop}."
