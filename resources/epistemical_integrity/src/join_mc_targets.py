"""Join official TruthfulQA multiple-choice targets onto parsed questions."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .parse_truthfulqa import make_answer_id, normalize_identity


def load_mc_task(mc_json_path: str) -> list[dict[str, Any]]:
    with open(mc_json_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("mc_task.json must be a list of records")
    return data


def _targets_to_list(targets: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for text, label in targets.items():
        out.append({"text": str(text), "label": int(label)})
    return out


def validate_mc_targets(
    question: str,
    mc0: dict[str, Any],
    mc1: dict[str, Any],
    mc2: dict[str, Any],
    best_answer: str,
    best_incorrect: str,
) -> dict[str, Any]:
    """Validate MC label structure and best-answer alignment. Fail loudly."""
    mc0_items = _targets_to_list(mc0)
    mc1_items = _targets_to_list(mc1)
    mc2_items = _targets_to_list(mc2)

    pos0 = [t for t in mc0_items if t["label"] == 1]
    neg0 = [t for t in mc0_items if t["label"] == 0]
    if len(pos0) != 1 or len(neg0) != 1:
        raise AssertionError(
            f"MC0 must have exactly one positive and one negative for: {question!r}"
        )

    pos1 = [t for t in mc1_items if t["label"] == 1]
    neg1 = [t for t in mc1_items if t["label"] == 0]
    if len(pos1) != 1:
        raise AssertionError(f"MC1 must have exactly one positive for: {question!r}")
    if len(neg1) < 1:
        raise AssertionError(f"MC1 must have >=1 negative for: {question!r}")

    pos2 = [t for t in mc2_items if t["label"] == 1]
    neg2 = [t for t in mc2_items if t["label"] == 0]
    if len(pos2) < 1 or len(neg2) < 1:
        raise AssertionError(f"MC2 must have both labels for: {question!r}")

    if normalize_identity(pos0[0]["text"]) != normalize_identity(best_answer):
        raise AssertionError(
            f"Best Answer != MC0 positive after normalization for: {question!r}\n"
            f"  best={best_answer!r}\n  mc0+={pos0[0]['text']!r}"
        )
    if normalize_identity(neg0[0]["text"]) != normalize_identity(best_incorrect):
        raise AssertionError(
            f"Best Incorrect Answer != MC0 negative after normalization for: {question!r}\n"
            f"  best_inc={best_incorrect!r}\n  mc0-={neg0[0]['text']!r}"
        )

    return {
        "mc0_targets": mc0_items,
        "mc1_targets": mc1_items,
        "mc2_targets": mc2_items,
        "n_mc1_false": len(neg1),
        "n_mc2_true": len(pos2),
        "n_mc2_false": len(neg2),
        "mc0_eligible": True,
        "mc1_eligible": True,
        "mc2_eligible": True,
    }


def join_mc_targets(questions: pd.DataFrame, mc_json_path: str) -> pd.DataFrame:
    """Join official MC targets by normalized question text (1:1)."""
    mc_records = load_mc_task(mc_json_path)
    by_norm: dict[str, dict[str, Any]] = {}
    for rec in mc_records:
        key = normalize_identity(rec["question"])
        if key in by_norm:
            raise AssertionError(f"Duplicated normalized question in mc_task.json: {key}")
        by_norm[key] = rec

    rows = []
    used_keys: set[str] = set()
    for _, q in questions.iterrows():
        key = q["question_norm"]
        if key not in by_norm:
            raise AssertionError(f"No mc_task.json match for question: {q['question']!r}")
        if key in used_keys:
            raise AssertionError(f"mc_task.json record matched twice: {q['question']!r}")
        used_keys.add(key)
        rec = by_norm[key]
        validated = validate_mc_targets(
            question=q["question"],
            mc0=rec["mc0_targets"],
            mc1=rec["mc1_targets"],
            mc2=rec["mc2_targets"],
            best_answer=q["best_answer"],
            best_incorrect=q["best_incorrect_answer"],
        )
        qid = q["question_id"]
        for t in validated["mc0_targets"]:
            t["answer_id"] = make_answer_id(
                qid, t["text"], "correct" if t["label"] == 1 else "incorrect"
            )
        for t in validated["mc1_targets"]:
            t["answer_id"] = make_answer_id(
                qid, t["text"], "correct" if t["label"] == 1 else "incorrect"
            )
        for t in validated["mc2_targets"]:
            t["answer_id"] = make_answer_id(
                qid, t["text"], "correct" if t["label"] == 1 else "incorrect"
            )
        row = q.to_dict()
        row.update(validated)
        rows.append(row)

    unmatched = set(by_norm) - used_keys
    if unmatched:
        # Official JSON may contain only the same 790; still fail if any unused
        # when counts differ.
        if len(by_norm) != len(questions):
            sample = next(iter(unmatched))
            raise AssertionError(
                f"{len(unmatched)} mc_task.json questions unmatched; e.g. {sample!r}"
            )

    out = pd.DataFrame(rows)
    assert len(out) == len(questions)
    return out
