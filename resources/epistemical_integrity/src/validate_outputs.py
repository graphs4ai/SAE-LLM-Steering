"""Validate pipeline outputs against acceptance criteria."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .build_splits import TARGET_COUNTS
from .build_variants import BINARY_ONLY_SPLITS, FULL_FORMAT_SPLITS


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_outputs(
    questions: pd.DataFrame,
    beliefs: pd.DataFrame,
    triples: pd.DataFrame,
    output_dir: str | Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    errors: list[str] = []
    warnings: list[str] = []

    # --- Split integrity ---
    counts = questions["split"].value_counts().to_dict()
    for split, expected in TARGET_COUNTS.items():
        got = int(counts.get(split, 0))
        if got != expected:
            errors.append(f"Split {split}: expected {expected}, got {got}")

    if questions["question_id"].duplicated().any():
        errors.append("Duplicated question_id in questions")

    split_sets = {
        s: set(g["question_id"]) for s, g in questions.groupby("split")
    }
    for s1, a in split_sets.items():
        for s2, b in split_sets.items():
            if s1 >= s2:
                continue
            inter = a & b
            if inter:
                errors.append(f"Split intersection {s1}∩{s2} non-empty ({len(inter)})")

    union = set().union(*split_sets.values()) if split_sets else set()
    if union != set(questions["question_id"]):
        errors.append("Union of splits != full question set")

    # --- Belief integrity ---
    rejected_ids = set(
        beliefs.loc[beliefs["rejection_reason"].notna(), "belief_id"]
    )
    binary_ok = beliefs[beliefs["eligible_binary"]]
    if not binary_ok.empty:
        missing_prov = binary_ok[
            binary_ok["source_answer"].isna() | binary_ok["source_field"].isna()
        ]
        if not missing_prov.empty:
            errors.append("Eligible beliefs missing provenance")
        bad_rel = binary_ok[
            ~binary_ok["semantic_relation_to_mc0_target"].isin(
                ["equivalent", "entails", "related_but_not_equivalent"]
            )
        ]
        if not bad_rel.empty:
            errors.append(
                "Eligible binary beliefs with invalid semantic_relation_to_mc0_target"
            )

    if triples.empty:
        errors.append("No belief triples generated")
    else:
        # Opposite polarity
        bel = beliefs.set_index("belief_id")
        for _, t in triples.iterrows():
            c = bel.loc[t["correct_belief_id"]]
            i = bel.loc[t["incorrect_belief_id"]]
            if c["polarity"] != "correct" or i["polarity"] != "incorrect":
                errors.append(f"Triple {t['belief_pair_id']} has wrong polarities")
                break
            if t["correct_belief_id"] in rejected_ids or t["incorrect_belief_id"] in rejected_ids:
                errors.append(f"Rejected belief in triple {t['belief_pair_id']}")
                break

        # Question weights sum to 1 per question (via pair_weight)
        for qid, g in triples.groupby("question_id"):
            w = float(g["pair_weight"].sum())
            if abs(w - 1.0) > 1e-6:
                errors.append(f"Question {qid} pair_weight sum={w}, expected 1")
                break

    # --- Variant files ---
    holdout_qids = set(questions.loc[questions["split"] == "holdout_test_behavior", "question_id"])
    for split in TARGET_COUNTS:
        split_dir = output_dir / split
        mc0 = _read_jsonl(split_dir / "mc0.jsonl")
        mc1 = _read_jsonl(split_dir / "mc1.jsonl")
        mc2 = _read_jsonl(split_dir / "mc2.jsonl")

        if not mc0:
            warnings.append(f"{split}/mc0.jsonl empty or missing")

        if split in BINARY_ONLY_SPLITS:
            if mc1 or mc2:
                errors.append(f"{split} must contain only MC0, found MC1/MC2")
            if (split_dir / "mc1.jsonl").exists() or (split_dir / "mc2.jsonl").exists():
                errors.append(f"{split} has MC1/MC2 files")
        elif split in FULL_FORMAT_SPLITS:
            if not mc1:
                errors.append(f"{split} missing MC1")
            if not mc2:
                errors.append(f"{split} missing MC2")

        # MC0 both orders
        orders = {r.get("answer_order") for r in mc0}
        if mc0 and not {"true-first", "false-first"}.issubset(orders):
            errors.append(f"{split} MC0 missing both answer orders")

        # Rejected beliefs must not appear
        for rec in mc0 + mc1 + mc2:
            bid = rec.get("belief_id")
            if bid in rejected_ids:
                errors.append(f"Rejected belief {bid} in {split} tasks")
                break
            if rec["question_id"] not in set(
                questions.loc[questions["split"] == split, "question_id"]
            ):
                errors.append(f"Question/split mismatch in {split}")
                break

        # Official MC targets match for a sample of questions
        qmap = questions.set_index("question_id")
        if mc1:
            sample = mc1[0]
            official = [
                (t["text"], t["label"]) for t in qmap.loc[sample["question_id"], "mc1_targets"]
            ]
            got = [(t["text"], t["label"]) for t in sample["targets"]]
            if official != got:
                errors.append(f"{split} MC1 targets diverge from official JSON")
        if mc2:
            sample = mc2[0]
            official = [
                (t["text"], t["label"]) for t in qmap.loc[sample["question_id"], "mc2_targets"]
            ]
            got = [(t["text"], t["label"]) for t in sample["targets"]]
            if official != got:
                errors.append(f"{split} MC2 targets diverge from official JSON")

    # Neutral prompt dedup by hash within generation summary is soft-checked
    neutral_hashes = set()
    for split in TARGET_COUNTS:
        for rec in _read_jsonl(output_dir / split / "mc0.jsonl"):
            if rec.get("belief_condition") == "neutral" and rec.get("neutral_prompt_hash"):
                # Same question should reuse the same hash
                key = (rec["question_id"], rec["neutral_prompt_hash"])
                neutral_hashes.add(key)

    # Holdout sealed check: development splits must not contain holdout IDs
    for split in ("feature_selection", "optimization", "behavior_validation"):
        for fmt in ("mc0", "mc1", "mc2"):
            for rec in _read_jsonl(output_dir / split / f"{fmt}.jsonl"):
                if rec["question_id"] in holdout_qids:
                    errors.append("Holdout question leaked into development split")
                    break

    report = {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "split_counts": {k: int(counts.get(k, 0)) for k in TARGET_COUNTS},
        "n_beliefs": int(len(beliefs)),
        "n_eligible_binary": int(beliefs["eligible_binary"].sum()),
        "n_rejected": int(beliefs["rejection_reason"].notna().sum()),
        "n_triples": int(len(triples)),
        "n_questions_with_triples": int(triples["question_id"].nunique())
        if len(triples)
        else 0,
    }
    if strict and errors:
        raise AssertionError(
            "Validation failed:\n" + "\n".join(f"- {e}" for e in errors)
        )
    return report
