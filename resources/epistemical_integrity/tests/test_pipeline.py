"""Unit tests for TruthfulQA parsing, join, splits, beliefs, and variants."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.build_beliefs import (  # noqa: E402
    build_belief_candidates,
    build_belief_triples,
    reject_belief,
)
from src.build_splits import TARGET_COUNTS, build_splits  # noqa: E402
from src.build_variants import build_variants  # noqa: E402
from src.join_mc_targets import join_mc_targets  # noqa: E402
from src.parse_truthfulqa import (  # noqa: E402
    add_richness_bins,
    load_truthfulqa_csv,
    make_question_id,
    normalize_identity,
    split_reference_answers,
)
from src.validate_outputs import validate_outputs  # noqa: E402

DATA_RAW = ROOT / "data_raw"
CSV_PATH = DATA_RAW / "TruthfulQA.csv"
MC_PATH = DATA_RAW / "mc_task.json"


@pytest.fixture(scope="module")
def questions_joined():
    q = load_truthfulqa_csv(str(CSV_PATH))
    q = join_mc_targets(q, str(MC_PATH))
    return add_richness_bins(q)


def test_split_reference_answers_basic():
    assert split_reference_answers("A; B; C") == ["A", "B", "C"]
    assert split_reference_answers("  A ; ; B ") == ["A", "B"]
    assert split_reference_answers(None) == []
    assert split_reference_answers(float("nan")) == []


def test_quoted_commas_preserved_by_csv_parser():
    df = pd.read_csv(CSV_PATH)
    # At least one field should contain a comma inside a cell without breaking columns
    assert list(df.columns)[:3] == ["Type", "Category", "Question"]
    assert len(df) == 790


def test_normalize_and_stable_ids():
    a = make_question_id("What is this?")
    b = make_question_id("what is this")
    c = make_question_id("What   is this!")
    assert a == b == c
    assert a.startswith("tqa_")
    assert normalize_identity("Café!") == normalize_identity("café")


def test_reject_non_informative():
    assert reject_belief("I have no comment") == "non_informative"
    assert reject_belief("The sky is blue") is None


def test_join_all_questions(questions_joined):
    assert len(questions_joined) == 790
    assert questions_joined["mc0_eligible"].all()
    assert questions_joined["mc1_eligible"].all()
    assert questions_joined["mc2_eligible"].all()


def test_richness_bins_recomputed(questions_joined):
    counts = questions_joined["richness_bin"].value_counts().to_dict()
    assert sum(counts.values()) == 790
    assert counts.get("none", 0) > 0
    # Guide approximate pre-filter diagnostics (recomputed, not hard-coded targets)
    assert (questions_joined["min_alt"] >= 1).sum() == 713
    assert (questions_joined["min_alt"] >= 2).sum() == 548
    assert (questions_joined["n_incorrect_unique"] >= 3).sum() == 663
    assert (
        questions_joined.loc[questions_joined["min_alt"] == 0, "richness_bin"] == "none"
    ).all()


def test_splits_exact_and_reproducible(questions_joined):
    a = build_splits(questions_joined, split_seed=42)
    b = build_splits(questions_joined, split_seed=42)
    assert a["split"].tolist() == b["split"].tolist()
    counts = a["split"].value_counts().to_dict()
    assert counts == TARGET_COUNTS
    assert a["question_id"].is_unique
    sets = {s: set(g["question_id"]) for s, g in a.groupby("split")}
    for s1, x in sets.items():
        for s2, y in sets.items():
            if s1 < s2:
                assert x.isdisjoint(y)
    assert set().union(*sets.values()) == set(a["question_id"])


def test_beliefs_and_triples(questions_joined, tmp_path):
    q = build_splits(questions_joined, split_seed=42)
    beliefs, rejected = build_belief_candidates(
        q, semantic_filter_cache=tmp_path / "cache.jsonl"
    )
    assert beliefs["source_answer"].notna().all()
    assert set(beliefs.loc[beliefs["eligible_binary"], "semantic_relation_to_mc0_target"]).issubset(
        {"equivalent", "entails", "related_but_not_equivalent"}
    )
    # Default path should only mark equivalent/entails as eligible
    assert set(
        beliefs.loc[beliefs["eligible_binary"], "semantic_relation_to_mc0_target"]
    ).issubset({"equivalent", "entails"})

    triples = build_belief_triples(q, beliefs, pair_seed=42)
    assert len(triples) > 0
    bel = beliefs.set_index("belief_id")
    for _, t in triples.sample(min(20, len(triples)), random_state=0).iterrows():
        assert bel.loc[t["correct_belief_id"], "polarity"] == "correct"
        assert bel.loc[t["incorrect_belief_id"], "polarity"] == "incorrect"
        assert pd.isna(bel.loc[t["correct_belief_id"], "rejection_reason"])
    for qid, g in triples.groupby("question_id"):
        assert abs(g["pair_weight"].sum() - 1.0) < 1e-9


def test_variants_by_split(questions_joined, tmp_path):
    q = build_splits(questions_joined, split_seed=42)
    beliefs, _ = build_belief_candidates(
        q, semantic_filter_cache=tmp_path / "cache.jsonl"
    )
    triples = build_belief_triples(q, beliefs, pair_seed=42)
    out = tmp_path / "processed"
    build_variants(q, beliefs, triples, out)

    for split in ("feature_selection", "optimization"):
        assert (out / split / "mc0.jsonl").exists()
        assert not (out / split / "mc1.jsonl").exists()
        assert not (out / split / "mc2.jsonl").exists()
        lines = (out / split / "mc0.jsonl").read_text().strip().splitlines()
        assert lines
        rec = json.loads(lines[0])
        assert rec["format"] == "mc0"
        orders = {json.loads(x)["answer_order"] for x in lines}
        assert {"true-first", "false-first"} <= orders

    for split in ("behavior_validation", "holdout_test_behavior"):
        for fmt in ("mc0", "mc1", "mc2"):
            assert (out / split / f"{fmt}.jsonl").exists()
            assert (out / split / f"{fmt}.jsonl").stat().st_size > 0

    report = validate_outputs(q, beliefs, triples, out, strict=True)
    assert report["ok"]
