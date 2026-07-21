"""Parse and canonicalize TruthfulQA CSV rows."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

import pandas as pd

EXPECTED_COLUMNS = [
    "Type",
    "Category",
    "Question",
    "Best Answer",
    "Best Incorrect Answer",
    "Correct Answers",
    "Incorrect Answers",
    "Source",
]


def split_reference_answers(value: object) -> list[str]:
    """Split semicolon-delimited reference answers, dropping blanks."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def normalize_identity(text: str) -> str:
    """Conservative normalization for joins and exact deduplication."""
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.!?]+$", "", text)
    return text.casefold()


def normalize_for_prompt(text: str) -> str:
    """Light normalization used when rendering prompts."""
    text = unicodedata.normalize("NFKC", str(text)).strip()
    text = re.sub(r"\s+", " ", text)
    if text and text[-1] not in ".!?":
        text = text + "."
    return text


def make_question_id(question: str) -> str:
    """Stable ID from normalized question text (never row index)."""
    digest = hashlib.sha256(normalize_identity(question).encode("utf-8")).hexdigest()
    return f"tqa_{digest[:16]}"


def make_answer_id(question_id: str, answer: str, polarity: str) -> str:
    payload = f"{question_id}|{polarity}|{normalize_identity(answer)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"ans_{digest}"


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_preserve(items: list[str]) -> list[str]:
    """Deduplicate by identity-normalized text while preserving first surface form."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = normalize_identity(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def load_truthfulqa_csv(csv_path: str) -> pd.DataFrame:
    """Load TruthfulQA CSV with standards-compliant parsing and stable IDs."""
    df = pd.read_csv(csv_path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"TruthfulQA CSV missing columns: {missing}")

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        question = str(row["Question"])
        correct_raw = split_reference_answers(row["Correct Answers"])
        incorrect_raw = split_reference_answers(row["Incorrect Answers"])
        correct_unique = unique_preserve(correct_raw)
        incorrect_unique = unique_preserve(incorrect_raw)
        best_correct = str(row["Best Answer"]).strip()
        best_incorrect = str(row["Best Incorrect Answer"]).strip()

        best_norm = normalize_identity(best_correct)
        best_inc_norm = normalize_identity(best_incorrect)
        alt_correct = [a for a in correct_unique if normalize_identity(a) != best_norm]
        alt_incorrect = [
            a for a in incorrect_unique if normalize_identity(a) != best_inc_norm
        ]

        qid = make_question_id(question)
        records.append(
            {
                "question_id": qid,
                "type": str(row["Type"]),
                "category": str(row["Category"]),
                "question": question,
                "question_norm": normalize_identity(question),
                "best_answer": best_correct,
                "best_incorrect_answer": best_incorrect,
                "correct_answers_raw": correct_raw,
                "incorrect_answers_raw": incorrect_raw,
                "correct_answers_unique": correct_unique,
                "incorrect_answers_unique": incorrect_unique,
                "n_correct_raw": len(correct_raw),
                "n_incorrect_raw": len(incorrect_raw),
                "n_correct_unique": len(correct_unique),
                "n_incorrect_unique": len(incorrect_unique),
                "n_alt_correct": len(alt_correct),
                "n_alt_incorrect": len(alt_incorrect),
                "min_alt": min(len(alt_correct), len(alt_incorrect)),
                "source": None if pd.isna(row["Source"]) else str(row["Source"]),
            }
        )

    out = pd.DataFrame(records)
    if out["question_id"].duplicated().any():
        dups = out.loc[out["question_id"].duplicated(keep=False), "question"].tolist()
        raise ValueError(f"Duplicate question_id after normalization: {dups[:5]}")
    if out["question_norm"].duplicated().any():
        raise ValueError("Duplicate normalized questions in CSV")
    return out


def richness_bin(min_alt: int) -> str:
    if min_alt <= 0:
        return "none"
    if min_alt == 1:
        return "low"
    if min_alt == 2:
        return "medium"
    return "rich"


def add_richness_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["richness_bin"] = out["min_alt"].astype(int).map(richness_bin)
    return out
