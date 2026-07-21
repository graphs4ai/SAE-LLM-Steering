"""CLI / programmatic entrypoint for the TruthfulQA split dataset pipeline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .build_beliefs import build_belief_candidates, build_belief_triples
from .build_splits import (
    TARGET_COUNTS,
    build_splits,
    split_distribution_frame,
    split_quality_report,
)
from .build_variants import build_variants
from .join_mc_targets import join_mc_targets
from .parse_truthfulqa import (
    add_richness_bins,
    file_sha256,
    load_truthfulqa_csv,
)
from .validate_outputs import validate_outputs

SPLIT_MANIFEST_COLS = [
    "question_id",
    "question",
    "type",
    "category",
    "source",
    "split",
    "split_seed",
    "richness_bin",
    "n_correct_unique",
    "n_incorrect_unique",
    "n_alt_correct",
    "n_alt_incorrect",
    "mc0_eligible",
    "mc1_eligible",
    "mc2_eligible",
]


def build_dataset(
    csv_path: str | Path,
    mc_json_path: str | Path,
    output_dir: str | Path,
    *,
    split_seed: int = 42,
    belief_template_version: str = "v1",
    semantic_filter_cache: str | Path | None = None,
    max_beliefs_per_polarity: int | None = None,
    allow_direct_beliefs: bool = False,
    emit_leave_one_out: bool = False,
    strict: bool = True,
    reports_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full deterministic dataset construction pipeline."""
    csv_path = Path(csv_path)
    mc_json_path = Path(mc_json_path)
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir) if reports_dir else output_dir.parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if emit_leave_one_out:
        raise NotImplementedError(
            "Leave-one-out scoring variants are reserved for a robustness experiment; "
            "set --emit-leave-one-out false (default)."
        )

    csv_hash = file_sha256(csv_path)
    mc_hash = file_sha256(mc_json_path)

    questions = load_truthfulqa_csv(str(csv_path))
    questions = join_mc_targets(questions, str(mc_json_path))
    questions = add_richness_bins(questions)
    questions = build_splits(questions, split_seed=split_seed)

    # Persist canonical question table
    questions.to_parquet(output_dir / "questions.parquet", index=False)

    manifest = questions[SPLIT_MANIFEST_COLS].copy()
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)

    cache_path = (
        Path(semantic_filter_cache)
        if semantic_filter_cache
        else output_dir / "semantic_filter.jsonl"
    )
    beliefs, rejected = build_belief_candidates(
        questions,
        allow_direct_beliefs=allow_direct_beliefs,
        semantic_filter_cache=cache_path,
    )
    beliefs.to_parquet(output_dir / "belief_candidates.parquet", index=False)
    if len(rejected):
        rejected.to_csv(reports_dir / "rejected_beliefs.csv", index=False)
    else:
        pd.DataFrame(
            columns=[
                "belief_id",
                "question_id",
                "source_answer",
                "polarity",
                "rejection_reason",
            ]
        ).to_csv(reports_dir / "rejected_beliefs.csv", index=False)

    triples = build_belief_triples(
        questions,
        beliefs,
        pair_seed=split_seed,
        max_beliefs_per_polarity=max_beliefs_per_polarity,
    )
    triples.to_parquet(output_dir / "belief_triples.parquet", index=False)

    variant_summary = build_variants(questions, beliefs, triples, output_dir)

    quality = split_quality_report(questions)
    dist = split_distribution_frame(questions)
    dist.to_csv(reports_dir / "split_distribution.csv", index=False)

    validation = validate_outputs(
        questions, beliefs, triples, output_dir, strict=strict
    )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_version": __version__,
        "belief_template_version": belief_template_version,
        "split_seed": split_seed,
        "target_counts": TARGET_COUNTS,
        "source_files": {
            "csv": str(csv_path),
            "csv_sha256": csv_hash,
            "mc_json": str(mc_json_path),
            "mc_json_sha256": mc_hash,
        },
        "n_questions": int(len(questions)),
        "richness_bin_counts": questions["richness_bin"].value_counts().to_dict(),
        "answer_richness_diagnostics": {
            "n_min_alt_ge_1": int((questions["min_alt"] >= 1).sum()),
            "n_min_alt_ge_2": int((questions["min_alt"] >= 2).sum()),
            "n_incorrect_unique_ge_3": int((questions["n_incorrect_unique"] >= 3).sum()),
        },
        "split_quality": quality,
        "variant_summary": variant_summary,
        "validation": validation,
        "options": {
            "max_beliefs_per_polarity": max_beliefs_per_polarity,
            "allow_direct_beliefs": allow_direct_beliefs,
            "emit_leave_one_out": emit_leave_one_out,
            "strict": strict,
            "semantic_filter_cache": str(cache_path),
        },
    }
    with open(reports_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    with open(reports_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2, ensure_ascii=False, default=str)

    return summary


def _parse_optional_int(value: str) -> int | None:
    if value.lower() in {"null", "none", ""}:
        return None
    return int(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build TruthfulQA belief-conditioned split datasets"
    )
    parser.add_argument("--csv", required=True, help="Path to TruthfulQA.csv")
    parser.add_argument("--mc-json", required=True, help="Path to mc_task.json")
    parser.add_argument("--output-dir", required=True, help="Processed data directory")
    parser.add_argument("--reports-dir", default=None, help="Reports directory")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--belief-template-version", default="v1")
    parser.add_argument(
        "--semantic-filter-cache",
        default=None,
        help="JSONL cache for semantic relation judgments",
    )
    parser.add_argument(
        "--max-beliefs-per-polarity",
        type=_parse_optional_int,
        default=None,
    )
    parser.add_argument("--allow-direct-beliefs", action="store_true", default=False)
    parser.add_argument("--emit-leave-one-out", action="store_true", default=False)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    summary = build_dataset(
        csv_path=args.csv,
        mc_json_path=args.mc_json,
        output_dir=args.output_dir,
        split_seed=args.split_seed,
        belief_template_version=args.belief_template_version,
        semantic_filter_cache=args.semantic_filter_cache,
        max_beliefs_per_polarity=args.max_beliefs_per_polarity,
        allow_direct_beliefs=args.allow_direct_beliefs,
        emit_leave_one_out=args.emit_leave_one_out,
        strict=args.strict,
        reports_dir=args.reports_dir,
    )
    print(json.dumps({"ok": summary["validation"]["ok"], "split_counts": summary["validation"]["split_counts"]}, indent=2))


if __name__ == "__main__":
    main()
