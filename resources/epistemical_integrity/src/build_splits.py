"""Deterministic greedy split assignment balanced on Type/Category/richness/MC."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd

TARGET_COUNTS = {
    "feature_selection": 316,
    "optimization": 237,
    "behavior_validation": 118,
    "holdout_test_behavior": 119,
}

SPLIT_NAMES = list(TARGET_COUNTS.keys())

COST_WEIGHTS = {
    "count": 10.0,
    "category": 4.0,
    "type": 2.0,
    "richness": 2.0,
    "mc": 1.0,
}


def _mc_eligibility_key(row: pd.Series | dict[str, Any]) -> str:
    return (
        f"mc0={int(bool(row['mc0_eligible']))}|"
        f"mc1={int(bool(row['mc1_eligible']))}|"
        f"mc2={int(bool(row['mc2_eligible']))}"
    )


def _combo_key(row: pd.Series | dict[str, Any]) -> str:
    return f"{row['type']}|{row['richness_bin']}|{_mc_eligibility_key(row)}"


def _props_error(counts: Counter[str], n: int, global_props: dict[str, float]) -> float:
    if n <= 0:
        return sum(abs(v) for v in global_props.values())
    err = 0.0
    for k, gp in global_props.items():
        err += abs(counts.get(k, 0) / n - gp)
    # Include keys present only locally
    for k, c in counts.items():
        if k not in global_props:
            err += c / n
    return err


class _SplitState:
    """Incremental counters for fast cost evaluation."""

    def __init__(self, global_props: dict[str, dict[str, float]]):
        self.global_props = global_props
        self.members: dict[str, list[str]] = {s: [] for s in SPLIT_NAMES}
        self.type_counts = {s: Counter() for s in SPLIT_NAMES}
        self.cat_counts = {s: Counter() for s in SPLIT_NAMES}
        self.rich_counts = {s: Counter() for s in SPLIT_NAMES}
        self.mc_counts = {s: Counter() for s in SPLIT_NAMES}

    def n(self, split: str) -> int:
        return len(self.members[split])

    def add(self, split: str, qid: str, meta: dict[str, Any]) -> None:
        self.members[split].append(qid)
        self.type_counts[split][meta["type"]] += 1
        self.cat_counts[split][meta["category"]] += 1
        self.rich_counts[split][meta["richness_bin"]] += 1
        self.mc_counts[split][meta["mc_key"]] += 1

    def remove(self, split: str, qid: str, meta: dict[str, Any]) -> None:
        self.members[split].remove(qid)
        self.type_counts[split][meta["type"]] -= 1
        self.cat_counts[split][meta["category"]] -= 1
        self.rich_counts[split][meta["richness_bin"]] -= 1
        self.mc_counts[split][meta["mc_key"]] -= 1

    def assignment_cost(self, split: str, meta: dict[str, Any]) -> float:
        target = TARGET_COUNTS[split]
        n = self.n(split)
        if n >= target:
            return float("inf")
        new_n = n + 1
        # Capacity pressure: prefer emptier splits relative to target
        fill_ratio = new_n / target
        remaining = (target - new_n) / target
        count_capacity_error = abs(1.0 - fill_ratio) * 0.25 + max(0.0, -remaining)

        type_c = self.type_counts[split].copy()
        cat_c = self.cat_counts[split].copy()
        rich_c = self.rich_counts[split].copy()
        mc_c = self.mc_counts[split].copy()
        type_c[meta["type"]] += 1
        cat_c[meta["category"]] += 1
        rich_c[meta["richness_bin"]] += 1
        mc_c[meta["mc_key"]] += 1

        return (
            COST_WEIGHTS["count"] * count_capacity_error
            + COST_WEIGHTS["category"]
            * _props_error(cat_c, new_n, self.global_props["category"])
            + COST_WEIGHTS["type"] * _props_error(type_c, new_n, self.global_props["type"])
            + COST_WEIGHTS["richness"]
            * _props_error(rich_c, new_n, self.global_props["richness"])
            + COST_WEIGHTS["mc"] * _props_error(mc_c, new_n, self.global_props["mc"])
        )

    def global_cost(self) -> float:
        total = 0.0
        n_all = sum(TARGET_COUNTS.values())
        for split in SPLIT_NAMES:
            n = self.n(split)
            target = TARGET_COUNTS[split]
            count_capacity_error = abs(n - target) / target
            total += (
                COST_WEIGHTS["count"] * count_capacity_error
                + COST_WEIGHTS["category"]
                * _props_error(self.cat_counts[split], n, self.global_props["category"])
                + COST_WEIGHTS["type"]
                * _props_error(self.type_counts[split], n, self.global_props["type"])
                + COST_WEIGHTS["richness"]
                * _props_error(self.rich_counts[split], n, self.global_props["richness"])
                + COST_WEIGHTS["mc"]
                * _props_error(self.mc_counts[split], n, self.global_props["mc"])
                + COST_WEIGHTS["count"] * abs(n / n_all - target / n_all)
            )
        return total


def build_splits(questions: pd.DataFrame, split_seed: int = 42) -> pd.DataFrame:
    """Assign each question to exactly one split with balanced greedy + swaps."""
    if len(questions) != sum(TARGET_COUNTS.values()):
        raise ValueError(
            f"Expected {sum(TARGET_COUNTS.values())} questions, got {len(questions)}"
        )

    df = questions.copy()
    rng = np.random.default_rng(split_seed)

    metas: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        metas[row["question_id"]] = {
            "type": row["type"],
            "category": row["category"],
            "richness_bin": row["richness_bin"],
            "mc_key": _mc_eligibility_key(row),
            "combo": _combo_key(row),
        }

    n = len(df)
    global_props = {
        "type": {k: v / n for k, v in Counter(df["type"]).items()},
        "category": {k: v / n for k, v in Counter(df["category"]).items()},
        "richness": {k: v / n for k, v in Counter(df["richness_bin"]).items()},
        "mc": {k: v / n for k, v in Counter(m["mc_key"] for m in metas.values()).items()},
    }

    combo_counts = Counter(m["combo"] for m in metas.values())
    by_category: dict[str, list[str]] = defaultdict(list)
    for qid, meta in metas.items():
        by_category[meta["category"]].append(qid)

    ordered_qids: list[str] = []
    for cat in sorted(by_category.keys(), key=lambda c: (len(by_category[c]), c)):
        qids = by_category[cat][:]
        rng.shuffle(qids)
        qids.sort(
            key=lambda qid: (
                combo_counts[metas[qid]["combo"]],
                metas[qid]["type"],
                metas[qid]["richness_bin"],
                qid,
            )
        )
        ordered_qids.extend(qids)

    state = _SplitState(global_props)
    split_of: dict[str, str] = {}

    for qid in ordered_qids:
        meta = metas[qid]
        best_split = None
        best_cost = float("inf")
        # Prefer splits farthest from capacity among equal costs via cost fn
        for split in SPLIT_NAMES:
            cost = state.assignment_cost(split, meta)
            if cost < best_cost or (
                cost == best_cost and (best_split is None or split < best_split)
            ):
                best_cost = cost
                best_split = split
        if best_split is None or best_cost == float("inf"):
            # Fallback: any split with remaining capacity
            for split in SPLIT_NAMES:
                if state.n(split) < TARGET_COUNTS[split]:
                    best_split = split
                    break
        if best_split is None:
            raise RuntimeError(f"Could not assign question {qid}")
        state.add(best_split, qid, meta)
        split_of[qid] = best_split

    # Seeded pairwise-swap refinement with incremental state
    current = state.global_cost()
    max_passes = 25
    candidates_per_pass = 2000
    for _ in range(max_passes):
        improved = False
        for _ in range(candidates_per_pass):
            s1, s2 = (str(x) for x in rng.choice(SPLIT_NAMES, size=2, replace=False))
            a = state.members[s1][int(rng.integers(0, len(state.members[s1])))]
            b = state.members[s2][int(rng.integers(0, len(state.members[s2])))]
            ma, mb = metas[a], metas[b]
            state.remove(s1, a, ma)
            state.remove(s2, b, mb)
            state.add(s1, b, mb)
            state.add(s2, a, ma)
            new_cost = state.global_cost()
            if new_cost + 1e-12 < current:
                current = new_cost
                split_of[a] = s2
                split_of[b] = s1
                improved = True
            else:
                state.remove(s1, b, mb)
                state.remove(s2, a, ma)
                state.add(s1, a, ma)
                state.add(s2, b, mb)
        if not improved:
            break

    for split in SPLIT_NAMES:
        if state.n(split) != TARGET_COUNTS[split]:
            raise AssertionError(
                f"Split {split} has {state.n(split)} questions, expected {TARGET_COUNTS[split]}"
            )

    out = df.copy()
    out["split"] = out["question_id"].map(split_of)
    out["split_seed"] = split_seed
    assert out["split"].isna().sum() == 0
    assert set(out["split"]) == set(SPLIT_NAMES)
    return out


def split_quality_report(questions: pd.DataFrame) -> dict[str, Any]:
    """Per-split diagnostics and max absolute proportion deviations."""
    n = len(questions)
    global_type = Counter(questions["type"])
    global_cat = Counter(questions["category"])
    global_rich = Counter(questions["richness_bin"])

    report: dict[str, Any] = {"overall": {"question_count": n}, "splits": {}}
    max_dev = {"type": 0.0, "category": 0.0, "richness_bin": 0.0}

    for split, g in questions.groupby("split", sort=True):
        sn = len(g)
        type_counts = Counter(g["type"])
        cat_counts = Counter(g["category"])
        rich_counts = Counter(g["richness_bin"])
        report["splits"][split] = {
            "question_count": sn,
            "type_counts": dict(type_counts),
            "type_proportions": {k: v / sn for k, v in type_counts.items()},
            "category_counts": dict(cat_counts),
            "category_proportions": {k: v / sn for k, v in cat_counts.items()},
            "richness_bin_counts": dict(rich_counts),
            "richness_bin_proportions": {k: v / sn for k, v in rich_counts.items()},
            "n_correct_unique": {
                "mean": float(g["n_correct_unique"].mean()),
                "median": float(g["n_correct_unique"].median()),
                "min": int(g["n_correct_unique"].min()),
                "max": int(g["n_correct_unique"].max()),
            },
            "n_incorrect_unique": {
                "mean": float(g["n_incorrect_unique"].mean()),
                "median": float(g["n_incorrect_unique"].median()),
                "min": int(g["n_incorrect_unique"].min()),
                "max": int(g["n_incorrect_unique"].max()),
            },
            "mc0_eligible": int(g["mc0_eligible"].sum()),
            "mc1_eligible": int(g["mc1_eligible"].sum()),
            "mc2_eligible": int(g["mc2_eligible"].sum()),
        }

        for k, gv in global_type.items():
            max_dev["type"] = max(max_dev["type"], abs(type_counts.get(k, 0) / sn - gv / n))
        for k, gv in global_cat.items():
            max_dev["category"] = max(
                max_dev["category"], abs(cat_counts.get(k, 0) / sn - gv / n)
            )
        for k, gv in global_rich.items():
            max_dev["richness_bin"] = max(
                max_dev["richness_bin"], abs(rich_counts.get(k, 0) / sn - gv / n)
            )

    report["max_abs_proportion_deviation"] = max_dev
    return report


def split_distribution_frame(questions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(questions)
    for split, g in questions.groupby("split", sort=True):
        sn = len(g)
        for col in ("type", "category", "richness_bin"):
            for key, cnt in Counter(g[col]).items():
                rows.append(
                    {
                        "split": split,
                        "variable": col,
                        "value": key,
                        "count": cnt,
                        "proportion": cnt / sn,
                        "global_proportion": float((questions[col] == key).mean()),
                        "abs_deviation": abs(
                            cnt / sn - (questions[col] == key).sum() / n
                        ),
                    }
                )
    return pd.DataFrame(rows)
