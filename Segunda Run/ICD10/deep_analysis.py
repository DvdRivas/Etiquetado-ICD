#!/usr/bin/env python3
"""
deep_analysis.py — In-depth analysis of 10 ICD labelling runs.

Produces:
  - Console: readable summary with tables (tabulate)
  - deep_analysis_report.json: raw numbers
  - unstable_cases.csv: unstable diagnoses for manual review

Sections:
  1 Loading and validation      5 match_type distribution
  2 Aggregate metrics per run   6 Cumulative NF dictionary effect
  3 Row-level stability         7 Agreement at two levels of the hierarchy
  4 LLM judge variance          8 Hierarchical depth vs instability

Parts 7 and 8 read columns that icd-experiment.py started writing later
(icd_code_complete, category, hierarchical_distance). Against older CSVs they
report that they were skipped instead of failing.

The script is ICD-version agnostic: it detects whether the CSVs carry
`icd11_code` or `icd10_code` and adapts, so the same file works unchanged in
the ICD11 and ICD10 folders.

Everything reads diagnosis_en, like the rest of the pipeline. The
diagnosis_es column is preserved in the CSVs as source data but is never
used here.

Usage:
  python deep_analysis.py [--runs-dir <path>] [--n-runs 10]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

# ── Constants ───────────────────────────────────────────────────────────────
T_CRIT_9 = 2.262  # Student's t, 95%, df=9

# Candidate names of the assigned-code column, in priority order.
CODE_COLUMN_CANDIDATES = ["icd11_code", "icd10_code"]

# Row-level stability categories
CATEGORIES = [
    "stable_consistent",
    "stable_inconsistent",
    "unstable",
    "NF_persistent",
    "NF_intermittent",
]


def detect_code_column(df: pd.DataFrame) -> str:
    """
    Detect which column holds the assigned code (icd11_code or icd10_code).
    Keeps this script usable from both folders without edits.

    A CSV may transiently carry BOTH columns — the ICD10 folder inherited
    `icd11_code` from when it was a copy of the ICD-11 script, and
    icd-experiment.py only renames it on its next run. When both are present
    the populated one wins, so the analysis never reports on an empty column.
    """
    present = [c for c in CODE_COLUMN_CANDIDATES if c in df.columns]

    if not present:
        print(f"[ERROR] No code column found. Expected one of "
              f"{CODE_COLUMN_CANDIDATES}, got: {list(df.columns)}")
        sys.exit(1)

    if len(present) == 1:
        return present[0]

    # Both present: keep the one holding actual data
    filled = {c: df[c].notna().sum() for c in present}
    best = max(present, key=lambda c: filled[c])
    print(f"[WARN] Several code columns present {filled}; using '{best}'")
    return best


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — Loading and validation
# ═══════════════════════════════════════════════════════════════════════════

def load_runs(runs_dir: Path, n_runs: int) -> dict[int, pd.DataFrame]:
    """Load the n CSVs, trying several encodings, return dict run->df."""
    runs = {}
    for i in range(1, n_runs + 1):
        path = runs_dir / f"dataset-run{i}.csv"
        if not path.exists():
            print(f"[ERROR] Does not exist: {path}")
            sys.exit(1)
        # Try UTF-8 first, then cp1252
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"[ERROR] Could not decode {path}")
            sys.exit(1)
        runs[i] = df
    return runs


def validate_runs(runs: dict[int, pd.DataFrame]) -> list[str]:
    """Validate that every CSV has the same rows and columns.
    Returns a list of warnings (empty = all good)."""
    warnings = []
    ref = runs[1]
    ref_filenames = set(ref["filename"])
    ref_cols = set(ref.columns)

    for i, df in runs.items():
        # Columns
        if set(df.columns) != ref_cols:
            extra = set(df.columns) - ref_cols
            missing = ref_cols - set(df.columns)
            warnings.append(
                f"Run {i}: columns differ. Extra={extra}, Missing={missing}"
            )
        # Rows
        if len(df) != len(ref):
            warnings.append(f"Run {i}: {len(df)} rows vs {len(ref)} in run1")
        cur_filenames = set(df["filename"])
        if cur_filenames != ref_filenames:
            only_ref = ref_filenames - cur_filenames
            only_cur = cur_filenames - ref_filenames
            warnings.append(
                f"Run {i}: filenames differ. Only in run1={only_ref}, "
                f"Only in run{i}={only_cur}"
            )
        # Duplicates
        dups = df["filename"].duplicated().sum()
        if dups > 0:
            warnings.append(f"Run {i}: {dups} duplicated filenames")

    return warnings


def validate_encoding(runs: dict[int, pd.DataFrame]) -> list[str]:
    """Detect encoding differences between runs by comparing diagnosis_en."""
    warnings = []
    ref = runs[1].set_index("filename")["diagnosis_en"]
    for i, df in runs.items():
        if i == 1:
            continue
        cur = df.set_index("filename")["diagnosis_en"]
        common = ref.index.intersection(cur.index)
        mismatches = (ref.loc[common] != cur.loc[common]).sum()
        if mismatches > 0:
            warnings.append(
                f"Run {i}: {mismatches} diagnosis_en differ from run1 "
                f"(possible encoding issue)"
            )
    return warnings


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — Aggregate metrics per run
# ═══════════════════════════════════════════════════════════════════════════

def is_nf(code: str) -> bool:
    return isinstance(code, str) and code.startswith("NF-")


def compute_run_metrics(df: pd.DataFrame, code_col: str) -> dict:
    """Compute metrics for one run under the 3 conventions."""
    total = len(df)

    nf_mask = df[code_col].apply(is_nf)
    n_nf = nf_mask.sum()
    n_resolved = total - n_nf

    # consistency only applies to resolved rows (NF holds NaN)
    resolved = df[~nf_mask]
    n_yes = (resolved["consistency"] == "yes").sum()
    n_no = (resolved["consistency"] == "no").sum()
    n_eval = n_yes + n_no  # should equal n_resolved

    coverage = n_resolved / total
    coherence = n_yes / n_eval if n_eval > 0 else np.nan

    # ── Convention A: NF = FN ──
    # TP = yes, FP = 0 (there is no "assigned a wrong code to an uncoded
    # case"), FN = no + NF, TN = 0
    # In the context of "assigning the right code":
    # TP = consistent (yes), FN = inconsistent (no) + not found (NF)
    tp_a = n_yes
    fn_a = n_no + n_nf
    fp_a = 0
    precision_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else np.nan
    recall_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else np.nan
    f1_a = (
        2 * precision_a * recall_a / (precision_a + recall_a)
        if (precision_a + recall_a) > 0
        else np.nan
    )

    # ── Convention B: NF = TN, inconsistent = FP ──
    # TP = yes, FP = no, TN = NF, FN = 0
    tp_b = n_yes
    fp_b = n_no
    tn_b = n_nf
    fn_b = 0
    precision_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else np.nan
    recall_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else np.nan
    f1_b = (
        2 * precision_b * recall_b / (precision_b + recall_b)
        if (precision_b + recall_b) > 0
        else np.nan
    )

    # ── Convention C: NF = TN, inconsistent = FN ──
    # TP = yes, FP = 0, TN = NF, FN = no
    tp_c = n_yes
    fp_c = 0
    tn_c = n_nf
    fn_c = n_no
    precision_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else np.nan
    recall_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else np.nan
    f1_c = (
        2 * precision_c * recall_c / (precision_c + recall_c)
        if (precision_c + recall_c) > 0
        else np.nan
    )

    return {
        "total": total,
        "resolved": n_resolved,
        "nf": n_nf,
        "yes": n_yes,
        "no": n_no,
        "coverage": coverage,
        "coherence": coherence,
        # Conv A: NF=FN
        "precision_A": precision_a,
        "recall_A": recall_a,
        "f1_A": f1_a,
        # Conv B: NF=TN, inconsistent=FP
        "precision_B": precision_b,
        "recall_B": recall_b,
        "f1_B": f1_b,
        # Conv C: NF=TN, inconsistent=FN
        "precision_C": precision_c,
        "recall_C": recall_c,
        "f1_C": f1_c,
    }


def aggregate_metrics(all_metrics: dict[int, dict]) -> dict:
    """Mean, std, 95% CI (Student's t, df=9) across the 10 runs."""
    keys = [
        "coverage", "coherence",
        "precision_A", "recall_A", "f1_A",
        "precision_B", "recall_B", "f1_B",
        "precision_C", "recall_C", "f1_C",
    ]
    agg = {}
    n = len(all_metrics)
    for k in keys:
        vals = np.array([all_metrics[r][k] for r in sorted(all_metrics)])
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)
        se = std / np.sqrt(n)
        ci_lo = mean - T_CRIT_9 * se
        ci_hi = mean + T_CRIT_9 * se
        agg[k] = {
            "mean": float(mean),
            "std": float(std),
            "ci95_lo": float(ci_lo),
            "ci95_hi": float(ci_hi),
        }
    return agg


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — Row-level assignment stability
# ═══════════════════════════════════════════════════════════════════════════

def build_code_matrix(runs: dict[int, pd.DataFrame], code_col: str) -> pd.DataFrame:
    """Build a filename x run matrix holding the assigned code."""
    frames = []
    for i, df in sorted(runs.items()):
        s = df.set_index("filename")[code_col].rename(f"run{i}")
        frames.append(s)
    return pd.concat(frames, axis=1)


def build_consistency_matrix(runs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Build a filename x run matrix holding consistency."""
    frames = []
    for i, df in sorted(runs.items()):
        s = df.set_index("filename")["consistency"].rename(f"run{i}")
        frames.append(s)
    return pd.concat(frames, axis=1)


def build_match_type_matrix(runs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Build a filename x run matrix holding match_type."""
    frames = []
    for i, df in sorted(runs.items()):
        s = df.set_index("filename")["match_type"].rename(f"run{i}")
        frames.append(s)
    return pd.concat(frames, axis=1)


def classify_stability(
    code_matrix: pd.DataFrame,
    consistency_matrix: pd.DataFrame,
    runs: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Classify every filename into a stability category."""
    n_runs = code_matrix.shape[1]
    records = []

    # Take diagnosis_en from run1 as the reference
    diag_map = runs[1].set_index("filename")["diagnosis_en"].to_dict()

    for fname, row in code_matrix.iterrows():
        codes = row.tolist()
        cons_row = consistency_matrix.loc[fname].tolist()
        counter = Counter(codes)
        modal_code, modal_count = counter.most_common(1)[0]
        n_unique = len(counter)
        agreement_pct = modal_count / n_runs

        all_nf = all(is_nf(c) for c in codes)
        any_nf = any(is_nf(c) for c in codes)
        all_same = n_unique == 1

        # Determine modal coherence over the runs that were evaluated
        if all_nf:
            category = "NF_persistent"
        elif any_nf and not all_nf:
            category = "NF_intermittent"
        elif all_same:
            # Check whether coherence is consistent
            eval_cons = [c for c in cons_row if c in ("yes", "no")]
            if eval_cons and all(c == "yes" for c in eval_cons):
                category = "stable_consistent"
            elif eval_cons and all(c == "no" for c in eval_cons):
                category = "stable_inconsistent"
            else:
                # Same code but mixed coherence -> reported in Part 4.
                # Classify by majority.
                n_yes = sum(1 for c in eval_cons if c == "yes")
                n_no = sum(1 for c in eval_cons if c == "no")
                if n_yes >= n_no:
                    category = "stable_consistent"
                else:
                    category = "stable_inconsistent"
        else:
            category = "unstable"

        records.append({
            "filename": fname,
            "diagnosis_en": diag_map.get(fname, ""),
            "category": category,
            "modal_code": modal_code,
            "modal_count": modal_count,
            "agreement_pct": agreement_pct,
            "n_unique_codes": n_unique,
            "all_codes": dict(counter),
        })

    return pd.DataFrame(records)


def stability_by_match_type(
    code_matrix: pd.DataFrame,
    match_type_matrix: pd.DataFrame,
    stability_df: pd.DataFrame,
) -> dict:
    """Cross stability with match_type. Checks whether exact_* is 100% stable."""
    results = {"by_match_type": {}, "deterministic_violations": []}

    for fname in code_matrix.index:
        mt_row = match_type_matrix.loc[fname]
        match_types = set(mt_row.tolist())
        cat = stability_df.loc[
            stability_df["filename"] == fname, "category"
        ].iloc[0]

        for mt in match_types:
            if mt not in results["by_match_type"]:
                results["by_match_type"][mt] = {
                    "total": 0,
                    **{c: 0 for c in CATEGORIES},
                }
            results["by_match_type"][mt]["total"] += 1
            results["by_match_type"][mt][cat] += 1

        # Check the determinism of exact_complete and exact_core
        if cat == "unstable":
            exact_runs = [
                mt_row.index[j]
                for j, mt in enumerate(mt_row)
                if mt in ("exact_complete", "exact_core")
            ]
            if exact_runs:
                # An unstable case that was exact in some run: potential bug
                codes_in_exact = {
                    r: code_matrix.loc[fname, r]
                    for r in exact_runs
                }
                results["deterministic_violations"].append({
                    "filename": fname,
                    "exact_runs_codes": codes_in_exact,
                    "all_codes": dict(Counter(code_matrix.loc[fname].tolist())),
                })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — Unstable coherence (LLM judge variance)
# ═══════════════════════════════════════════════════════════════════════════

def find_judge_variance(
    code_matrix: pd.DataFrame,
    consistency_matrix: pd.DataFrame,
    runs: dict[int, pd.DataFrame],
    code_col: str,
) -> pd.DataFrame:
    """Diagnoses with a stable code but variable coherence (mixed yes/no).
    This isolates the LLM judge's variance from the assigner's variance."""
    diag_map = runs[1].set_index("filename")["diagnosis_en"].to_dict()
    records = []

    for fname in code_matrix.index:
        codes = code_matrix.loc[fname].tolist()
        n_unique = len(set(codes))
        if n_unique != 1:
            continue  # only stable codes are of interest
        if is_nf(codes[0]):
            continue  # NF rows are not evaluated

        cons_vals = consistency_matrix.loc[fname].tolist()
        eval_vals = [c for c in cons_vals if c in ("yes", "no")]
        if not eval_vals:
            continue
        if len(set(eval_vals)) == 1:
            continue  # coherence is stable, not of interest here

        n_yes = sum(1 for c in eval_vals if c == "yes")
        n_no = sum(1 for c in eval_vals if c == "no")

        records.append({
            "filename": fname,
            "diagnosis_en": diag_map.get(fname, ""),
            code_col: codes[0],
            "n_yes": n_yes,
            "n_no": n_no,
            "n_eval": len(eval_vals),
            "pct_yes": n_yes / len(eval_vals),
            "consistency_detail": {
                f"run{i+1}": cons_vals[i]
                for i in range(len(cons_vals))
            },
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — match_type distribution
# ═══════════════════════════════════════════════════════════════════════════

def match_type_distribution(runs: dict[int, pd.DataFrame]) -> dict:
    """Count/percentage of match_type per run + aggregate."""
    all_types = sorted(
        set().union(*(df["match_type"].unique() for df in runs.values()))
    )
    per_run = {}
    matrix = {mt: [] for mt in all_types}

    for i, df in sorted(runs.items()):
        counts = df["match_type"].value_counts()
        row = {}
        for mt in all_types:
            c = int(counts.get(mt, 0))
            row[mt] = {"count": c, "pct": c / len(df)}
            matrix[mt].append(c)
        per_run[i] = row

    # Aggregate
    agg = {}
    for mt in all_types:
        vals = np.array(matrix[mt], dtype=float)
        agg[mt] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "min": int(np.min(vals)),
            "max": int(np.max(vals)),
        }

    # Compare runs 1-2 vs 3-10 (possible pipeline change)
    comparison = {}
    for mt in all_types:
        early = matrix[mt][:2]
        late = matrix[mt][2:]
        comparison[mt] = {
            "early_mean": float(np.mean(early)),
            "late_mean": float(np.mean(late)),
            "diff": float(np.mean(late) - np.mean(early)),
        }

    return {
        "per_run": per_run,
        "aggregate": agg,
        "early_vs_late": comparison,
        "all_types": all_types,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — Effect of the cumulative NF dictionary
# ═══════════════════════════════════════════════════════════════════════════

def analyze_nf_drift(code_matrix: pd.DataFrame) -> dict:
    """Analyse whether NFs in later runs are a subset of the early ones."""
    n_runs = code_matrix.shape[1]
    run_cols = [f"run{i}" for i in range(1, n_runs + 1)]

    nf_sets = {}
    for col in run_cols:
        nf_sets[col] = set(
            code_matrix.index[code_matrix[col].apply(is_nf)]
        )

    # Are later NFs a subset of the early ones?
    nf_run1 = nf_sets["run1"]
    subset_analysis = {}
    for i in range(2, n_runs + 1):
        nf_ri = nf_sets[f"run{i}"]
        only_in_ri = nf_ri - nf_run1
        only_in_r1 = nf_run1 - nf_ri
        subset_analysis[f"run{i}"] = {
            "nf_count": len(nf_ri),
            "shared_with_run1": len(nf_ri & nf_run1),
            "only_in_this_run": sorted(only_in_ri),
            "in_run1_not_here": sorted(only_in_r1),
            "is_subset_of_run1": nf_ri.issubset(nf_run1),
        }

    # Persistent NF vs churn
    all_nf = set.intersection(*nf_sets.values())
    any_nf = set.union(*nf_sets.values())
    churn = any_nf - all_nf  # diagnoses entering/leaving NF

    # Is independence compromised?
    independence_compromised = len(all_nf) > 0 and all(
        nf_sets[f"run{i}"].issubset(any_nf)
        for i in range(2, n_runs + 1)
    )

    return {
        "nf_per_run": {k: len(v) for k, v in nf_sets.items()},
        "nf_persistent_all_10": sorted(all_nf),
        "nf_persistent_count": len(all_nf),
        "nf_any_run": sorted(any_nf),
        "nf_any_count": len(any_nf),
        "nf_churn": sorted(churn),
        "nf_churn_count": len(churn),
        "subset_analysis": subset_analysis,
        "independence_warning": independence_compromised,
    }



# ═══════════════════════════════════════════════════════════════════════════
# PART 7 — Agreement at two levels of the hierarchy
# ═══════════════════════════════════════════════════════════════════════════

def build_column_matrix(runs: dict[int, pd.DataFrame], column: str) -> pd.DataFrame | None:
    """filename x run matrix for any column, or None when it is absent."""
    frames = []
    for i, df in sorted(runs.items()):
        if column not in df.columns:
            return None
        frames.append(df.set_index("filename")[column].rename(f"run{i}"))
    return pd.concat(frames, axis=1)


def analyze_level_agreement(runs: dict[int, pd.DataFrame]) -> dict | None:
    """
    Separate DISAGREEING ON THE SUBCODE from DISAGREEING ON THE CATEGORY.

    Part 3 counts a row as unstable whenever the runs differ, but the two
    kinds of difference are not equally serious. Landing on C22.0 in one run
    and C22.1 in another still places the case in the same category (C22,
    malignant neoplasm of liver); landing on C22 versus K75 does not. Only
    the second is a classification error in the sense that matters, because
    `category` is what the pipeline actually reports.

    Requires `icd_code_complete` and `category`; returns None when the CSVs
    predate them.
    """
    full_matrix = build_column_matrix(runs, "icd_code_complete")
    cat_matrix  = build_column_matrix(runs, "category")
    if full_matrix is None or cat_matrix is None:
        return None

    buckets = {
        "stable_both":        [],   # same subcode in every run
        "subcode_only":       [],   # subcode varies, category holds
        "category_disagree":  [],   # category itself varies
    }

    for fname in full_matrix.index:
        full_vals = [v for v in full_matrix.loc[fname].tolist() if isinstance(v, str) and v]
        cat_vals  = [v for v in cat_matrix.loc[fname].tolist() if isinstance(v, str) and v]
        if not full_vals or not cat_vals:
            continue

        n_full = len(set(full_vals))
        n_cat  = len(set(cat_vals))

        if n_cat > 1:
            bucket = "category_disagree"
        elif n_full > 1:
            bucket = "subcode_only"
        else:
            bucket = "stable_both"

        buckets[bucket].append({
            "filename":       fname,
            "codes":          dict(Counter(full_vals)),
            "categories":     dict(Counter(cat_vals)),
            "n_unique_code":  n_full,
            "n_unique_category": n_cat,
        })

    total = sum(len(v) for v in buckets.values())
    return {
        "total":   total,
        "counts":  {k: len(v) for k, v in buckets.items()},
        "pct":     {k: (len(v) / total if total else 0.0) for k, v in buckets.items()},
        "buckets": buckets,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PART 8 — Hierarchical depth vs instability
# ═══════════════════════════════════════════════════════════════════════════

def analyze_depth_stability(runs: dict[int, pd.DataFrame],
                            stability_df: pd.DataFrame) -> dict | None:
    """
    Cross `hierarchical_distance` with the Part 3 stability category.

    The question is whether categories sitting deeper in the classification
    are harder to pin down. Depth is a property of the classification, not of
    the pipeline — chapter II nests three levels of blocks while chapter I
    nests one — so if instability tracks depth, the difficulty comes from the
    terminology rather than from the model.

    Requires `hierarchical_distance`; returns None when the CSVs predate it.
    """
    depth_matrix = build_column_matrix(runs, "hierarchical_distance")
    if depth_matrix is None:
        return None

    category_of = dict(zip(stability_df["filename"], stability_df["category"]))

    by_depth: dict[str, dict] = {}
    for fname in depth_matrix.index:
        values = [v for v in depth_matrix.loc[fname].tolist()
                  if pd.notna(v) and str(v).strip() != ""]
        if not values:
            continue
        # Modal depth: with a stable category the depth is stable too, so the
        # mode simply ignores runs where the category changed.
        depth = str(Counter(str(v).strip() for v in values).most_common(1)[0][0])
        stability = category_of.get(fname)
        if stability is None:
            continue

        entry = by_depth.setdefault(depth, {"total": 0, **{c: 0 for c in CATEGORIES}})
        entry["total"] += 1
        entry[stability] += 1

    for entry in by_depth.values():
        unstable = entry["unstable"] + entry["NF_intermittent"]
        entry["unstable_pct"] = unstable / entry["total"] if entry["total"] else 0.0

    return {"by_depth": by_depth}


# ═══════════════════════════════════════════════════════════════════════════
# Methodological warnings
# ═══════════════════════════════════════════════════════════════════════════

def collect_warnings(
    validation_warnings: list[str],
    encoding_warnings: list[str],
    stability_results: dict,
    nf_analysis: dict,
    judge_variance_df: pd.DataFrame,
    match_dist: dict,
) -> list[str]:
    """Collect every methodological warning."""
    warnings = []

    # Basic validation
    for w in validation_warnings + encoding_warnings:
        warnings.append(f"[VALIDATION] {w}")

    # Determinism violated
    violations = stability_results.get("deterministic_violations", [])
    if violations:
        warnings.append(
            f"[BUG] {len(violations)} case(s) with match_type exact_complete/"
            f"exact_core are NOT 100% stable across runs. "
            f"This breaks the determinism expectation. "
            f"Filenames: {[v['filename'] for v in violations]}"
        )

    # NF non-independence
    if nf_analysis["independence_warning"]:
        warnings.append(
            f"[INDEPENDENCE] The cumulative NF dictionary compromises "
            f"independence between runs. {nf_analysis['nf_persistent_count']} "
            f"diagnoses are NF in EVERY run (cached). "
            f"The Part 2 confidence intervals assume independence, but the "
            f"runs share state through the NF dictionary."
        )

    nf_churn = nf_analysis["nf_churn_count"]
    if nf_churn > 0:
        warnings.append(
            f"[NF-CHURN] {nf_churn} diagnosis(es) enter/leave NF across runs. "
            f"This means the pipeline DOES retry some diagnoses, but the "
            f"outcome varies."
        )

    # Judge variance
    if len(judge_variance_df) > 0:
        warnings.append(
            f"[LLM-JUDGE] {len(judge_variance_df)} diagnosis(es) have a stable "
            f"code but variable coherence across runs. That is evaluator "
            f"(MedGemma) variance, not pipeline variance. It affects the "
            f"coherence/precision metrics."
        )

    # Early vs late difference in match_type
    comp = match_dist["early_vs_late"]
    for mt, vals in comp.items():
        if abs(vals["diff"]) > 2:  # mean difference above 2 cases
            warnings.append(
                f"[MATCH-SHIFT] match_type '{mt}' differs between runs 1-2 "
                f"(mean={vals['early_mean']:.1f}) and runs 3-10 "
                f"(mean={vals['late_mean']:.1f}). Check whether the pipeline "
                f"changed."
            )

    return warnings


# ═══════════════════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════════════════

def print_part2(all_metrics: dict, agg: dict):
    """Print per-run and aggregate metrics."""
    print("\n" + "=" * 76)
    print("  PART 2 — Aggregate metrics per run")
    print("=" * 76)

    # Per-run table: coverage and coherence
    rows = []
    for i in sorted(all_metrics):
        m = all_metrics[i]
        rows.append([
            f"Run {i}",
            m["resolved"], m["nf"],
            f"{m['coverage']:.1%}",
            m["yes"], m["no"],
            f"{m['coherence']:.1%}",
        ])
    print("\n── Coverage and Coherence ──")
    print(tabulate(
        rows,
        headers=["Run", "Resolved", "NF", "Coverage",
                 "Yes", "No", "Coherence"],
        tablefmt="simple",
    ))

    # Per-run table: conventions A/B/C
    print("\n── Precision / Recall / F1 under three conventions ──")
    print("  Conv A: NF=FN (penalises non-coverage)")
    print("  Conv B: NF=TN, inconsistent=FP (penalises false positives)")
    print("  Conv C: NF=TN, inconsistent=FN (penalises false negatives)")

    rows = []
    for i in sorted(all_metrics):
        m = all_metrics[i]
        rows.append([
            f"Run {i}",
            f"{m['precision_A']:.3f}", f"{m['recall_A']:.3f}",
            f"{m['f1_A']:.3f}",
            f"{m['precision_B']:.3f}", f"{m['recall_B']:.3f}",
            f"{m['f1_B']:.3f}",
            f"{m['precision_C']:.3f}", f"{m['recall_C']:.3f}",
            f"{m['f1_C']:.3f}",
        ])
    print(tabulate(
        rows,
        headers=["Run", "P_A", "R_A", "F1_A", "P_B", "R_B", "F1_B",
                 "P_C", "R_C", "F1_C"],
        tablefmt="simple",
    ))

    # Aggregate
    print("\n── Aggregate (10 runs, 95% CI Student's t df=9) ──")
    rows = []
    for k in ["coverage", "coherence",
              "precision_A", "recall_A", "f1_A",
              "precision_B", "recall_B", "f1_B",
              "precision_C", "recall_C", "f1_C"]:
        a = agg[k]
        rows.append([
            k,
            f"{a['mean']:.4f}",
            f"{a['std']:.4f}",
            f"[{a['ci95_lo']:.4f}, {a['ci95_hi']:.4f}]",
        ])
    print(tabulate(
        rows,
        headers=["Metric", "Mean", "Std", "95% CI"],
        tablefmt="simple",
    ))


def print_part3(stability_df: pd.DataFrame, stability_results: dict):
    """Print the row-level stability analysis."""
    print("\n" + "=" * 76)
    print("  PART 3 — Row-level assignment stability")
    print("=" * 76)

    cat_counts = stability_df["category"].value_counts()
    total = len(stability_df)
    rows = []
    for cat in CATEGORIES:
        c = cat_counts.get(cat, 0)
        rows.append([cat, c, f"{c/total:.1%}"])
    print(tabulate(
        rows,
        headers=["Category", "Cases", "%"],
        tablefmt="simple",
    ))

    # Top unstable
    unstable = stability_df[stability_df["category"] == "unstable"].sort_values(
        "agreement_pct"
    )
    if len(unstable) > 0:
        print(f"\n── Top unstable (max 20, lowest agreement first) ──")
        rows = []
        for _, r in unstable.head(20).iterrows():
            codes_str = ", ".join(
                f"{code}({cnt})"
                for code, cnt in sorted(
                    r["all_codes"].items(), key=lambda x: -x[1]
                )
            )
            rows.append([
                r["filename"],
                r["diagnosis_en"][:60] + ("..." if len(r["diagnosis_en"]) > 60 else ""),
                f"{r['agreement_pct']:.0%}",
                r["n_unique_codes"],
                codes_str,
            ])
        print(tabulate(
            rows,
            headers=["Filename", "Diagnosis", "Agreement", "#Codes",
                     "Codes(n)"],
            tablefmt="simple",
        ))

    # Cross with match_type
    violations = stability_results.get("deterministic_violations", [])
    if violations:
        print(f"\n── ⚠ DETERMINISM VIOLATIONS ({len(violations)}) ──")
        for v in violations:
            print(f"  {v['filename']}: {v['all_codes']}")
            print(f"    exact runs: {v['exact_runs_codes']}")
    else:
        print("\n  ✓ exact_complete/exact_core: 100% deterministic")

    # Stability by match_type
    print("\n── Stability crossed with match_type ──")
    by_mt = stability_results["by_match_type"]
    rows = []
    for mt in sorted(by_mt):
        d = by_mt[mt]
        rows.append([
            mt, d["total"],
            d["stable_consistent"], d["stable_inconsistent"],
            d["unstable"], d["NF_persistent"], d["NF_intermittent"],
        ])
    print(tabulate(
        rows,
        headers=["match_type", "Cases", "Stable.Cons", "Stable.Incons",
                 "Unstable", "NF_perm", "NF_inter"],
        tablefmt="simple",
    ))


def print_part4(judge_variance_df: pd.DataFrame, code_col: str):
    """Print the LLM judge variance analysis."""
    print("\n" + "=" * 76)
    print("  PART 4 — LLM judge variance (stable code, variable coherence)")
    print("=" * 76)

    if len(judge_variance_df) == 0:
        print("  No case with a stable code and variable coherence.")
        return

    print(f"  {len(judge_variance_df)} case(s) detected:\n")
    rows = []
    for _, r in judge_variance_df.iterrows():
        rows.append([
            r["filename"],
            r["diagnosis_en"][:50] + ("..." if len(r["diagnosis_en"]) > 50 else ""),
            r[code_col],
            f"{r['n_yes']}/{r['n_eval']}",
            f"{r['pct_yes']:.0%}",
        ])
    print(tabulate(
        rows,
        headers=["Filename", "Diagnosis", "Code", "Yes/Eval", "%Yes"],
        tablefmt="simple",
    ))


def print_part5(match_dist: dict):
    """Print the match_type distribution."""
    print("\n" + "=" * 76)
    print("  PART 5 — match_type distribution")
    print("=" * 76)

    all_types = match_dist["all_types"]

    # Per run
    print("\n── Count per run ──")
    rows = []
    for i in sorted(match_dist["per_run"]):
        row = [f"Run {i}"]
        for mt in all_types:
            c = match_dist["per_run"][i][mt]["count"]
            row.append(c)
        rows.append(row)
    print(tabulate(
        rows,
        headers=["Run"] + all_types,
        tablefmt="simple",
    ))

    # Aggregate
    print("\n── Aggregate (mean ± std) ──")
    rows = []
    for mt in all_types:
        a = match_dist["aggregate"][mt]
        rows.append([
            mt,
            f"{a['mean']:.1f} ± {a['std']:.1f}",
            a["min"], a["max"],
        ])
    print(tabulate(
        rows,
        headers=["match_type", "Mean ± Std", "Min", "Max"],
        tablefmt="simple",
    ))

    # Early vs late
    print("\n── Runs 1-2 vs Runs 3-10 ──")
    rows = []
    for mt in all_types:
        c = match_dist["early_vs_late"][mt]
        flag = " ⚠" if abs(c["diff"]) > 2 else ""
        rows.append([
            mt,
            f"{c['early_mean']:.1f}",
            f"{c['late_mean']:.1f}",
            f"{c['diff']:+.1f}{flag}",
        ])
    print(tabulate(
        rows,
        headers=["match_type", "Mean 1-2", "Mean 3-10", "Diff"],
        tablefmt="simple",
    ))


def print_part6(nf_analysis: dict):
    """Print the cumulative NF analysis."""
    print("\n" + "=" * 76)
    print("  PART 6 — Effect of the cumulative NF dictionary")
    print("=" * 76)

    print("\n── NF per run ──")
    rows = []
    for k, v in sorted(nf_analysis["nf_per_run"].items()):
        rows.append([k, v])
    print(tabulate(rows, headers=["Run", "NF count"], tablefmt="simple"))

    print(f"\n  Persistent NF (every run):    "
          f"{nf_analysis['nf_persistent_count']}")
    print(f"  NF in at least one run:       "
          f"{nf_analysis['nf_any_count']}")
    print(f"  NF with churn (enter/leave):  "
          f"{nf_analysis['nf_churn_count']}")

    if nf_analysis["nf_churn_count"] > 0:
        print(f"\n  Diagnoses with intermittent NF:")
        for fname in nf_analysis["nf_churn"]:
            print(f"    - {fname}")

    sa = nf_analysis["subset_analysis"]
    print("\n── Relation with run1 NF ──")
    rows = []
    for run_key in sorted(sa, key=lambda x: int(x.replace("run", ""))):
        d = sa[run_key]
        rows.append([
            run_key, d["nf_count"], d["shared_with_run1"],
            len(d["only_in_this_run"]),
            len(d["in_run1_not_here"]),
            "Yes" if d["is_subset_of_run1"] else "No",
        ])
    print(tabulate(
        rows,
        headers=["Run", "NF", "Shared\nwith run1",
                 "Only in\nthis run", "In run1\nnot here", "⊆ run1?"],
        tablefmt="simple",
    ))



def print_part7(level: dict | None):
    """Print the two-level agreement analysis."""
    print("\n" + "=" * 76)
    print("  PART 7 — Agreement at two levels of the hierarchy")
    print("=" * 76)

    if level is None:
        print("  Skipped: the CSVs have no icd_code_complete / category columns.")
        print("  Re-run icd-experiment.py to populate them.")
        return

    labels = {
        "stable_both":       "Same subcode in every run",
        "subcode_only":      "Subcode varies, CATEGORY holds",
        "category_disagree": "CATEGORY itself varies",
    }
    rows = []
    for key, label in labels.items():
        rows.append([label, level["counts"][key], f"{level['pct'][key]:.1%}"])
    print(tabulate(rows, headers=["Outcome", "Cases", "%"], tablefmt="simple"))

    mild = level["counts"]["subcode_only"]
    if mild:
        print(f"\n  {mild} case(s) would count as unstable in Part 3 yet agree on")
        print("  the category the pipeline actually reports:")
        for rec in level["buckets"]["subcode_only"][:10]:
            codes = ", ".join(f"{c}({n})" for c, n in
                              sorted(rec["codes"].items(), key=lambda x: -x[1]))
            cat = next(iter(rec["categories"]))
            print(f"    {rec['filename']:<26} {cat:<8} <- {codes}")

    severe = level["counts"]["category_disagree"]
    if severe:
        print(f"\n  {severe} case(s) disagree on the category itself:")
        for rec in level["buckets"]["category_disagree"][:10]:
            cats = ", ".join(f"{c}({n})" for c, n in
                             sorted(rec["categories"].items(), key=lambda x: -x[1]))
            print(f"    {rec['filename']:<26} {cats}")


def print_part8(depth: dict | None):
    """Print the depth vs instability analysis."""
    print("\n" + "=" * 76)
    print("  PART 8 — Hierarchical depth vs instability")
    print("=" * 76)

    if depth is None:
        print("  Skipped: the CSVs have no hierarchical_distance column.")
        print("  Re-run icd-experiment.py to populate it.")
        return

    by_depth = depth["by_depth"]
    if not by_depth:
        print("  No usable depth values.")
        return

    def _sort_key(k: str):
        try:
            return (0, int(k))
        except ValueError:
            return (1, 0)

    rows = []
    for key in sorted(by_depth, key=_sort_key):
        d = by_depth[key]
        rows.append([
            key, d["total"],
            d["stable_consistent"], d["stable_inconsistent"],
            d["unstable"], d["NF_persistent"], d["NF_intermittent"],
            f"{d['unstable_pct']:.1%}",
        ])
    print(tabulate(
        rows,
        headers=["Depth", "Cases", "Stable.Cons", "Stable.Incons",
                 "Unstable", "NF_perm", "NF_inter", "% unstable"],
        tablefmt="simple",
    ))
    print("\n  Depth counts the levels between the chapter and the category, so it")
    print("  is a property of the classification, not of the pipeline. Instability")
    print("  rising with depth would point at the terminology, not at the model.")


def print_warnings(warnings: list[str]):
    """Print the methodological warnings."""
    print("\n" + "=" * 76)
    print("  ⚠ METHODOLOGICAL WARNINGS")
    print("=" * 76)
    if not warnings:
        print("  No warnings.")
        return
    for i, w in enumerate(warnings, 1):
        print(f"\n  {i}. {w}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════

class NumpyEncoder(json.JSONEncoder):
    """Convert numpy types to native Python types for JSON."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def export_json(
    output_path: Path,
    all_metrics: dict,
    agg: dict,
    stability_summary: dict,
    stability_results: dict,
    judge_variance_records: list,
    match_dist: dict,
    nf_analysis: dict,
    warnings: list[str],
    level_agreement: dict | None = None,
    depth_stability: dict | None = None,
):
    """Export every raw number to JSON."""
    metrics_ser = {}
    for i, m in all_metrics.items():
        metrics_ser[str(i)] = m

    report = {
        "part2_metrics_per_run": metrics_ser,
        "part2_aggregate": agg,
        "part3_stability_summary": stability_summary,
        "part3_deterministic_violations": stability_results.get(
            "deterministic_violations", []
        ),
        "part4_judge_variance": judge_variance_records,
        "part5_match_type_distribution": {
            "aggregate": match_dist["aggregate"],
            "early_vs_late": match_dist["early_vs_late"],
        },
        "part6_nf_analysis": {
            "nf_per_run": nf_analysis["nf_per_run"],
            "nf_persistent_count": nf_analysis["nf_persistent_count"],
            "nf_any_count": nf_analysis["nf_any_count"],
            "nf_churn_count": nf_analysis["nf_churn_count"],
            "nf_churn_filenames": nf_analysis["nf_churn"],
            "independence_warning": nf_analysis["independence_warning"],
        },
        "part7_level_agreement": (
            {"counts": level_agreement["counts"],
             "pct": level_agreement["pct"],
             "subcode_only": level_agreement["buckets"]["subcode_only"],
             "category_disagree": level_agreement["buckets"]["category_disagree"]}
            if level_agreement else None
        ),
        "part8_depth_stability": depth_stability,
        "warnings": warnings,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"\n[OK] JSON report: {output_path}")


def export_unstable_csv(output_path: Path, stability_df: pd.DataFrame):
    """Export unstable diagnoses to CSV."""
    unstable = stability_df[
        stability_df["category"].isin(["unstable", "NF_intermittent"])
    ].copy()
    # Convert dict to string for the CSV
    unstable["all_codes"] = unstable["all_codes"].apply(
        lambda d: "; ".join(f"{k}({v})" for k, v in sorted(d.items(), key=lambda x: -x[1]))
    )
    unstable = unstable.sort_values("agreement_pct")
    unstable.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[OK] Unstable cases: {output_path} ({len(unstable)} rows)")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="In-depth analysis of 10 ICD labelling runs"
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory holding dataset-run{1..N}.csv (default: script folder)",
    )
    parser.add_argument("--n-runs", type=int, default=10)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    print("=" * 76)
    print("  DEEP ANALYSIS — 10 ICD labelling runs")
    print("=" * 76)

    # ── PART 1 ──
    print("\n── PART 1: Loading and validation ──")
    runs = load_runs(args.runs_dir, args.n_runs)
    print(f"  Loaded {len(runs)} runs from {args.runs_dir}")

    code_col = detect_code_column(runs[1])
    print(f"  Code column detected: {code_col}")

    val_warnings = validate_runs(runs)
    enc_warnings = validate_encoding(runs)

    if val_warnings:
        for w in val_warnings:
            print(f"  [WARN] {w}")
    else:
        print("  ✓ Every run: same rows, same columns")

    if enc_warnings:
        for w in enc_warnings:
            print(f"  [WARN] {w}")
    else:
        print("  ✓ Consistent encoding across runs")

    # ── PART 2 ──
    all_metrics = {}
    for i, df in runs.items():
        all_metrics[i] = compute_run_metrics(df, code_col)
    agg = aggregate_metrics(all_metrics)
    print_part2(all_metrics, agg)

    # ── PART 3 ──
    code_matrix = build_code_matrix(runs, code_col)
    consistency_matrix = build_consistency_matrix(runs)
    match_type_matrix = build_match_type_matrix(runs)

    stability_df = classify_stability(code_matrix, consistency_matrix, runs)
    stability_results = stability_by_match_type(
        code_matrix, match_type_matrix, stability_df
    )
    print_part3(stability_df, stability_results)

    # Stability summary for JSON
    cat_counts = stability_df["category"].value_counts().to_dict()
    stability_summary = {
        cat: {"count": cat_counts.get(cat, 0),
              "pct": cat_counts.get(cat, 0) / len(stability_df)}
        for cat in CATEGORIES
    }

    # ── PART 4 ──
    judge_variance_df = find_judge_variance(
        code_matrix, consistency_matrix, runs, code_col
    )
    print_part4(judge_variance_df, code_col)

    # ── PART 5 ──
    match_dist = match_type_distribution(runs)
    print_part5(match_dist)

    # ── PART 6 ──
    nf_analysis = analyze_nf_drift(code_matrix)
    print_part6(nf_analysis)

    # ── PART 7 ──
    level_agreement = analyze_level_agreement(runs)
    print_part7(level_agreement)

    # ── PART 8 ──
    depth_stability = analyze_depth_stability(runs, stability_df)
    print_part8(depth_stability)

    # ── Warnings ──
    warnings = collect_warnings(
        val_warnings, enc_warnings,
        stability_results, nf_analysis,
        judge_variance_df, match_dist,
    )
    print_warnings(warnings)

    # ── Export ──
    judge_records = (
        judge_variance_df.to_dict(orient="records")
        if len(judge_variance_df) > 0
        else []
    )

    export_json(
        script_dir / "deep_analysis_report.json",
        all_metrics, agg, stability_summary, stability_results,
        judge_records, match_dist, nf_analysis, warnings,
        level_agreement, depth_stability,
    )
    export_unstable_csv(script_dir / "unstable_cases.csv", stability_df)


if __name__ == "__main__":
    main()
