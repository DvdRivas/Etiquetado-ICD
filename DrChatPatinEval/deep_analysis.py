#!/usr/bin/env python3
"""
deep_analysis.py — Análisis en profundidad de 10 corridas de etiquetado ICD-11.

Genera:
  - Consola: resumen legible con tablas (tabulate)
  - deep_analysis_report.json: números crudos
  - unstable_cases.csv: diagnósticos inestables para revisión manual

Uso:
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

# ── Constantes ──────────────────────────────────────────────────────────────
T_CRIT_9 = 2.262  # t de Student, 95%, df=9


# ═══════════════════════════════════════════════════════════════════════════
# PARTE 1 — Carga y validación
# ═══════════════════════════════════════════════════════════════════════════

def load_runs(runs_dir: Path, n_runs: int) -> dict[int, pd.DataFrame]:
    """Carga los n CSVs, intenta múltiples encodings, retorna dict run->df."""
    runs = {}
    for i in range(1, n_runs + 1):
        path = runs_dir / f"dataset-run{i}.csv"
        if not path.exists():
            print(f"[ERROR] No existe: {path}")
            sys.exit(1)
        # Intentar UTF-8 primero, luego cp1252
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"[ERROR] No se pudo decodificar {path}")
            sys.exit(1)
        runs[i] = df
    return runs


def validate_runs(runs: dict[int, pd.DataFrame]) -> list[str]:
    """Valida que todos los CSVs tengan las mismas 201 filas y columnas.
    Retorna lista de advertencias (vacía = todo OK)."""
    warnings = []
    ref = runs[1]
    ref_filenames = set(ref["filename"])
    ref_cols = set(ref.columns)

    for i, df in runs.items():
        # Columnas
        if set(df.columns) != ref_cols:
            extra = set(df.columns) - ref_cols
            missing = ref_cols - set(df.columns)
            warnings.append(
                f"Run {i}: columnas difieren. Extra={extra}, Faltan={missing}"
            )
        # Filas
        if len(df) != len(ref):
            warnings.append(f"Run {i}: {len(df)} filas vs {len(ref)} en run1")
        cur_filenames = set(df["filename"])
        if cur_filenames != ref_filenames:
            only_ref = ref_filenames - cur_filenames
            only_cur = cur_filenames - ref_filenames
            warnings.append(
                f"Run {i}: filenames difieren. Solo en run1={only_ref}, "
                f"Solo en run{i}={only_cur}"
            )
        # Duplicados
        dups = df["filename"].duplicated().sum()
        if dups > 0:
            warnings.append(f"Run {i}: {dups} filenames duplicados")

    return warnings


def validate_encoding(runs: dict[int, pd.DataFrame]) -> list[str]:
    """Detecta diferencias de encoding entre corridas comparando diagnosis_es."""
    warnings = []
    ref = runs[1].set_index("filename")["diagnosis_es"]
    for i, df in runs.items():
        if i == 1:
            continue
        cur = df.set_index("filename")["diagnosis_es"]
        common = ref.index.intersection(cur.index)
        mismatches = (ref.loc[common] != cur.loc[common]).sum()
        if mismatches > 0:
            warnings.append(
                f"Run {i}: {mismatches} diagnosis_es difieren de run1 "
                f"(posible encoding)"
            )
    return warnings


# ═══════════════════════════════════════════════════════════════════════════
# PARTE 2 — Métricas agregadas por corrida
# ═══════════════════════════════════════════════════════════════════════════

def is_nf(code: str) -> bool:
    return isinstance(code, str) and code.startswith("NF-")


def compute_run_metrics(df: pd.DataFrame) -> dict:
    """Calcula métricas para una corrida bajo las 3 convenciones."""
    total = len(df)

    nf_mask = df["icd11_code"].apply(is_nf)
    n_nf = nf_mask.sum()
    n_resolved = total - n_nf

    # consistency solo para resueltos (NF tiene NaN)
    resolved = df[~nf_mask]
    n_yes = (resolved["consistency"] == "yes").sum()
    n_no = (resolved["consistency"] == "no").sum()
    n_eval = n_yes + n_no  # debería ser == n_resolved

    coverage = n_resolved / total
    coherence = n_yes / n_eval if n_eval > 0 else np.nan

    # ── Convención A: NF = FN ──
    # TP = yes, FP = 0 (no hay "asignó código incorrecto a caso sin código"),
    # FN = no + NF, TN = 0
    # En contexto de "asignar código correcto":
    # TP = coherente (yes), FN = incoherente (no) + no encontrado (NF)
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

    # ── Convención B: NF = TN, incoherente = FP ──
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

    # ── Convención C: NF = TN, incoherente = FN ──
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
        # Conv B: NF=TN, incoherente=FP
        "precision_B": precision_b,
        "recall_B": recall_b,
        "f1_B": f1_b,
        # Conv C: NF=TN, incoherente=FN
        "precision_C": precision_c,
        "recall_C": recall_c,
        "f1_C": f1_c,
    }


def aggregate_metrics(all_metrics: dict[int, dict]) -> dict:
    """Media, std, IC 95% (t-student df=9) sobre las 10 corridas."""
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
# PARTE 3 — Estabilidad de asignación a nivel de fila
# ═══════════════════════════════════════════════════════════════════════════

def build_code_matrix(runs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Construye matriz filename × run con icd11_code."""
    frames = []
    for i, df in sorted(runs.items()):
        s = df.set_index("filename")["icd11_code"].rename(f"run{i}")
        frames.append(s)
    return pd.concat(frames, axis=1)


def build_consistency_matrix(runs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Construye matriz filename × run con consistency."""
    frames = []
    for i, df in sorted(runs.items()):
        s = df.set_index("filename")["consistency"].rename(f"run{i}")
        frames.append(s)
    return pd.concat(frames, axis=1)


def build_match_type_matrix(runs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Construye matriz filename × run con match_type."""
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
    """Clasifica cada filename en categorías de estabilidad."""
    n_runs = code_matrix.shape[1]
    records = []

    # Obtener diagnosis_es de run1 como referencia
    diag_map = runs[1].set_index("filename")["diagnosis_es"].to_dict()

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

        # Determinar coherencia modal
        # Para corridas donde se asignó el código modal y se evaluó
        if all_nf:
            category = "NF_persistente"
        elif any_nf and not all_nf:
            category = "NF_intermitente"
        elif all_same:
            # Verificar si coherencia es consistente
            eval_cons = [c for c in cons_row if c in ("yes", "no")]
            if eval_cons and all(c == "yes" for c in eval_cons):
                category = "estable_consistente"
            elif eval_cons and all(c == "no" for c in eval_cons):
                category = "estable_inconsistente"
            else:
                # Mismo código pero coherencia mixta → se reporta en Parte 4
                # Clasificar según la mayoría
                n_yes = sum(1 for c in eval_cons if c == "yes")
                n_no = sum(1 for c in eval_cons if c == "no")
                if n_yes >= n_no:
                    category = "estable_consistente"
                else:
                    category = "estable_inconsistente"
        else:
            category = "inestable"

        records.append({
            "filename": fname,
            "diagnosis_es": diag_map.get(fname, ""),
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
    """Cruza estabilidad con match_type. Verifica si exact_* es 100% estable."""
    # Para cada filename, obtener los match_types usados
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
                    "estable_consistente": 0,
                    "estable_inconsistente": 0,
                    "inestable": 0,
                    "NF_persistente": 0,
                    "NF_intermitente": 0,
                }
            results["by_match_type"][mt]["total"] += 1
            results["by_match_type"][mt][cat] += 1

        # Verificar determinismo de exact_complete y exact_core
        if cat == "inestable":
            exact_runs = [
                mt_row.index[j]
                for j, mt in enumerate(mt_row)
                if mt in ("exact_complete", "exact_core")
            ]
            if exact_runs:
                # Un caso inestable que en alguna corrida fue exact: bug potencial
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
# PARTE 4 — Coherencia inestable (varianza del juez LLM)
# ═══════════════════════════════════════════════════════════════════════════

def find_judge_variance(
    code_matrix: pd.DataFrame,
    consistency_matrix: pd.DataFrame,
    runs: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Diagnósticos con código estable pero coherencia variable (yes/no mixto).
    Esto aísla varianza del juez LLM de varianza del asignador."""
    diag_map = runs[1].set_index("filename")["diagnosis_es"].to_dict()
    records = []

    for fname in code_matrix.index:
        codes = code_matrix.loc[fname].tolist()
        n_unique = len(set(codes))
        if n_unique != 1:
            continue  # Solo interesa código estable
        if is_nf(codes[0]):
            continue  # NF no tiene evaluación

        cons_vals = consistency_matrix.loc[fname].tolist()
        eval_vals = [c for c in cons_vals if c in ("yes", "no")]
        if not eval_vals:
            continue
        if len(set(eval_vals)) == 1:
            continue  # Coherencia estable, no interesa aquí

        n_yes = sum(1 for c in eval_vals if c == "yes")
        n_no = sum(1 for c in eval_vals if c == "no")

        records.append({
            "filename": fname,
            "diagnosis_es": diag_map.get(fname, ""),
            "icd11_code": codes[0],
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
# PARTE 5 — Distribución de match_type
# ═══════════════════════════════════════════════════════════════════════════

def match_type_distribution(runs: dict[int, pd.DataFrame]) -> dict:
    """Conteo/porcentaje de match_type por corrida + agregado."""
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

    # Agregado
    agg = {}
    for mt in all_types:
        vals = np.array(matrix[mt], dtype=float)
        agg[mt] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "min": int(np.min(vals)),
            "max": int(np.max(vals)),
        }

    # Comparar runs 1-2 vs 3-10 (posible cambio de pipeline)
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
# PARTE 6 — Efecto del diccionario NF acumulativo
# ═══════════════════════════════════════════════════════════════════════════

def analyze_nf_drift(code_matrix: pd.DataFrame) -> dict:
    """Analiza si NFs en corridas tardías son subconjunto de NFs tempranas."""
    n_runs = code_matrix.shape[1]
    run_cols = [f"run{i}" for i in range(1, n_runs + 1)]

    nf_sets = {}
    for col in run_cols:
        nf_sets[col] = set(
            code_matrix.index[code_matrix[col].apply(is_nf)]
        )

    # ¿NFs tardías ⊆ NFs tempranas?
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

    # NF persistentes vs churn
    all_nf = set.intersection(*nf_sets.values())
    any_nf = set.union(*nf_sets.values())
    churn = any_nf - all_nf  # Diagnósticos que entran/salen de NF

    # Independencia comprometida?
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
# Advertencias metodológicas
# ═══════════════════════════════════════════════════════════════════════════

def collect_warnings(
    validation_warnings: list[str],
    encoding_warnings: list[str],
    stability_results: dict,
    nf_analysis: dict,
    judge_variance_df: pd.DataFrame,
    match_dist: dict,
) -> list[str]:
    """Recopila todas las advertencias metodológicas."""
    warnings = []

    # Validación básica
    for w in validation_warnings + encoding_warnings:
        warnings.append(f"[VALIDACIÓN] {w}")

    # Determinismo violado
    violations = stability_results.get("deterministic_violations", [])
    if violations:
        warnings.append(
            f"[BUG] {len(violations)} caso(s) con match_type exact_complete/"
            f"exact_core NO son 100% estables entre corridas. "
            f"Esto viola la expectativa de determinismo. "
            f"Filenames: {[v['filename'] for v in violations]}"
        )

    # NF no-independencia
    if nf_analysis["independence_warning"]:
        warnings.append(
            f"[INDEPENDENCIA] El diccionario NF acumulativo compromete la "
            f"independencia entre corridas. {nf_analysis['nf_persistent_count']} "
            f"diagnósticos son NF en TODAS las corridas (cacheados). "
            f"Los intervalos de confianza de la Parte 2 asumen independencia, "
            f"pero las corridas comparten estado a través del diccionario NF."
        )

    nf_churn = nf_analysis["nf_churn_count"]
    if nf_churn > 0:
        warnings.append(
            f"[NF-CHURN] {nf_churn} diagnóstico(s) entran/salen de NF entre "
            f"corridas. Esto indica que el pipeline SÍ reintenta algunos "
            f"diagnósticos, pero el resultado varía."
        )

    # Varianza del juez
    if len(judge_variance_df) > 0:
        warnings.append(
            f"[JUEZ-LLM] {len(judge_variance_df)} diagnóstico(s) tienen código "
            f"estable pero coherencia variable entre corridas. Esto es varianza "
            f"del evaluador (MedGemma), no del pipeline. Afecta métricas de "
            f"coherencia/precisión."
        )

    # Diferencia early vs late en match_type
    comp = match_dist["early_vs_late"]
    for mt, vals in comp.items():
        if abs(vals["diff"]) > 2:  # >2 casos de diferencia media
            warnings.append(
                f"[MATCH-SHIFT] match_type '{mt}' difiere entre runs 1-2 "
                f"(media={vals['early_mean']:.1f}) y runs 3-10 "
                f"(media={vals['late_mean']:.1f}). Verificar si hubo cambio "
                f"de pipeline."
            )

    return warnings


# ═══════════════════════════════════════════════════════════════════════════
# Formateo de salida
# ═══════════════════════════════════════════════════════════════════════════

def print_part2(all_metrics: dict, agg: dict):
    """Imprime métricas por corrida y agregadas."""
    print("\n" + "=" * 76)
    print("  PARTE 2 — Métricas agregadas por corrida")
    print("=" * 76)

    # Tabla por corrida: cobertura y coherencia
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
    print("\n── Cobertura y Coherencia ──")
    print(tabulate(
        rows,
        headers=["Run", "Resueltos", "NF", "Cobertura",
                 "Yes", "No", "Coherencia"],
        tablefmt="simple",
    ))

    # Tabla por corrida: convenciones A/B/C
    print("\n── Precision / Recall / F1 bajo tres convenciones ──")
    print("  Conv A: NF=FN (penaliza no-cobertura)")
    print("  Conv B: NF=TN, incoherente=FP (penaliza falsos positivos)")
    print("  Conv C: NF=TN, incoherente=FN (penaliza falsos negativos)")

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

    # Agregado
    print("\n── Agregado (10 corridas, IC 95% t-Student df=9) ──")
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
        headers=["Métrica", "Media", "Std", "IC 95%"],
        tablefmt="simple",
    ))


def print_part3(stability_df: pd.DataFrame, stability_results: dict):
    """Imprime análisis de estabilidad a nivel de fila."""
    print("\n" + "=" * 76)
    print("  PARTE 3 — Estabilidad de asignación a nivel de fila")
    print("=" * 76)

    cat_counts = stability_df["category"].value_counts()
    total = len(stability_df)
    rows = []
    for cat in ["estable_consistente", "estable_inconsistente", "inestable",
                "NF_persistente", "NF_intermitente"]:
        c = cat_counts.get(cat, 0)
        rows.append([cat, c, f"{c/total:.1%}"])
    print(tabulate(
        rows,
        headers=["Categoría", "Casos", "%"],
        tablefmt="simple",
    ))

    # Top inestables
    inestable = stability_df[stability_df["category"] == "inestable"].sort_values(
        "agreement_pct"
    )
    if len(inestable) > 0:
        print(f"\n── Top inestables (máx 20, menor acuerdo primero) ──")
        rows = []
        for _, r in inestable.head(20).iterrows():
            codes_str = ", ".join(
                f"{code}({cnt})"
                for code, cnt in sorted(
                    r["all_codes"].items(), key=lambda x: -x[1]
                )
            )
            rows.append([
                r["filename"],
                r["diagnosis_es"][:60] + ("..." if len(r["diagnosis_es"]) > 60 else ""),
                f"{r['agreement_pct']:.0%}",
                r["n_unique_codes"],
                codes_str,
            ])
        print(tabulate(
            rows,
            headers=["Filename", "Diagnóstico", "Acuerdo", "#Códigos",
                     "Códigos(n)"],
            tablefmt="simple",
        ))

    # Cruce con match_type
    violations = stability_results.get("deterministic_violations", [])
    if violations:
        print(f"\n── ⚠ VIOLACIONES DE DETERMINISMO ({len(violations)}) ──")
        for v in violations:
            print(f"  {v['filename']}: {v['all_codes']}")
            print(f"    exact runs: {v['exact_runs_codes']}")
    else:
        print("\n  ✓ exact_complete/exact_core: 100% deterministas")

    # Estabilidad por match_type
    print("\n── Estabilidad cruzada con match_type ──")
    by_mt = stability_results["by_match_type"]
    rows = []
    for mt in sorted(by_mt):
        d = by_mt[mt]
        stable = d["estable_consistente"] + d["estable_inconsistente"]
        rows.append([
            mt, d["total"],
            d["estable_consistente"], d["estable_inconsistente"],
            d["inestable"], d["NF_persistente"], d["NF_intermitente"],
        ])
    print(tabulate(
        rows,
        headers=["match_type", "Casos", "Est.Cons", "Est.Incons",
                 "Inestable", "NF_perm", "NF_inter"],
        tablefmt="simple",
    ))


def print_part4(judge_variance_df: pd.DataFrame):
    """Imprime análisis de varianza del juez LLM."""
    print("\n" + "=" * 76)
    print("  PARTE 4 — Varianza del juez LLM (código estable, coherencia variable)")
    print("=" * 76)

    if len(judge_variance_df) == 0:
        print("  Ningún caso con código estable y coherencia variable.")
        return

    print(f"  {len(judge_variance_df)} caso(s) detectados:\n")
    rows = []
    for _, r in judge_variance_df.iterrows():
        rows.append([
            r["filename"],
            r["diagnosis_es"][:50] + ("..." if len(r["diagnosis_es"]) > 50 else ""),
            r["icd11_code"],
            f"{r['n_yes']}/{r['n_eval']}",
            f"{r['pct_yes']:.0%}",
        ])
    print(tabulate(
        rows,
        headers=["Filename", "Diagnóstico", "Código", "Yes/Eval", "%Yes"],
        tablefmt="simple",
    ))


def print_part5(match_dist: dict):
    """Imprime distribución de match_type."""
    print("\n" + "=" * 76)
    print("  PARTE 5 — Distribución de match_type")
    print("=" * 76)

    all_types = match_dist["all_types"]

    # Por corrida
    print("\n── Conteo por corrida ──")
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

    # Agregado
    print("\n── Agregado (media ± std) ──")
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
        headers=["match_type", "Media ± Std", "Min", "Max"],
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
        headers=["match_type", "Media 1-2", "Media 3-10", "Diff"],
        tablefmt="simple",
    ))


def print_part6(nf_analysis: dict):
    """Imprime análisis de NF acumulativo."""
    print("\n" + "=" * 76)
    print("  PARTE 6 — Efecto del diccionario NF acumulativo")
    print("=" * 76)

    print("\n── NF por corrida ──")
    rows = []
    for k, v in sorted(nf_analysis["nf_per_run"].items()):
        rows.append([k, v])
    print(tabulate(rows, headers=["Run", "NF count"], tablefmt="simple"))

    print(f"\n  NF persistentes (todas las corridas): "
          f"{nf_analysis['nf_persistent_count']}")
    print(f"  NF en al menos una corrida:           "
          f"{nf_analysis['nf_any_count']}")
    print(f"  NF con churn (entran/salen):           "
          f"{nf_analysis['nf_churn_count']}")

    if nf_analysis["nf_churn_count"] > 0:
        print(f"\n  Diagnósticos con NF intermitente:")
        for fname in nf_analysis["nf_churn"]:
            print(f"    - {fname}")

    sa = nf_analysis["subset_analysis"]
    print("\n── Relación con NF de run1 ──")
    rows = []
    for run_key in sorted(sa, key=lambda x: int(x.replace("run", ""))):
        d = sa[run_key]
        rows.append([
            run_key, d["nf_count"], d["shared_with_run1"],
            len(d["only_in_this_run"]),
            len(d["in_run1_not_here"]),
            "Sí" if d["is_subset_of_run1"] else "No",
        ])
    print(tabulate(
        rows,
        headers=["Run", "NF", "Compartidos\ncon run1",
                 "Solo en\neste run", "En run1\nno aquí", "⊆ run1?"],
        tablefmt="simple",
    ))


def print_warnings(warnings: list[str]):
    """Imprime advertencias metodológicas."""
    print("\n" + "=" * 76)
    print("  ⚠ ADVERTENCIAS METODOLÓGICAS")
    print("=" * 76)
    if not warnings:
        print("  Ninguna advertencia.")
        return
    for i, w in enumerate(warnings, 1):
        print(f"\n  {i}. {w}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Exportación
# ═══════════════════════════════════════════════════════════════════════════

class NumpyEncoder(json.JSONEncoder):
    """Convierte tipos numpy a tipos nativos de Python para JSON."""
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
):
    """Exporta todos los números crudos a JSON."""
    # Convertir métricas a serializable
    metrics_ser = {}
    for i, m in all_metrics.items():
        metrics_ser[str(i)] = m

    report = {
        "parte2_metrics_per_run": metrics_ser,
        "parte2_aggregate": agg,
        "parte3_stability_summary": stability_summary,
        "parte3_deterministic_violations": stability_results.get(
            "deterministic_violations", []
        ),
        "parte4_judge_variance": judge_variance_records,
        "parte5_match_type_distribution": {
            "aggregate": match_dist["aggregate"],
            "early_vs_late": match_dist["early_vs_late"],
        },
        "parte6_nf_analysis": {
            "nf_per_run": nf_analysis["nf_per_run"],
            "nf_persistent_count": nf_analysis["nf_persistent_count"],
            "nf_any_count": nf_analysis["nf_any_count"],
            "nf_churn_count": nf_analysis["nf_churn_count"],
            "nf_churn_filenames": nf_analysis["nf_churn"],
            "independence_warning": nf_analysis["independence_warning"],
        },
        "warnings": warnings,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"\n[OK] Reporte JSON: {output_path}")


def export_unstable_csv(output_path: Path, stability_df: pd.DataFrame):
    """Exporta diagnósticos inestables a CSV."""
    unstable = stability_df[
        stability_df["category"].isin(["inestable", "NF_intermitente"])
    ].copy()
    # Convertir dict a string para CSV
    unstable["all_codes"] = unstable["all_codes"].apply(
        lambda d: "; ".join(f"{k}({v})" for k, v in sorted(d.items(), key=lambda x: -x[1]))
    )
    unstable = unstable.sort_values("agreement_pct")
    unstable.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[OK] Casos inestables: {output_path} ({len(unstable)} filas)")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Análisis en profundidad de 10 corridas ICD-11"
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Directorio con dataset-run{1..N}.csv",
    )
    parser.add_argument("--n-runs", type=int, default=10)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    print("=" * 76)
    print("  DEEP ANALYSIS — 10 corridas de etiquetado ICD-11")
    print("=" * 76)

    # ── PARTE 1 ──
    print("\n── PARTE 1: Carga y validación ──")
    runs = load_runs(args.runs_dir, args.n_runs)
    print(f"  Cargadas {len(runs)} corridas desde {args.runs_dir}")

    val_warnings = validate_runs(runs)
    enc_warnings = validate_encoding(runs)

    if val_warnings:
        for w in val_warnings:
            print(f"  [WARN] {w}")
    else:
        print("  ✓ Todas las corridas: mismas 201 filas, mismas columnas")

    if enc_warnings:
        for w in enc_warnings:
            print(f"  [WARN] {w}")
    else:
        print("  ✓ Encoding consistente entre corridas")

    # ── PARTE 2 ──
    all_metrics = {}
    for i, df in runs.items():
        all_metrics[i] = compute_run_metrics(df)
    agg = aggregate_metrics(all_metrics)
    print_part2(all_metrics, agg)

    # ── PARTE 3 ──
    code_matrix = build_code_matrix(runs)
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
        for cat in ["estable_consistente", "estable_inconsistente",
                     "inestable", "NF_persistente", "NF_intermitente"]
    }

    # ── PARTE 4 ──
    judge_variance_df = find_judge_variance(
        code_matrix, consistency_matrix, runs
    )
    print_part4(judge_variance_df)

    # ── PARTE 5 ──
    match_dist = match_type_distribution(runs)
    print_part5(match_dist)

    # ── PARTE 6 ──
    nf_analysis = analyze_nf_drift(code_matrix)
    print_part6(nf_analysis)

    # ── Advertencias ──
    warnings = collect_warnings(
        val_warnings, enc_warnings,
        stability_results, nf_analysis,
        judge_variance_df, match_dist,
    )
    print_warnings(warnings)

    # ── Exportar ──
    judge_records = (
        judge_variance_df.to_dict(orient="records")
        if len(judge_variance_df) > 0
        else []
    )

    export_json(
        script_dir / "deep_analysis_report.json",
        all_metrics, agg, stability_summary, stability_results,
        judge_records, match_dist, nf_analysis, warnings,
    )
    export_unstable_csv(script_dir / "unstable_cases.csv", stability_df)


if __name__ == "__main__":
    main()
