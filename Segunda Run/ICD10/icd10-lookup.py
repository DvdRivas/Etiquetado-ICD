"""
icd10-lookup.py
---------------
REVERSE process of icd-experiment.py (ICD-10).

Reads the `icd10_code` column of the CSV and resolves the official title of
each code, which is written to the `icd10_lookup` column.

Then it asks an LM Studio model whether that title is consistent with the
original diagnosis, and writes "yes" or "no" to the `consistency` column.

PHASE 1 — Reverse lookup (deterministic, offline, no LLM):
  1. NF-XXXX codes  -> resolved against nf_dictionary.json (they are not ICD-10)
  2. ICD-10 codes   -> the WHO ClaML catalogue (icd102019en.xml)

PHASE 2 — Consistency evaluation (LLM):
  Asks "is the lookup consistent with the diagnosis?" -> yes / no
  - Lenient criterion: the title will always be broader than the diagnosis
    because codes were truncated to category level. Correctly encompassing
    the diagnosis counts as "yes".
  - Rows holding an NF code are skipped by default (their lookup is the
    diagnosis itself, so they would always score "yes"). Use --eval-nf to
    include them.
  - Rows whose code resolved to no title are marked "no".

This file replaces the old `icd11-lookup.py` of this folder, which was a
literal copy of the ICD-11 script (same WHO release/11 endpoints,
icd11_code/icd11_lookup columns). When reading a CSV that still carries
those legacy columns, they are migrated automatically to
icd10_code/icd10_lookup (see `_migrate_legacy_columns`).

Input/output CSV (same file, overwritten in place):
    filename | diagnosis_en | diagnosis_es | clinical_summary
            | icd10_code | icd_code_complete | category | chapter
            | hierarchical_distance | icd10_lookup | consistency

This script does not touch icd_code_complete / category / chapter /
hierarchical_distance / hierarchy_path / mapping_relation (icd-experiment.py
fills them); it only preserves them when rewriting the CSV.

Iterative execution (dataset-run1.csv .. dataset-run10.csv):
  When --csv is omitted the script processes every dataset-run{1..10}.csv
  found next to it, reusing the same catalogue and loaded model across all
  10 runs. Pass --csv to process a single file (one-off / test mode).

Dependencies:
    pip install rapidfuzz tqdm lmstudio

Usage:
    python icd10-lookup.py                      # runs all 10 runs
    python icd10-lookup.py --csv "dataset-run1.csv" --rows 5   # test

    # LM Studio on another machine of the local network
    python icd10-lookup.py --lms-host 192.168.1.50:1234

    # Reverse lookup only, without evaluating consistency
    python icd10-lookup.py --skip-consistency

    # Redo rows already processed
    python icd10-lookup.py --overwrite

LM Studio server: env LMS_HOST or --lms-host (default localhost:1234).

No WHO credentials and no network are needed. The ICD-10 API has no
/codeinfo (404) and answers in English only, so titles are read from the
same ClaML catalogue that icd-experiment.py used to assign the codes — which
also guarantees the reverse lookup agrees with the forward pass. Titles are
in English, like the whole pipeline; diagnosis_es is preserved as source
data but never read. See `ICD10Resolver`.
"""

import os
import csv
import json
import re
import argparse
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import lmstudio as lms
from tqdm import tqdm

from icd10_index import ICD10Index, CLAML_FILENAME

from run_logger import RunLogger

# Kept separate from the icd-experiment.py log: the two phases run
# independently, and mixing them would make it hard to tell which
# produced what.
DEFAULT_LOG_NAME = "lookup-runs.log"

# Sampling settings sent with every consistency call.
#
# temperature=0 is greedy decoding: the argmax token is taken at each step
# and the random generator is never consulted. It is fixed here rather than
# inherited from whatever the LM Studio UI happens to have set, so the judge
# is as documented as the assigner in icd-experiment.py.
#
# A per-request seed does not exist in the SDK; `seed` lives in
# LlmLoadModelConfig and only applies when the model is LOADED. It is passed
# anyway as cheap insurance should temperature ever be raised.
#
# Note for interpreting the results: at temperature 0 any difference between
# runs is NOT sampling variance. It comes from non-determinism in the
# inference engine — continuous batching, floating point non-associativity on
# GPU, KV cache reuse — which no seed can remove.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED        = 42

PREDICTION_CONFIG = {"temperature": DEFAULT_TEMPERATURE}



# ─────────────────────────────────────────────────────────────────────────────
# 1. ROBUST CSV READING
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = ["diagnosis_es", "diagnosis_en", "clinical_summary",
                    "icd10_code", "icd_code_complete", "category", "chapter",
                    "hierarchical_distance", "hierarchy_path",
                    "mapping_relation",
                    "icd10_lookup", "consistency"]

# Columns inherited from when this folder held a literal copy of the ICD-11
# script.
_LEGACY_COLUMN_MAP = {
    "icd11_code":   "icd10_code",
    "icd11_lookup": "icd10_lookup",
}


def _migrate_legacy_columns(rows: list, fieldnames: list) -> tuple[list, list]:
    renamed_any = False
    new_fieldnames = []
    for col in fieldnames:
        new_col = _LEGACY_COLUMN_MAP.get(col, col)
        if new_col != col:
            renamed_any = True
        if new_col not in new_fieldnames:
            new_fieldnames.append(new_col)

    if renamed_any:
        print(f"[CSV] Legacy ICD-11 columns migrated: "
              f"{ {k: v for k, v in _LEGACY_COLUMN_MAP.items() if k in fieldnames} }")
        for row in rows:
            for old_col, new_col in _LEGACY_COLUMN_MAP.items():
                if old_col in row:
                    row[new_col] = row.pop(old_col)

    return rows, new_fieldnames


def read_csv_robust(path: str) -> tuple[list, list]:
    """
    Read the CSV tolerating mixed encodings (UTF-8 rows and cp1252 rows in
    the same file). On rewrite everything is normalized to UTF-8.

    Returns (rows, fieldnames) preserving the original order and migrating
    legacy icd11_* -> icd10_* columns.
    """
    with open(path, "rb") as f:
        raw = f.read()

    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]

    fallbacks = 0
    decoded   = []
    for line in raw.split(b"\n"):
        try:
            decoded.append(line.decode("utf-8"))
        except UnicodeDecodeError:
            decoded.append(line.decode("cp1252", errors="replace"))
            fallbacks += 1

    if fallbacks:
        print(f"[CSV] Mixed encoding: {fallbacks} lines decoded as cp1252. "
              f"Everything will be normalized to UTF-8 on save.")

    reader     = csv.DictReader(decoded)
    rows       = list(reader)
    fieldnames = list(reader.fieldnames or [])

    rows, fieldnames = _migrate_legacy_columns(rows, fieldnames)

    for col in REQUIRED_COLUMNS:
        if col not in fieldnames:
            fieldnames.append(col)

    return rows, fieldnames


# ─────────────────────────────────────────────────────────────────────────────
# 2. NF DICTIONARY
# ─────────────────────────────────────────────────────────────────────────────

NF_DICT_PATH = Path(__file__).parent / "nf_dictionary.json"


def load_nf_labels() -> dict:
    """
    Load nf_dictionary.json and invert it to { "NF-0001": "label", ... } so
    an NF code can be resolved back to its original text.
    """
    if not NF_DICT_PATH.exists():
        return {}
    with open(NF_DICT_PATH, encoding="utf-8") as f:
        nf_dict = json.load(f)
    return {
        entry["code"]: entry.get("label", "")
        for entry in nf_dict.values()
        if entry.get("code")
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. ICD-10 REVERSE LOOKUP (offline, from the ClaML catalogue)
# ─────────────────────────────────────────────────────────────────────────────

class ICD10Resolver:
    """
    Resolves ICD-10 code -> official title, entirely offline.

    Everything comes from the same ClaML catalogue that icd-experiment.py
    used to assign the codes, so the reverse lookup is guaranteed to agree
    with the forward pass and needs no network at all.

    This replaces the previous WHO API client. That API has no /codeinfo
    (404) and answers in English only, so it required resolving one code at
    a time over the network for no added value.

    Titles prefer the long form when ClaML provides one: `preferredLong`
    carries the parent context ("Malignant neoplasm: Liver cell carcinoma"
    instead of just "Liver cell carcinoma"), which is what the consistency
    auditor needs to judge whether the category encompasses the diagnosis.
    """

    def __init__(self, index: "ICD10Index | None"):
        self.index = index

    @property
    def available(self) -> bool:
        return self.index is not None and len(self.index) > 0

    def lookup(self, code: str) -> str:
        """Official title for a code, or "" when it is not in the catalogue."""
        if not self.available:
            return ""
        return self.index.title(code)


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONSISTENCY EVALUATION (LLM)
# ─────────────────────────────────────────────────────────────────────────────

CONSISTENCY_SYSTEM_PROMPT = """You are a clinical coding auditor.

You receive a clinical DIAGNOSIS and the OFFICIAL TITLE of the ICD-10
category assigned to it. Answer a single question:

    Is the category title consistent with the diagnosis?

Important context: the codes were truncated to category level, so the title
will always be BROADER than the diagnosis. That is NOT an error.

Answer "yes" when:
- The category correctly encompasses the diagnosis, even if it is broader.
  E.g. diagnosis "AL amyloidosis" / title "Amyloidosis" -> yes
  E.g. diagnosis "Hepatocellular carcinoma" / title "Malignant neoplasm of
      liver and intrahepatic bile ducts" -> yes
- The title is a synonym or terminological variant of the diagnosis.
- The title names the same clinical entity under another denomination.

Answer "no" when:
- The category belongs to a different system, organ or pathological process.
  E.g. diagnosis "Renal malakoplakia" / title "Diseases of oesophagus" -> no
- The category shares words with the diagnosis but designates another entity.
  E.g. diagnosis "Q fever" / title "Yellow fever" -> no
- The title is so unspecific that it carries no real classification.
  E.g. title "Other specified conditions" or "Other disorders" -> no

Answer with ONE word only: yes or no.
No quotes, no punctuation, no explanations."""


async def check_consistency(diagnosis: str, lookup: str, model) -> str:
    """
    Ask the LLM whether the lookup is consistent with the diagnosis.
    Returns "yes" or "no". On error it returns "" (row left unevaluated).
    """
    chat = lms.Chat(CONSISTENCY_SYSTEM_PROMPT)
    chat.add_user_message(
        f'DIAGNOSIS: "{diagnosis}"\n'
        f'ICD-10 TITLE: "{lookup}"\n\n'
        f'Is the title consistent with the diagnosis? Answer yes or no.'
    )

    raw = ""
    try:
        result = await model.respond(chat, config=PREDICTION_CONFIG)
        raw    = _parse_channel_response(result.content).strip().lower()
        raw    = re.sub(r"[^a-z]", "", raw)

        if raw.startswith("yes"):
            return "yes"
        if raw.startswith("no"):
            return "no"

        print(f"\n    [LLM-consistency] Unexpected answer: {raw[:60]!r}")
        return ""
    except Exception as e:
        print(f"\n    [LLM-consistency] Error: {e} | raw: {raw[:100]}")
        return ""


def _parse_channel_response(raw: str) -> str:
    """Extract the final channel when the model uses the channel format."""
    match = re.search(
        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|\Z)",
        raw, re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    if "<|channel|>" in raw:
        parts = raw.split("<|end|>")
        last  = parts[-1].strip()
        if last:
            last = re.sub(r"<\|[^|]+\|>", "", last).strip()
            if last:
                return last
    return raw.strip()


async def run_consistency_phase(
    rows:    list,
    model,
    eval_nf: bool,
    batch:   int = 10,
) -> dict:
    """
    Evaluate diagnosis vs icd10_lookup consistency on the given rows.
    Writes the `consistency` column in place. Returns statistics.

    Receives an already connected model (reused across the 10 runs).
    """
    stats = {"yes": 0, "no": 0, "skipped_nf": 0, "no_lookup": 0, "error": 0}

    pending = []
    for row in rows:
        code   = (row.get("icd10_code")   or "").strip()
        lookup = (row.get("icd10_lookup") or "").strip()

        # NF codes: there was no real ICD-10 labelling to audit. Their lookup
        # is the diagnosis itself, so comparing them would always yield "yes"
        # and inflate the metric.
        if code.startswith("NF-") and not eval_nf:
            row["consistency"] = ""
            stats["skipped_nf"] += 1
            continue

        # A code that resolved to no entity cannot be consistent.
        if code and not lookup:
            row["consistency"] = "no"
            stats["no_lookup"] += 1
            continue

        if not lookup or not (row.get("diagnosis_en") or "").strip():
            row["consistency"] = ""
            continue

        pending.append(row)

    if not pending:
        print("[LLM] No rows to evaluate.\n")
        return stats

    with tqdm(total=len(pending), desc="Evaluating consistency", unit="row") as pbar:
        for i in range(0, len(pending), batch):
            chunk    = pending[i:i + batch]
            verdicts = await asyncio.gather(*[
                check_consistency(
                    (r.get("diagnosis_en") or "").strip(),
                    (r.get("icd10_lookup") or "").strip(),
                    model,
                )
                for r in chunk
            ])
            for row, verdict in zip(chunk, verdicts):
                row["consistency"] = verdict
                if verdict in ("yes", "no"):
                    stats[verdict] += 1
                else:
                    stats["error"] += 1
            pbar.update(len(chunk))

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 5. PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def resolve_row(row: dict, who: ICD10Resolver, nf_labels: dict) -> str:
    """
    Determine the icd10_lookup value for a row.
    Returns the outcome category, for statistics.
    """
    code = (row.get("icd10_code") or "").strip()

    if not code:
        row["icd10_lookup"] = ""
        return "no_code"

    # NF codes do not exist in ICD-10: they are resolved locally
    if code.startswith("NF-"):
        label = nf_labels.get(code, "")
        row["icd10_lookup"] = f"[NF] {label}" if label else "[NF] no label"
        return "nf"

    title = who.lookup(code)
    if title:
        row["icd10_lookup"] = title
        return "resolved"

    row["icd10_lookup"] = ""
    return "unresolved"


async def run_for_csv(csv_path: str, who: ICD10Resolver, overwrite: bool,
                      max_rows: int | None, workers: int, model,
                      skip_consistency: bool, eval_nf: bool):
    rows_in, fieldnames = read_csv_robust(csv_path)

    if not rows_in:
        print(f"[CSV] {csv_path} has no data rows.\n")
        return

    nf_labels = load_nf_labels()
    print(f"[NF]  Dictionary loaded: {len(nf_labels)} codes\n")

    # Row selection
    candidates = rows_in if max_rows is None else rows_in[:max_rows]
    if overwrite:
        pending = [r for r in candidates if (r.get("icd10_code") or "").strip()]
    else:
        pending = [
            r for r in candidates
            if (r.get("icd10_code") or "").strip()
            and not (r.get("icd10_lookup") or "").strip()
        ]

    total_with_code = sum(
        1 for r in candidates if (r.get("icd10_code") or "").strip()
    )
    print(f"[CSV] {len(rows_in)} rows | {total_with_code} with icd10_code | "
          f"{len(pending)} to resolve")
    if not overwrite and total_with_code > len(pending):
        print(f"[CSV] {total_with_code - len(pending)} already had "
              f"icd10_lookup (use --overwrite to redo them)")
    print()

    # ── PHASE 1: reverse lookup code -> title ────────────────────────────
    stats = {"resolved": 0, "nf": 0, "unresolved": 0, "no_code": 0}

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(tqdm(
                pool.map(lambda r: resolve_row(r, who, nf_labels), pending),
                total=len(pending),
                desc="Resolving codes",
                unit="row",
            ))
        for outcome in results:
            stats[outcome] = stats.get(outcome, 0) + 1
    else:
        print("[CSV] Reverse lookup: nothing to do.\n")

    # ── PHASE 2: LLM consistency evaluation ──────────────────────────────
    cstats = None
    if not skip_consistency:
        if overwrite:
            to_eval = [r for r in candidates if (r.get("icd10_code") or "").strip()]
        else:
            to_eval = [
                r for r in candidates
                if (r.get("icd10_code") or "").strip()
                and not (r.get("consistency") or "").strip()
            ]

        if to_eval:
            print(f"\n[LLM] {len(to_eval)} rows to evaluate\n")
            cstats = await run_consistency_phase(to_eval, model, eval_nf)
        else:
            print("\n[LLM] Consistency: nothing to do "
                  "(use --overwrite to re-evaluate).\n")

    # ── Single write, preserving every original column ───────────────────
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_in:
            writer.writerow({k: (row.get(k) or "") for k in fieldnames})

    print(f"\n[CSV] Updated in place: {csv_path}\n")

    # ── Reports ──────────────────────────────────────────────────────────
    if pending:
        t = max(len(pending), 1)
        print("── Reverse lookup statistics ───────────────────────────")
        print(f"  Processed                  : {len(pending)}")
        print(f"  Resolved (catalogue)       : {stats['resolved']}    ({stats['resolved']/t*100:.1f}%)")
        print(f"  NF (local dictionary)      : {stats['nf']}          ({stats['nf']/t*100:.1f}%)")
        print(f"  Unresolved                 : {stats['unresolved']} ({stats['unresolved']/t*100:.1f}%)")
        print("────────────────────────────────────────────────────────\n")

        if stats["unresolved"]:
            print(f"[!] {stats['unresolved']} codes could not be resolved.")
            print("    Usually truncated codes that do not exist as an entity")
            print("    of their own in ICD-10, or malformed codes.\n")

    if cstats:
        evaluated = cstats["yes"] + cstats["no"]
        e = max(evaluated, 1)
        print("── Consistency diagnosis_en vs icd10_lookup ────────────")
        print(f"  Evaluated by the LLM       : {evaluated}")
        print(f"  Consistent    (yes)        : {cstats['yes']} ({cstats['yes']/e*100:.1f}%)")
        print(f"  Inconsistent  (no)         : {cstats['no']}  ({cstats['no']/e*100:.1f}%)")
        if cstats["no_lookup"]:
            print(f"  Marked \"no\", no lookup     : {cstats['no_lookup']}")
        if cstats["skipped_nf"]:
            print(f"  Skipped for being NF       : {cstats['skipped_nf']} "
                  f"(use --eval-nf to include them)")
        if cstats["error"]:
            print(f"  LLM errors                 : {cstats['error']}")
        print("────────────────────────────────────────────────────────\n")


async def run_all(csv_paths: list[str], who: ICD10Resolver, overwrite: bool,
                  max_rows: int | None, workers: int,
                  model_name: str, lms_host: str,
                  skip_consistency: bool, eval_nf: bool,
                  logger: RunLogger):
    """
    Run run_for_csv over several CSVs, reusing a single LLM client.

    Unlike icd-experiment.py there is nothing to reset between runs: the
    consistency judge keeps no per-diagnosis cache, so every row really does
    call the LLM on every run. The only cache here maps code -> official
    title, which is deterministic reference data and cannot couple runs.
    """
    if skip_consistency:
        for index, csv_path in enumerate(csv_paths, 1):
            logger.run_header(index, len(csv_paths), csv_path)
            await run_for_csv(csv_path, who, overwrite, max_rows, workers,
                              None, skip_consistency, eval_nf)
        return

    async with lms.AsyncClient(lms_host) as client:
        print(f"[LLM] Host: {lms_host}")
        print(f"[LLM] Connecting to model: {model_name}")
        try:
            model = await client.llm.model(model_name)
        except Exception as e:
            print(f"[LLM] ERROR connecting to {lms_host}: {e}")
            print("[LLM] Check that LM Studio is serving on that IP:port")
            print("      (Settings > Developer > Serve on Local Network).\n")
            return
        print("[LLM] Connected\n")

        for index, csv_path in enumerate(csv_paths, 1):
            logger.run_header(index, len(csv_paths), csv_path)
            await run_for_csv(csv_path, who, overwrite, max_rows, workers,
                              model, skip_consistency, eval_nf)


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
N_RUNS     = 10


def _default_csv_paths() -> list[str]:
    """dataset-run1.csv .. dataset-run{N_RUNS}.csv found next to the script."""
    paths = [SCRIPT_DIR / f"dataset-run{i}.csv" for i in range(1, N_RUNS + 1)]
    existing = [str(p) for p in paths if p.exists()]
    missing  = [p.name for p in paths if not p.exists()]
    if missing:
        print(f"[CSV] Notice: not found {missing}")
    return existing


def main():
    parser = argparse.ArgumentParser(
        description="ICD-10 reverse lookup: code -> official title. Without "
                    "--csv it processes dataset-run1.csv .. dataset-run10.csv."
    )
    parser.add_argument("--csv", default=None,
                        help="Single CSV to process in place. If omitted, "
                             "processes dataset-run1.csv .. dataset-run10.csv "
                             "next to the script.")
    parser.add_argument("--claml", default=None,
                        help=f"Path to the WHO ClaML XML ({CLAML_FILENAME}). If "
                             f"omitted it is searched next to the script and in "
                             f"its parent folders.")
    # No --lang: the catalogue is English, like the whole pipeline.
    parser.add_argument("--workers", type=int, default=8,
                        help="Threads used to resolve codes (default: 8)")
    parser.add_argument("--model", default="medgemma-27b-it",
                        help="LM Studio model used to evaluate consistency")
    parser.add_argument("--lms-host",
                        default=os.environ.get("LMS_HOST", "10.8.0.45:1234"),
                        help="LM Studio server host:port (or env LMS_HOST). "
                             "Default: localhost:1234")
    parser.add_argument("--skip-consistency", action="store_true",
                        help="Only do the reverse lookup, without evaluating consistency")
    parser.add_argument("--eval-nf", action="store_true",
                        help="Also evaluate rows holding an NF code "
                             "(skipped by default: their lookup is the "
                             "diagnosis itself and would always score \"yes\")")
    parser.add_argument("--overwrite", action="store_true",
                        help="Redo rows that already have icd10_lookup/consistency")
    parser.add_argument("--rows", type=int, default=None,
                        help="Process only the first N rows of each CSV (test mode)")
    parser.add_argument("--log", default=None,
                        help="Session log file (default: lookup-runs.log next "
                             "to the script). Appended, never truncated. Kept "
                             "separate from the icd-experiment.py log because "
                             "the two phases run independently.")
    parser.add_argument("--no-log", action="store_true",
                        help="Do not write the session log")
    args = parser.parse_args()

    if args.csv:
        if not Path(args.csv).exists():
            print(f"[ERROR] File not found --csv: {args.csv}")
            return
        csv_paths = [args.csv]
    else:
        csv_paths = _default_csv_paths()
        if not csv_paths:
            print(f"[ERROR] No dataset-run{{1..{N_RUNS}}}.csv found in {SCRIPT_DIR}")
            return
        print(f"[CSV] Iterative execution: {len(csv_paths)} runs found\n")

    index = ICD10Index.load(args.claml)
    if index is None:
        print(f"[ERROR] {CLAML_FILENAME} not found next to the script nor in "
              f"its parent folders. There is no way to resolve the codes.")
        return
    print(f"[ICD-10] Catalogue: {len(index)} classes | version {index.version}")
    print(f"[ICD-10] Source: {index.source}\n")

    who = ICD10Resolver(index)

    log_path = args.log or (SCRIPT_DIR / DEFAULT_LOG_NAME)
    with RunLogger(log_path, enabled=not args.no_log) as logger:
        logger.session_header({
            "phase":            "reverse lookup + consistency",
            "icd version":      "ICD-10",
            "model":            args.model if not args.skip_consistency else "(skipped)",
            "lms host":         args.lms_host,
            "temperature":      DEFAULT_TEMPERATURE,
            "seed (load-time)": DEFAULT_SEED,
            "runs":             len(csv_paths),
            "rows per run":     args.rows if args.rows is not None else "all",
            "workers":          args.workers,
            "overwrite":        args.overwrite,
            "eval NF rows":     args.eval_nf,
            "catalogue":        f"{len(index)} classes, version {index.version}",
            "catalogue source": index.source,
        })
        asyncio.run(run_all(csv_paths, who, args.overwrite, args.rows, args.workers,
                            args.model, args.lms_host, args.skip_consistency,
                            args.eval_nf, logger))


if __name__ == "__main__":
    main()
