"""
icd-experiment.py
-----------------
Assigns ICD-11 codes (complete code + category + chapter) to clinical
diagnoses using:
  1. Lexical index from the Orphanet product1 JSON — names + synonyms → ICD-11
  2. WHO ICD-11 API /search                       — candidates validated semantically by an LLM
  3. NF dictionary                                — last resort (sequential NF-XXXX)

Matching cascade (entirely in ENGLISH, over diagnosis_en):
    0. Disease        — the LLM answers "which disease is this?" and gives
                        its standard name (every row)
    1. Cache          — avoids recomputing repeated diagnoses
    2. Exact          — lexical index, complete phrase and then disease name
    3. Fuzzy          — lexical index >=85, ONE candidate the LLM validates
    4. WHO /search    — BOTH queries, wide pool, the LLM chooses one or none
    5. NF             — not found, recorded in nf_dictionary.json

Complete phrase vs disease name:
  Diagnoses carry the cause and the context attached to them ("Acute
  intermittent porphyria due to HMBS mutation"), which blocks the match even
  when the entity does exist in the ontology. Identifying the disease
  ("Acute intermittent porphyria") widens RECALL.

  This is not text trimming but clinical identification: the name may use
  vocabulary absent from the input ("Cryptococcus neoformans
  meningoencephalitis" -> "Cryptococcosis").

  Semantic validation is ALWAYS done against the complete phrase, because
  the cause can change the assigned ICD-11 category (drug-induced diabetes
  vs type 2 diabetes). The complete phrase is always tried first.

Names, apostrophes and brackets:
  Before comparing any name (diagnosis, lexical index entry or synonym) it
  is normalized: lowercase, dashes -> space, collapsed whitespace, ALL
  apostrophe variants removed, and brackets turned into separators. Without
  the apostrophe rule, "Parkinson's disease" written with a curly quote and
  the same string with a straight quote — or "Parkinsons disease" — do not
  match each other. Without the bracket rule, eponyms written as
  "[Takayasu]" never match the token "takayasu". See `_normalize`.

Dagger/asterisk codes:
  Orphanet writes some references with the dual-classification marker
  ("A39.1+", "B00.4*"). The classification stores them unmarked, so they are
  stripped on read (`strip_dagger`); otherwise they silently fail to match.

Multiple codes (DisorderMappingRelation):
  A single Orphanet disorder may carry more than one ICD-11 code
  (ExternalReferenceList with several Source=ICD-11 entries), each with its
  own mapping relation (DisorderMappingRelation):
    E    - Exact mapping: the disorder and the code are equivalent.
    NTBT - The disorder is NARROWER than the code (the code is broader).
    BTNT - The disorder is BROADER than the code (the code is a more
           specific subtype).
    ND   - Relation not decided.
  When the lexical index resolves a diagnosis to a disorder holding more
  than one code, the candidate codes and their mapping relations are handed
  to the LLM (gemma/medgemma) together with the diagnosis (diagnosis_en) so
  it can pick the clinically most adequate code. See `disambiguate_icd_code`.

Why English everywhere:
  - ICD-11 is authored in English; translations index fewer terms and
    synonyms, both in the API and in Orphanet.
  - en_product1.json covers 11645 disorders with 6553 mapped to ICD-11,
    against 11456/6143 in the Spanish release.
  - The whole pipeline (prompts, WHO lookups, NF labels) runs in English so
    that the ICD-10 and ICD-11 folders behave identically and results stay
    comparable.
  - The diagnosis_es column is preserved as source data but NO pipeline step
    reads it.

Code hierarchy (icd_code_complete -> category -> chapter):
  Instead of aggressively truncating the code at the dot, the real code
  hierarchy is walked upwards:
    icd_code_complete     — code exactly as the source returned it (lexical
                            index, WHO /search or NF), untruncated.
    category              — first hierarchy level after the last block
                            ("5A11.2" -> "5A11"; "BA00.0Z" -> "BA00").
    chapter               — ICD-11 chapter that category belongs to, obtained
                            by climbing the entity "parent" links in the WHO
                            API until reaching classKind="chapter".
    hierarchical_distance — how many levels sit between the chapter and the
                            category (depth of the category inside its
                            chapter).
  `icd11_code` is kept for backwards compatibility and equals `category`.

Input/output CSV (same file, overwritten in place):
    filename | diagnosis_en | diagnosis_es | core_diagnosis | clinical_summary
            | icd11_code | icd_code_complete | category | chapter
            | hierarchical_distance | icd11_lookup | match_type
            | match_source | consistency

Columns written by this script:
    icd11_code            — same as `category` (backwards compatibility)
    icd_code_complete     — untruncated code
    category              — code up to category level (hierarchical walk, not
                            a text cut)
    chapter               — ICD-11 chapter of that category
    hierarchical_distance — levels between chapter and category
    core_diagnosis        — disease name identified by the LLM (equal to
                            diagnosis_en when it already was the standard name)
    match_type            — cascade step that resolved the row
    match_source          — "complete" | "core" | "none"
                            ("core" = resolved thanks to the disease name)

This script only fills the columns above. The reverse lookup
(`icd11_lookup`) is implemented separately in icd11-lookup.py.

Columns describing the code hierarchy:
    icd_code_complete     — code as the source returned it, untruncated
    category              — code truncated to category level
    chapter               — chapter that category belongs to
    hierarchical_distance — levels between chapter and category
    hierarchy_path        — the chain that produces that distance, from
                            chapter down, tagged with each level's kind:
                            "II[chapter] > C00-C97[block] > C00-C75[block] >
                             C15-C26[block] > C22[category]"
                            Element count is always distance + 1, which makes
                            the number auditable and shows WHY two categories
                            sit at different depths. Empty for NF codes.
    mapping_relation      — Orphanet's DisorderMappingRelation for the chosen
                            code (E / NTBT / BTNT / ND). Informative only: it
                            never changes the assignment. EMPTY means "not
                            applicable", not "missing": only lexical-index
                            matches (exact_* and fuzzy_*) come from a curated
                            mapping. Catalogue and NF rows leave it blank.

Iterative execution (dataset-run1.csv .. dataset-run10.csv):
  When --csv is omitted the script processes every dataset-run{1..10}.csv
  found next to it, reusing the same lexical index, WHO client and loaded
  model across all 10 runs. Pass --csv to process a single file (one-off /
  test mode).

Location of en_product1.json:
  Neither --product1 nor a particular working directory is required. The
  file is searched starting from the SCRIPT folder and climbing up to 3
  levels, checking at each level the directory itself and its direct
  subfolders. That works both with a local copy next to the script (which
  wins) and with the shared repository copy living in a sibling folder. See
  `find_product1`.

Dependencies:
    pip install lmstudio rapidfuzz tqdm requests

Usage:
    python icd-experiment.py                       # runs all 10 runs
    python icd-experiment.py --csv "dataset-run1.csv" --rows 5   # test

    # LM Studio server on another machine of the local network
    python icd-experiment.py --lms-host 192.168.1.50:1234

LM Studio server: localhost:1234 by default. To use another machine pass
--lms-host IP:PORT or set the LMS_HOST environment variable. On the server
machine network access must be enabled in LM Studio:
Settings > Developer > "Serve on Local Network".

WHO credentials: read from the WHO_CLIENT_ID and WHO_CLIENT_SECRET
environment variables, or via --who-client-id / --who-client-secret.
"""

import os
import json
import csv
import re
import argparse
import asyncio
from pathlib import Path
from datetime import date

import requests
import urllib3
import lmstudio as lms
from rapidfuzz import process, fuzz
from tqdm import tqdm

from run_logger import RunLogger

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ICD_VERSION = "11"
ICD_SOURCE  = "ICD-11"          # Source string as it appears in ExternalReferenceList

# Sampling settings sent with every LLM call.
#
# temperature=0 is greedy decoding: the argmax token is taken at each step
# and the random generator is never consulted. It is fixed here rather than
# inherited from whatever the LM Studio UI happens to have set, so the runs
# stay comparable.
#
# A per-request seed does not exist in the SDK (LlmPredictionConfig exposes
# temperature, topK/topP/minP and repeatPenalty only). `seed` lives in
# LlmLoadModelConfig, i.e. it applies when the model is LOADED, and it is
# ignored if LM Studio already had the model in memory. It is passed anyway
# as cheap insurance should temperature ever be raised.
#
# Note for interpreting the results: at temperature 0 any difference between
# runs is NOT sampling variance. It comes from non-determinism in the
# inference engine — continuous batching, floating point non-associativity on
# GPU, KV cache reuse — which no seed can remove.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED        = 42

PREDICTION_CONFIG = {"temperature": DEFAULT_TEMPERATURE}



# ─────────────────────────────────────────────────────────────────────────────
# 1. UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

_APOSTROPHES_RE = re.compile(r"[‘’´`']")
_BRACKETS_RE    = re.compile(r"[\[\]()]")
_DAGGER_RE      = re.compile(r"[+*†]+$")


def _normalize(text: str) -> str:
    """
    Shared normalization for every comparison.

    Lowercase, apostrophes removed (so "Parkinson's", "Parkinson’s" and
    "Parkinsons" collapse), brackets turned into separators, dashes turned
    into spaces, whitespace collapsed.

    Brackets matter because classifications put eponyms and alternative
    names inside them — "Aortic arch syndrome [Takayasu]". Glued to the
    word, "[takayasu]" never matches the token "takayasu", so the entity
    becomes unreachable by name. Kept identical to icd10_index.normalize so
    both folders compare strings the same way.
    """
    text = text.lower().strip()
    text = _APOSTROPHES_RE.sub("", text)
    text = _BRACKETS_RE.sub(" ", text)
    text = re.sub(r"[\-–—]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_dagger(code: str) -> str:
    """
    Drop the dagger/asterisk marker of dual classification.

    Orphanet writes some of its references as "A39.1+" or "B00.4*", while
    the classification stores them unmarked, so without stripping they
    silently fail to match.
    """
    return _DAGGER_RE.sub("", (code or "").strip())


def truncate_code(code: str) -> str:
    """
    Extract the category of an ICD-11 code: everything left of the first dot
    (first level after the last block).

        "5A11.2"  -> "5A11"
        "BA00.0Z" -> "BA00"
        "1C62"    -> "1C62"
        "NF-0001" -> "NF-0001"   (NF codes are never truncated)
    """
    code = (code or "").strip()
    if not code or code.startswith("NF-"):
        return code
    return code.split(".", 1)[0]


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


def _clean_json_response(raw: str) -> str:
    """Strip ```json ... ``` markdown fences when the model adds them."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2. LEXICAL INDEX from the product1 JSON (ICD-11)
# ─────────────────────────────────────────────────────────────────────────────

def build_lexical_index(product1_path: str) -> dict:
    """
    Build the lexical index from Orphanet product1: a dictionary mapping a
    normalized name (canonical plus every synonym) to the disorder node.
    It is not a knowledge graph — there are no relations between nodes, only
    a name -> code lookup table — hence "lexical index" rather than "KG".

    Node structure:
        {
            "canonical_name": str,
            "orpha_code":     str,
            "icd_codes":      [{"code": str, "mapping_relation": str}, ...],
        }

    `mapping_relation` is the short code (E, NTBT, BTNT, ND) taken from
    DisorderMappingRelation — used to disambiguate when a disorder carries
    more than one ICD-11 code (see `disambiguate_icd_code`).
    """
    print(f"[Lexical index] Loading {product1_path} ...")

    with open(product1_path, encoding="utf-8") as f:
        data = json.load(f)

    disorders = (
        data.get("JDBOR", [{}])[0]
            .get("DisorderList", [{}])[0]
            .get("Disorder", [])
    )

    lexical_index = {}
    total    = 0
    with_icd = 0

    for disorder in disorders:
        total += 1
        orpha_code = disorder.get("OrphaCode", "")
        names      = disorder.get("Name", [])
        canonical  = _pick_label(names, preferred_lang="en")
        if not canonical:
            continue

        icd_codes = _extract_icd_codes(disorder)
        if icd_codes:
            with_icd += 1

        node = {
            "canonical_name": canonical,
            "orpha_code":     orpha_code,
            "icd_codes":      icd_codes,
        }

        # The canonical name takes precedence
        lexical_index[_normalize(canonical)] = node

        # Synonyms never overwrite the canonical entry
        for syn_block in disorder.get("SynonymList", []):
            for syn in syn_block.get("Synonym", []):
                label = syn.get("label", "").strip()
                if label:
                    key = _normalize(label)
                    if key not in lexical_index:
                        lexical_index[key] = node

    print(f"[Lexical index] {total} disorders | {with_icd} with {ICD_SOURCE} | "
          f"{len(lexical_index)} total entries\n")
    return lexical_index


def _pick_label(names: list, preferred_lang: str = "en") -> str:
    for n in names:
        if n.get("lang") == preferred_lang:
            return n["label"].strip()
    return names[0]["label"].strip() if names else ""


def _extract_relation_code(rel_list: list) -> str:
    """Extract the short code (E, NTBT, BTNT, ND) from DisorderMappingRelation."""
    if not rel_list:
        return "ND"
    label = rel_list[0].get("Name", [{}])[0].get("label", "")
    if not label:
        return "ND"
    return label.split()[0].strip()


def _extract_icd_codes(disorder: dict) -> list:
    """
    Extract the ICD-11 codes from ExternalReferenceList, each one with its
    mapping relation (DisorderMappingRelation). A disorder may hold more
    than one entry.
    """
    codes = []
    for ref_block in disorder.get("ExternalReferenceList", []):
        for ref in ref_block.get("ExternalReference", []):
            if ref.get("Source") != ICD_SOURCE:
                continue
            code = strip_dagger(ref.get("Reference", ""))
            if not code:
                continue
            relation = _extract_relation_code(ref.get("DisorderMappingRelation"))
            codes.append({"code": code, "mapping_relation": relation})
    return codes


# ─────────────────────────────────────────────────────────────────────────────
# 3. WHO ICD-11 API
# ─────────────────────────────────────────────────────────────────────────────

class WHOClient:
    """WHO ICD-11 API client with OAuth2 token handling."""

    TOKEN_URL    = "https://icdaccessmanagement.who.int/connect/token"
    BASE_URL     = "https://id.who.int/icd/release/11/2024-01/mms"
    SEARCH_URL   = f"{BASE_URL}/search"
    CODEINFO_URL = f"{BASE_URL}/codeinfo"

    # Every request is made in English: ICD-11 is authored in English and its
    # translations index fewer terms, so results stay richer and comparable
    # with the ICD-10 folder, which only supports English at all.
    LANG = "en"

    def __init__(self, client_id: str, client_secret: str):
        self.client_id     = client_id
        self.client_secret = client_secret
        self.token         = None

    def authenticate(self) -> bool:
        if not self.client_id or not self.client_secret:
            print("[WHO] No credentials — WHO API disabled\n")
            return False
        try:
            r = requests.post(
                self.TOKEN_URL,
                data={
                    "client_id":     self.client_id,
                    "client_secret": self.client_secret,
                    "scope":         "icdapi_access",
                    "grant_type":    "client_credentials",
                },
                verify=False,
                timeout=15,
            )
            r.raise_for_status()
            self.token = r.json()["access_token"]
            print("[WHO] Token obtained successfully\n")
            return True
        except Exception as e:
            print(f"[WHO] Authentication error: {e}\n")
            return False

    def _headers(self) -> dict:
        return {
            "Authorization":   f"Bearer {self.token}",
            "Accept":          "application/json",
            "Accept-Language": self.LANG,
            "API-Version":     "v2",
        }

    @staticmethod
    def _strip_highlight(text: str) -> str:
        """The API marks matches with <em> tags; they are stripped."""
        return re.sub(r"</?em[^>]*>", "", text or "").strip()

    def search(self, text: str, limit: int = 5) -> list:
        """
        Search ICD-11 candidates for a free-text diagnosis.
        Returns [{"icd11_code": str, "title": str}, ...] (at most `limit`).
        """
        if not self.token or not text.strip():
            return []

        params = {
            "q":                       text,
            "useFlexisearch":          "true",
            "flatResults":             "true",
            "highlightingEnabled":     "false",
            "includeKeywordResult":    "false",
        }

        try:
            r = requests.get(
                self.SEARCH_URL,
                headers=self._headers(),
                params=params,
                verify=False,
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"\n    [WHO-search] Error: {e}")
            return []

        candidates = []
        seen       = set()
        for entity in (data.get("destinationEntities") or []):
            code  = (entity.get("theCode") or "").strip()
            title = self._strip_highlight(entity.get("title", ""))
            if not code or not title or code in seen:
                continue
            seen.add(code)
            candidates.append({"icd11_code": code, "title": title})
            if len(candidates) >= limit:
                break

        return candidates

    # ── hierarchy: category -> chapter ────────────────────────────────────

    def _get_json(self, url: str):
        r = requests.get(url, headers=self._headers(), verify=False, timeout=15)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _entity_title(entity: dict) -> str:
        title = (entity or {}).get("title")
        if isinstance(title, dict):
            return title.get("@value", "")
        return title or ""

    def get_chapter_for_code(self, code: str) -> tuple[str, str, int | None, str]:
        """
        Climb the "parent" links of the entity behind `code` until reaching
        the classKind="chapter" entity. Returns
        (chapter_code, chapter_title, hierarchical_distance, hierarchy_path).

        `hierarchical_distance` is the number of parent hops walked from the
        category up to the chapter. `hierarchy_path` records the chain that
        produced it, written from chapter down and tagged with each level's
        classKind:

            "01[chapter] > BlockL1-INF[block] > 1C1G[category]"

        Keeping the chain makes the distance auditable rather than an
        unexplained number — the element count is always distance + 1 — and
        it is collected here because the walk already visits every step; a
        second pass would mean paying the API twice.

        On any API failure it returns ("", "", None, ""): the caller decides
        how to degrade (empty column, never a crash).
        """
        info = self._get_json(f"{self.CODEINFO_URL}/{code}")
        if not info:
            return "", "", None, ""

        stem_id = info.get("stemId", "")
        if not stem_id:
            return "", "", None, ""

        current = self._get_json(stem_id)
        if not current:
            return "", "", None, ""

        def _step(entity: dict) -> str:
            """
            Label one level of the chain.

            Chapters and categories carry `code` ("02", "2C12"), but ICD-11
            BLOCKS do not: their `code` is an empty string and the identifier
            lives in `codeRange` ("2B70-2C1Z") with `blockId` as a fallback
            ("BlockL3-2B7"). Reading only `code` rendered every block as "?".

            `codeRange` is the direct analogue of an ICD-10 block code
            (C00-C97), so preferring it also keeps the two folders' paths
            comparable.
            """
            kind = (entity.get("classKind") or "?").strip()
            label = (
                (entity.get("code") or "").strip()
                or (entity.get("codeRange") or "").strip()
                or (entity.get("blockId") or "").strip()
                or "?"
            )
            return f"{label}[{kind}]"

        chain    = [_step(current)]
        distance = 0
        guard    = 0
        while (current.get("classKind", "").lower() != "chapter"
               and current.get("parent") and guard < 10):
            parent_uri = current["parent"][0]
            nxt = self._get_json(parent_uri)
            if not nxt:
                break
            current   = nxt
            distance += 1
            guard    += 1
            chain.append(_step(current))

        chapter_code  = (current.get("code") or "").strip()
        chapter_title = self._entity_title(current)
        return chapter_code, chapter_title, distance, " > ".join(reversed(chain))


# ─────────────────────────────────────────────────────────────────────────────
# 4. PERSISTENT NF DICTIONARY
# ─────────────────────────────────────────────────────────────────────────────

NF_DICT_PATH = Path(__file__).parent / "nf_dictionary.json"


def load_nf_dictionary() -> dict:
    if NF_DICT_PATH.exists():
        with open(NF_DICT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_nf_dictionary(nf_dict: dict) -> None:
    with open(NF_DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(nf_dict, f, ensure_ascii=False, indent=2)


def get_or_create_nf_code(diagnosis: str, nf_dict: dict) -> str:
    """
    Return the existing NF code for the diagnosis, or mint a new one.
    Globally sequential format: NF-0001, NF-0002, ...

    Labels are stored in English (diagnosis_en), like the rest of the
    pipeline.
    """
    key = _normalize(diagnosis)
    if key in nf_dict:
        return nf_dict[key]["code"]

    # Next number: highest existing + 1 (robust against gaps)
    max_num = 0
    for entry in nf_dict.values():
        m = re.match(r"NF-(\d+)$", entry.get("code", ""))
        if m:
            max_num = max(max_num, int(m.group(1)))

    code = f"NF-{max_num + 1:04d}"
    nf_dict[key] = {
        "code":       code,
        "label":      diagnosis,
        "first_seen": date.today().isoformat(),
    }
    return code


# ─────────────────────────────────────────────────────────────────────────────
# 4b. PERSISTENT CHAPTER CACHE (category -> chapter)
# ─────────────────────────────────────────────────────────────────────────────

CHAPTER_CACHE_PATH = Path(__file__).parent / "chapter_cache.json"


def load_chapter_cache() -> dict:
    if CHAPTER_CACHE_PATH.exists():
        with open(CHAPTER_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_chapter_cache(cache: dict) -> None:
    with open(CHAPTER_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


async def resolve_chapter(who_client: WHOClient, category: str, cache: dict) -> tuple[str, str, str]:
    """
    Resolve (chapter, hierarchical_distance, hierarchy_path) for a category,
    backed by a persistent cache so the WHO API is not queried again for the
    same category across rows and across runs.

    Cache entries written before `hierarchy_path` existed lack that key, so
    they are treated as stale and recomputed. Caching this is safe: the
    classification hierarchy is deterministic, so unlike the per-diagnosis
    caches it introduces no run-to-run coupling.
    """
    category = (category or "").strip()
    if not category:
        return "", "", ""
    if category.startswith("NF-"):
        # NF codes are not part of the classification: no chain to walk.
        return "NF", "0", ""

    entry = cache.get(category)
    # Entries written before `hierarchy_path` existed lack the key; entries
    # written while blocks resolved to "?" are equally stale. Both are
    # recomputed rather than served.
    if (entry is not None and "hierarchy_path" in entry
            and "?[" not in entry.get("hierarchy_path", "")):
        return (entry.get("chapter_title", ""),
                str(entry.get("hierarchical_distance", "")),
                entry.get("hierarchy_path", ""))

    chapter_code, chapter_title, distance, path = "", "", None, ""
    if who_client.token:
        try:
            chapter_code, chapter_title, distance, path = await asyncio.to_thread(
                who_client.get_chapter_for_code, category
            )
        except Exception as e:
            print(f"\n    [WHO-chapter] {category}: {e}")

    cache[category] = {
        "chapter_code":          chapter_code,
        "chapter_title":         chapter_title,
        "hierarchical_distance": distance,
        "hierarchy_path":        path,
    }
    return chapter_title, ("" if distance is None else str(distance)), path


# ─────────────────────────────────────────────────────────────────────────────
# 5. PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

DISEASE_IDENTIFICATION_SYSTEM_PROMPT = """You are a clinician identifying diseases for medical classification.

You are given a clinical diagnosis as written in a case report. It usually
carries the causative agent, the clinical context and the comorbidities along
with the disease itself.

Answer one question: WHICH DISEASE IS THIS?

Reply with the standard name of that disease — the name under which it would
be listed in a medical classification such as ICD-11. Use your medical
knowledge to name the entity; do not merely delete words from the input. The
standard name often uses vocabulary that never appears in the input.

Requirements for your answer:
- It must be a NAMED DISEASE, never a bare symptom, sign or body part.
- It must be the disease actually present in the patient, not the trigger,
  the comorbidity or the setting.
- Prefer the most specific disease name that is a real classification entry.
  If the highly specific variant is not a classified entity, name the disease
  it belongs to.
- If the input is already the standard disease name, repeat it unchanged.

Answer with the disease name ONLY. No quotes, no explanation, no preamble.

════════ EXAMPLES ════════
Input:  Acute intermittent porphyria due to HMBS mutation
Answer: Acute intermittent porphyria

Input:  Cryptococcus neoformans meningoencephalitis
Answer: Cryptococcosis

Input:  Amoxicillin-induced rash in Epstein-Barr virus infectious mononucleosis
Answer: Drug eruption

Input:  Minocycline-induced hyperpigmentation type III
Answer: Drug-induced hyperpigmentation

Input:  Immune checkpoint inhibitor (pembrolizumab)-induced diabetes mellitus in metastatic melanoma
Answer: Drug-induced diabetes mellitus

Input:  Pediatric Takayasu arteritis
Answer: Takayasu arteritis

Input:  Vascular Ehlers-Danlos syndrome due to COL3A1 mutation
Answer: Vascular Ehlers-Danlos syndrome

Input:  Disseminated bartonellosis due to Bartonella henselae
Answer: Bartonellosis

Input:  Generalized argyria of unidentified origin
Answer: Argyria

Input:  Post-traumatic Morel-Lavallee lesion
Answer: Morel-Lavallee lesion

Input:  Emphysematous vertebral osteomyelitis due to ESBL-producing Klebsiella pneumoniae
Answer: Vertebral osteomyelitis

Input:  Free-floating iris pigment epithelial cyst in the vitreous
Answer: Iris cyst

Input:  Spur cell hemolytic anemia in advanced alcoholic cirrhosis
Answer: Acquired haemolytic anaemia

Input:  Jejunal variceal bleeding secondary to noncirrhotic portal hypertension
Answer: Bleeding intestinal varices

Input:  Ectopic intranasal tooth
Answer: Ectopic tooth

Input:  Pearly penile papules (normal anatomical variant)
Answer: Pearly penile papules

Input:  Secondary hemophagocytic lymphohistiocytosis due to COVID-19
Answer: Secondary haemophagocytic lymphohistiocytosis

Input:  Merkel cell carcinoma
Answer: Merkel cell carcinoma"""


SEMANTIC_MATCH_SYSTEM_PROMPT = """You are a specialist in medical terminology.
Your task is to determine whether any of the candidate terms from a medical
ontology is a clinical equivalent of the input diagnosis.

Equivalence rules:
- Accept general clinical equivalences: "Pulmonary tuberculosis" is equivalent to
  "Primary pulmonary tuberculosis" because in a clinical setting they refer to
  the same disease process.
- Accept when the candidate is a more specific form of the same concept.
- Reject (null) when the candidate is a clearly different concept, even if it
  shares words with the input.
- If several candidates are valid, choose the clinically most precise one.
- Answer ONLY with the exact candidate term, or null if none is equivalent.
- No quotes, no explanations, just the term or null.

Examples:
  Input: "Pulmonary tuberculosis"
  Candidates: ["Primary pulmonary tuberculosis", "Miliary tuberculosis", "Tuberculosis"]
  Answer: Primary pulmonary tuberculosis

  Input: "Idiopathic hemoptysis"
  Candidates: ["Idiopathic pulmonary hemosiderosis", "Neonatal pulmonary hemorrhage"]
  Answer: null

  Input: "Nontuberculous mycobacteria"
  Candidates: ["Pulmonary nontuberculous mycobacterial infection", "Multifocal tuberculosis"]
  Answer: Pulmonary nontuberculous mycobacterial infection"""


ICD_CODE_DISAMBIGUATION_SYSTEM_PROMPT = f"""You are a clinical coder choosing the single best ICD-{ICD_VERSION} code
for a diagnosed disease, when Orphanet maps that disease to more than one
ICD-{ICD_VERSION} code.

Each candidate code carries a mapping relation:
  E    - Exact mapping: the disease and the code are clinically equivalent.
  NTBT - The disease is NARROWER than the code (the code is broader/more
         general than the disease).
  BTNT - The disease is BROADER than the code (the code is a more specific
         subtype of the disease).
  ND   - The relation has not been decided.

Given the disease name and the full diagnosis (which may specify a subtype,
cause or context), pick the candidate code whose scope best matches the
diagnosis:
  - Prefer "E" when it is clinically adequate for the diagnosis.
  - If the diagnosis text points at a specific subtype and one candidate
    (often "BTNT") targets exactly that subtype, prefer it over a broader
    "E"/"NTBT" candidate.
  - If nothing distinguishes the candidates clinically, prefer "E", then the
    first candidate listed.

Answer ONLY with the exact code string of the chosen candidate. No quotes,
no explanation, no preamble."""


# ─────────────────────────────────────────────────────────────────────────────
# 6. LLM CALLS
# ─────────────────────────────────────────────────────────────────────────────

async def semantic_match_with_llm(diagnosis: str, candidates: list, model) -> str | None:
    """
    The LLM picks the equivalent candidate, or None when none of them is.

    Called from two different steps, and the number of candidates changes
    what it is really being asked:

      - Fuzzy lexicon step, ONE candidate: rapidfuzz already decided (strict
        threshold, single best hit) and the LLM acts as a VETO — confirm the
        equivalence or reject it and let the row continue down the cascade.

      - Catalogue/search step, a POOL of candidates: string similarity cannot
        settle the classification's wording, so recall is deliberately wide
        and the LLM SELECTS. Answering null is a valid outcome and sends the
        row to NF.

    Both cases can return None, which is what keeps a bad retrieval from
    turning into a wrong code.

    Prompt and candidates are both in English, which is also the language of
    the diagnoses being matched (diagnosis_en) — keeping a single language
    throughout improves the model's accuracy.
    """
    if not candidates:
        return None

    chat = lms.Chat(SEMANTIC_MATCH_SYSTEM_PROMPT)
    chat.add_user_message(
        f'Input diagnosis: "{diagnosis}"\n'
        f'Candidates: {json.dumps(candidates, ensure_ascii=False)}\n'
        f'Which candidate is equivalent? Answer only the exact term or null.'
    )

    raw_text = ""
    try:
        result   = await model.respond(chat, config=PREDICTION_CONFIG)
        raw_text = _parse_channel_response(result.content)
        raw_text = _clean_json_response(raw_text).strip().strip('"').strip("'")

        if raw_text.lower() == "null" or not raw_text:
            return None

        for candidate in candidates:
            if raw_text.lower() == candidate.lower():
                return candidate

        # Tolerate minor wording variations in the model's answer
        best = process.extractOne(
            raw_text.lower(),
            [c.lower() for c in candidates],
            scorer=fuzz.ratio,
            score_cutoff=90,
        )
        if best:
            return candidates[best[2]]
        return None

    except Exception as e:
        print(f"\n    [LLM-semantic] Error: {e} | raw: {raw_text[:200]}")
        return None


_disease_cache: dict = {}

# Signs that the model returned prose instead of a disease name.
_PROSE_RE = re.compile(r"[.;:]\s|\b(the|this|is|are|refers|means|should)\b",
                       re.IGNORECASE)


def _is_valid_disease_name(raw: str) -> bool:
    """
    Is the model's answer a usable disease name?

    It is not validated by character length — naming the entity can produce a
    longer phrase than the input ("Jejunal variceal bleeding" -> "Bleeding
    intestinal varices") — but by shape: between 1 and 10 words, without
    sentence punctuation or explanatory filler.
    """
    raw = (raw or "").strip()
    if len(raw) < 3:
        return False
    if not 1 <= len(raw.split()) <= 10:
        return False
    if _PROSE_RE.search(raw):
        return False
    return True


async def identify_disease(diagnosis: str, model) -> str:
    """
    Ask the LLM which disease the diagnosis is, and return its standard name
    as it would appear in a medical classification.

    This is clinical identification, not text trimming: the resulting name
    may use vocabulary absent from the input ("Cryptococcus neoformans
    meningoencephalitis" -> "Cryptococcosis"). That is why it runs for EVERY
    row: no textual heuristic can decide beforehand which rows need it.

    On error or invalid answer the original diagnosis is returned, so the
    cascade degrades to the previous behaviour instead of breaking.
    """
    diagnosis = (diagnosis or "").strip()
    if not diagnosis:
        return diagnosis

    key = _normalize(diagnosis)
    if key in _disease_cache:
        return _disease_cache[key]

    chat = lms.Chat(DISEASE_IDENTIFICATION_SYSTEM_PROMPT)
    chat.add_user_message(f"Input:  {diagnosis}\nAnswer:")

    name = diagnosis
    try:
        result = await model.respond(chat, config=PREDICTION_CONFIG)
        raw    = _parse_channel_response(result.content)
        raw    = _clean_json_response(raw).strip().strip('"').strip("'")
        raw    = raw.split("\n")[0].strip()
        raw    = re.sub(r"^Answer:\s*", "", raw, flags=re.IGNORECASE).strip()

        if _is_valid_disease_name(raw):
            name = raw
        elif raw:
            print(f"\n    [LLM-disease] Answer discarded: {raw[:80]!r}")
    except Exception as e:
        print(f"\n    [LLM-disease] Error: {e}")

    _disease_cache[key] = name
    return name


async def disambiguate_icd_code(diagnosis_en: str, disease_name: str,
                                candidates: list, model) -> tuple[str, str]:
    """
    Pick a single ICD code among several candidates of the same Orphanet
    disorder, using DisorderMappingRelation plus the diagnosis
    (diagnosis_en) as clinical context.

    `candidates`: [{"code": str, "mapping_relation": str}, ...]
    Returns (code, mapping_relation) of the chosen candidate.

    With a single candidate it returns straight away without calling the LLM.
    On error or unrecognized answer it falls back to a deterministic rule:
    prefer relation "E" (exact mapping), otherwise the first candidate.
    """
    # Deduplicate by code, keeping the first relation seen
    seen = {}
    for c in candidates:
        code = c.get("code", "")
        if code and code not in seen:
            seen[code] = c.get("mapping_relation", "ND")
    candidates = [{"code": c, "mapping_relation": r} for c, r in seen.items()]

    if not candidates:
        return "", ""
    if len(candidates) == 1:
        return candidates[0]["code"], candidates[0]["mapping_relation"]

    def _fallback() -> tuple[str, str]:
        for c in candidates:
            if c["mapping_relation"] == "E":
                return c["code"], c["mapping_relation"]
        return candidates[0]["code"], candidates[0]["mapping_relation"]

    lines = "\n".join(
        f'- code: "{c["code"]}", mapping_relation: "{c["mapping_relation"]}"'
        for c in candidates
    )
    chat = lms.Chat(ICD_CODE_DISAMBIGUATION_SYSTEM_PROMPT)
    chat.add_user_message(
        f'Disease name: "{disease_name}"\n'
        f'Diagnosis: "{diagnosis_en}"\n'
        f'Candidates:\n{lines}\n\n'
        f'Which code is the best match? Answer only the code.'
    )

    raw = ""
    try:
        result = await model.respond(chat, config=PREDICTION_CONFIG)
        raw    = _parse_channel_response(result.content)
        raw    = _clean_json_response(raw).strip().strip('"').strip("'")

        for c in candidates:
            if raw == c["code"]:
                return c["code"], c["mapping_relation"]

        # Tolerate minor variations (whitespace, casing)
        raw_norm = re.sub(r"\s+", "", raw).upper()
        for c in candidates:
            if re.sub(r"\s+", "", c["code"]).upper() == raw_norm:
                return c["code"], c["mapping_relation"]

        print(f"\n    [LLM-disambig] Unrecognized code: {raw[:60]!r} — "
              f"applying deterministic tie-break")
    except Exception as e:
        print(f"\n    [LLM-disambig] Error: {e} | raw: {raw[:100]}")

    return _fallback()


# ─────────────────────────────────────────────────────────────────────────────
# 7. MATCHING CASCADE
# ─────────────────────────────────────────────────────────────────────────────

_lookup_cache: dict = {}


def reset_run_caches() -> None:
    """
    Clear the per-diagnosis caches so every run starts cold.

    CRITICAL for the experiment. The 10 runs share one process and all 10
    CSVs hold the SAME 201 diagnoses, so without this reset run 1 populates
    `_lookup_cache` with every diagnosis and runs 2..10 hit the cache
    100% of the time: no LLM call is made and the 10 CSVs come out byte for
    byte identical. `deep_analysis.py` would then report perfect stability —
    an artefact of the cache, not a property of the pipeline.

    Inside a single run both caches are wanted: they stop repeated diagnoses
    in the same CSV from being recomputed. It is only their survival ACROSS
    runs that destroys the measurement.

    Not reset here, on purpose:
      - the ICD catalogue / lexical index: static reference data, identical
        in every run by definition, and reloading it would only waste time.
      - nf_dictionary.json: it assigns stable identifiers to unmatched
        diagnoses but never decides anything (it is written only after the
        cascade has already failed), so it must persist for NF codes to mean
        the same thing across runs.
    """
    _disease_cache.clear()
    _lookup_cache.clear()



def _make_node_result(diagnosis: str, node: dict, match_type: str, score: float) -> dict:
    return {
        "input":        diagnosis,
        "matched_name": node["canonical_name"],
        "orpha_code":   node["orpha_code"],
        "icd_codes":    node["icd_codes"],
        "match_type":   match_type,
        "score":        score,
        "has_code":     bool(node["icd_codes"]),
    }


def lookup_exact(diagnosis: str, lexical_index: dict) -> dict | None:
    normalized = _normalize(diagnosis)
    if normalized in lexical_index:
        return _make_node_result(diagnosis, lexical_index[normalized], "exact", 100.0)
    return None


def lookup_fuzzy(diagnosis: str, lexical_index: dict, threshold: int) -> dict | None:
    normalized = _normalize(diagnosis)
    result = process.extractOne(
        normalized,
        list(lexical_index.keys()),
        scorer=fuzz.token_sort_ratio,
        score_cutoff=threshold,
    )
    if result:
        best_key, score, _ = result
        return _make_node_result(diagnosis, lexical_index[best_key], "fuzzy", round(score, 1))
    return None


async def _finalize_node_result(diagnosis_en: str, result: dict, model) -> dict:
    """
    Reduce `icd_codes` (possibly several) to a single code, using
    `disambiguate_icd_code` when needed, and leave the result ready to be
    cached and returned.
    """
    code, relation = await disambiguate_icd_code(
        diagnosis_en, result["matched_name"], result["icd_codes"], model
    )
    result["icd_code_complete"] = code
    result["mapping_relation"]  = relation
    return result


def _to_nf(diagnosis: str, nf_dict: dict) -> dict:
    """Diagnosis unmatched by every source -> sequential NF code."""
    code = get_or_create_nf_code(diagnosis, nf_dict)
    return {
        "input":             diagnosis,
        "matched_name":      None,
        "orpha_code":        None,
        "icd_codes":         [{"code": code, "mapping_relation": ""}],
        "icd_code_complete": code,
        "mapping_relation":  "",
        "match_type":        "nf",
        "score":             0.0,
        "has_code":          True,
    }


async def _who_lookup(complete: str, core: str, model,
                      who_client: WHOClient) -> dict | None:
    """
    WHO API step with widened recall.

    Calls /search TWICE — with the complete phrase and with the identified
    disease name — and merges the candidates into a single pool deduplicated
    by code. The LLM then picks using ALWAYS the complete phrase as the
    reference: the name only served to surface candidates the long phrase
    could not retrieve, but the cause is what decides which one is right
    (e.g. "drug-induced diabetes" instead of "type 2").

    Each candidate remembers which query produced it, so `match_source` can
    be reported as complete | core.

    There is no /autocode fallback: in the reference run (201 cases) that
    endpoint did not resolve a single diagnosis. /search with flexisearch
    rarely returns an empty list, and when the LLM rejects every candidate,
    /autocode usually proposes one of those same discarded codes.
    """
    if not who_client.token:
        return None

    queries = [("complete", complete)]
    if core and _normalize(core) != _normalize(complete):
        queries.append(("core", core))

    # requests is blocking: run it off-thread so the event loop keeps going
    pool: dict = {}          # icd11_code -> {title, source}
    for source, query in queries:
        if not query:
            continue
        for cand in await asyncio.to_thread(who_client.search, query):
            code = cand["icd11_code"]
            if code not in pool:      # the complete query has priority
                pool[code] = {"title": cand["title"], "source": source}

    if pool:
        by_title = {}
        for code, info in pool.items():
            by_title.setdefault(info["title"], (code, info["source"]))

        chosen = await semantic_match_with_llm(complete, list(by_title.keys()), model)
        if chosen and chosen in by_title:
            code, source = by_title[chosen]
            return {
                "input":             complete,
                "matched_name":      chosen,
                "orpha_code":        None,
                "icd_codes":         [{"code": code, "mapping_relation": ""}],
                "icd_code_complete": code,
                "mapping_relation":  "",
                "match_type":        "who_search",
                "match_source":      source,
                "score":             0.0,
                "has_code":          True,
            }

    return None


async def lookup_icd11(
    diagnosis_en:  str,
    lexical_index: dict,
    model,
    who_client:    WHOClient,
    nf_dict:       dict,
    threshold:     int,
) -> dict:
    """
    Double-query cascade, entirely in ENGLISH:

        0. Identify the disease (LLM) -> standard name
        1. Cache
        2. Lexicon exact           · complete phrase -> name
        3. Lexicon fuzzy           · 1 candidate, the LLM validates it
        4. WHO /search             · wide pool, the LLM chooses one or none
        5. NF                        · English label

    Precedence: the COMPLETE phrase is always tried before the name, because
    it is more specific. The name only widens recall when the complete
    phrase fails.

    Semantic validation is ALWAYS done against the complete phrase, even when
    the candidate came from the name: that keeps the cause weighing on the
    final decision. If the model misidentified the disease, the retrieved
    candidate will not validate and the row falls to NF instead of receiving
    a wrong code — a visible failure rather than a silent one.

    In steps 2 and 3, when there is a match but the node holds no ICD code,
    the cascade moves on. When the node holds MORE THAN ONE code, it is
    disambiguated with `disambiguate_icd_code` before accepting the result.
    """
    if not diagnosis_en:
        return _to_nf("", nf_dict)

    # 0. Which disease is it? (returns the original phrase if already the name)
    core = await identify_disease(diagnosis_en, model)
    has_core = _normalize(core) != _normalize(diagnosis_en)

    cache_key = _normalize(diagnosis_en)

    # 1. Cache
    if cache_key in _lookup_cache:
        return _lookup_cache[cache_key].copy()

    def _cache_and_return(result: dict) -> dict:
        result.setdefault("match_source", "complete")
        result["core_diagnosis"] = core
        _lookup_cache[cache_key] = result
        return result

    # Variants to try, ordered by specificity
    variants = [("complete", diagnosis_en)]
    if has_core:
        variants.append(("core", core))

    # 2. Exact — reliable, no extra validation
    for source, term in variants:
        result = lookup_exact(term, lexical_index)
        if result and result["has_code"]:
            result = await _finalize_node_result(diagnosis_en, result, model)
            result["match_type"]   = f"exact_{source}"
            result["match_source"] = source
            return _cache_and_return(result)

    # 3. Fuzzy + LLM semantic validation (against the complete phrase)
    for source, term in variants:
        result = lookup_fuzzy(term, lexical_index, threshold)
        if result and result["has_code"]:
            confirmed = await semantic_match_with_llm(
                diagnosis_en, [result["matched_name"]], model
            )
            if confirmed:
                result = await _finalize_node_result(diagnosis_en, result, model)
                result["match_type"]   = f"fuzzy_{source}"
                result["match_source"] = source
                return _cache_and_return(result)

    # 4. WHO ICD-11 API: unified pool of both queries
    who_result = await _who_lookup(diagnosis_en, core, model, who_client)
    if who_result:
        return _cache_and_return(who_result)

    # 5. NF — English label
    result = _to_nf(diagnosis_en, nf_dict)
    result["match_source"] = "none"
    return _cache_and_return(result)


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = ["diagnosis_es", "diagnosis_en", "core_diagnosis",
                    "clinical_summary",
                    "icd11_code", "icd_code_complete", "category", "chapter",
                    "hierarchical_distance", "hierarchy_path",
                    "mapping_relation",
                    "icd11_lookup", "match_type", "match_source"]


def read_csv_robust(path: str) -> tuple[list, list]:
    """
    Read the CSV tolerating mixed encodings (UTF-8 rows and cp1252 rows in
    the same file, typical when it was partially edited in Excel).

    Decodes line by line: tries UTF-8 and falls back to cp1252 only on the
    lines that fail. On rewrite everything is normalized to UTF-8.

    Returns (rows, fieldnames) preserving the original column order and
    appending any missing required column.
    """
    with open(path, "rb") as f:
        raw = f.read()

    if raw.startswith(b"\xef\xbb\xbf"):      # BOM
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

    for col in REQUIRED_COLUMNS:
        if col not in fieldnames:
            fieldnames.append(col)

    return rows, fieldnames


async def process_row(
    row:            dict,
    lexical_index:  dict,
    model,
    who_client:     WHOClient,
    nf_dict:        dict,
    threshold:      int,
    chapter_cache:  dict,
) -> dict:
    """
    Assign icd11_code plus the hierarchy columns (icd_code_complete,
    category, chapter, hierarchical_distance) to one row. Every other column
    is preserved.

    Only diagnosis_en feeds the pipeline; diagnosis_es is kept in the CSV as
    source data but is never read.
    """
    diagnosis_en = (row.get("diagnosis_en") or "").strip()

    if not diagnosis_en:
        row["icd11_code"]            = ""
        row["icd_code_complete"]     = ""
        row["category"]              = ""
        row["chapter"]               = ""
        row["hierarchical_distance"] = ""
        row["hierarchy_path"]        = ""
        row["mapping_relation"]      = ""
        row["core_diagnosis"]        = ""
        row["match_type"]            = ""
        row["match_source"]          = ""
        return row

    result = await lookup_icd11(
        diagnosis_en, lexical_index, model, who_client, nf_dict, threshold
    )

    icd_code_complete = result.get("icd_code_complete", "")
    category          = truncate_code(icd_code_complete)
    chapter, hierarchical_distance, hierarchy_path = await resolve_chapter(
        who_client, category, chapter_cache)

    row["icd11_code"]            = category        # backwards compatible: equals category
    row["icd_code_complete"]     = icd_code_complete
    row["category"]              = category
    row["chapter"]               = chapter
    row["hierarchical_distance"] = hierarchical_distance
    row["hierarchy_path"]        = hierarchy_path
    row["mapping_relation"]      = result.get("mapping_relation", "")
    row["core_diagnosis"]        = result.get("core_diagnosis", "")
    row["match_type"]            = result["match_type"]
    row["match_source"]          = result.get("match_source", "")
    return row


async def process_csv_file(
    csv_path:      str,
    lexical_index: dict,
    model,
    who_client:    WHOClient,
    nf_dict:       dict,
    chapter_cache: dict,
    threshold:     int,
    max_rows:      int | None,
) -> None:
    """Process a single CSV in place, reusing the already loaded index/client/model."""
    rows_in, fieldnames = read_csv_robust(csv_path)

    if not rows_in:
        print(f"[CSV] {csv_path} has no data rows. Nothing to process.\n")
        return

    if max_rows is not None:
        rows_to_process = rows_in[:max_rows]
        print(f"[CSV] Test mode: {len(rows_to_process)} of {len(rows_in)} rows")
    else:
        rows_to_process = rows_in
        print(f"[CSV] {len(rows_in)} rows loaded")

    with_en = sum(1 for r in rows_to_process if (r.get("diagnosis_en") or "").strip())
    print(f"[CSV] diagnosis_en: {with_en}")

    if with_en == 0:
        print("[!] No row has diagnosis_en: the whole matching pipeline "
              "(lexical index and WHO API) stays idle and everything falls to NF.")
    print()

    BATCH = 20

    with tqdm(total=len(rows_to_process), desc=f"Processing {Path(csv_path).name}", unit="row") as pbar:
        for i in range(0, len(rows_to_process), BATCH):
            batch = rows_to_process[i:i + BATCH]
            tasks = [
                process_row(row, lexical_index, model, who_client, nf_dict, threshold, chapter_cache)
                for row in batch
            ]
            await asyncio.gather(*tasks)
            pbar.update(len(batch))

    _print_stats(rows_to_process)

    # Overwrite the same CSV, preserving all of its original columns
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_in:
            writer.writerow({k: (row.get(k) or "") for k in fieldnames})

    print(f"[CSV] Updated in place: {csv_path}\n")


async def run_pipeline(
    product1_path:     str,
    csv_paths:         list[str],
    model_name:        str,
    lms_host:          str,
    who_client_id:     str,
    who_client_secret: str,
    threshold:         int,
    max_rows:          int | None,
    log_path:          str | None = None,
    logging_enabled:   bool = True,
    seed:              int = DEFAULT_SEED,
):
    with RunLogger(log_path, enabled=logging_enabled) as logger:
        await _run_all(product1_path, csv_paths, model_name, lms_host,
                       who_client_id, who_client_secret, threshold, max_rows,
                       seed, logger)


async def _run_all(product1_path, csv_paths, model_name, lms_host,
                   who_client_id, who_client_secret, threshold, max_rows,
                   seed, logger):
    # Lexical index (built once for every run)
    lexical_index = build_lexical_index(product1_path)

    # WHO client
    who_client = WHOClient(who_client_id, who_client_secret)
    who_client.authenticate()

    # NF dictionary
    nf_dict   = load_nf_dictionary()
    nf_before = len(nf_dict)
    print(f"[NF]  Dictionary loaded: {nf_before} entries\n")

    # Chapter cache
    chapter_cache = load_chapter_cache()
    print(f"[Chapters] Cache loaded: {len(chapter_cache)} categories\n")

    logger.session_header({
        "icd version":      ICD_SOURCE,
        "model":            model_name,
        "lms host":         lms_host,
        "temperature":      DEFAULT_TEMPERATURE,
        "seed (load-time)": seed,
        "runs":             len(csv_paths),
        "fuzzy threshold":  threshold,
        "rows per run":     max_rows if max_rows is not None else "all",
        "who api":          "authenticated" if who_client.token else "UNAVAILABLE",
        "product1":         product1_path,
        "lexical index":    f"{len(lexical_index)} entries",
    })

    async with lms.AsyncClient(lms_host) as client:
        print(f"[LLM] Host: {lms_host}")
        print(f"[LLM] Connecting to model: {model_name}")
        try:
            # The seed only applies when this call actually loads the model;
            # it is ignored if LM Studio already holds it in memory. At
            # temperature 0 it changes nothing either way.
            model = await client.llm.model(model_name, config={"seed": seed})
        except Exception as e:
            print(f"[LLM] ERROR connecting to {lms_host}: {e}")
            print("[LLM] Check that LM Studio is serving on that IP:port")
            print("      (Settings > Developer > Serve on Local Network) and that")
            print("      the server machine firewall allows the port.\n")
            return
        print(f"[LLM] Connected\n")

        for index, csv_path in enumerate(csv_paths, 1):
            logger.run_header(index, len(csv_paths), csv_path)
            # Every run must start cold, otherwise runs 2..10 are served from
            # run 1's cache and the experiment measures nothing.
            reset_run_caches()
            await process_csv_file(
                csv_path, lexical_index, model, who_client, nf_dict,
                chapter_cache, threshold, max_rows,
            )

    nf_after = len(nf_dict)
    save_nf_dictionary(nf_dict)
    save_chapter_cache(chapter_cache)

    print(f"[NF]  {nf_before} -> {nf_after} entries (+{nf_after - nf_before})")
    print(f"[NF]  Saved to: {NF_DICT_PATH}")
    print(f"[Chapters] Cache saved: {len(chapter_cache)} categories in {CHAPTER_CACHE_PATH}\n")


def _print_stats(rows: list):
    LABELS = [
        ("exact_complete",  "Exact lexical index (complete)"),
        ("exact_core",      "Exact lexical index (name)"),
        ("fuzzy_complete",  "Fuzzy lexicon + LLM validates (complete)"),
        ("fuzzy_core",      "Fuzzy lexicon + LLM validates (name)"),
        ("who_search",      "WHO /search pool + LLM chooses"),
        ("nf",              "NF (not found)"),
    ]
    counts  = {k: 0 for k, _ in LABELS}
    sources = {"complete": 0, "core": 0, "none": 0}
    n_core  = 0
    total   = 0

    for row in rows:
        mt = row.get("match_type")
        if not mt:
            continue
        total += 1
        counts[mt] = counts.get(mt, 0) + 1

        src = row.get("match_source") or ""
        if src in sources:
            sources[src] += 1

        core = (row.get("core_diagnosis") or "").strip()
        diag = (row.get("diagnosis_en")   or "").strip()
        if core and _normalize(core) != _normalize(diag):
            n_core += 1

    t = max(total, 1)
    print(f"\n── Matching statistics ─────────────────────────────────")
    print(f"  Diagnoses processed         : {total}")
    for key, label in LABELS:
        print(f"  {label:<32}: {counts.get(key, 0):>3} ({counts.get(key, 0)/t*100:5.1f}%)")

    resolved = total - counts.get("nf", 0)
    print(f"  {'':-<32}   {'':->11}")
    print(f"  {'With ' + ICD_SOURCE + ' code':<32}: {resolved:>3} ({resolved/t*100:5.1f}%)")
    print(f"────────────────────────────────────────────────────────")
    print(f"  Name differs from phrase    : {n_core} of {total}")
    print(f"  Resolved by complete phrase : {sources['complete']}")
    print(f"  Resolved by disease name    : {sources['core']}   <- gain from this step")
    print(f"────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# 9. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
N_RUNS     = 10


PRODUCT1_FILENAME = "en_product1.json"
PRODUCT1_SEARCH_DEPTH = 3          # how many parent directories are walked


def find_product1(explicit: str | None = None) -> str | None:
    """
    Locate en_product1.json without depending on the directory the script is
    invoked from.

    When --product1 is given, that path is honoured (relative to the cwd) and
    nothing else is searched. Otherwise the search walks from the SCRIPT
    folder upwards (up to PRODUCT1_SEARCH_DEPTH levels) and, at each level,
    looks at:
      1. the directory itself     (.../ICD11/en_product1.json)
      2. its direct subfolders    (.../DrChatPatinEval/en_product1.json)

    That order makes a local copy next to the script win over the shared
    repository copy, which usually lives in a sibling folder.

    Returns the path found, or None when there is none.
    """
    if explicit:
        path = Path(explicit).expanduser()
        return str(path) if path.is_file() else None

    levels = [SCRIPT_DIR, *list(SCRIPT_DIR.parents)[:PRODUCT1_SEARCH_DEPTH]]
    for directory in levels:
        direct = directory / PRODUCT1_FILENAME
        if direct.is_file():
            return str(direct)
        for match in sorted(directory.glob(f"*/{PRODUCT1_FILENAME}")):
            if match.is_file():
                return str(match)

    return None


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
        description="Assign ICD-11 codes (complete + category + chapter) using "
                    "the Orphanet lexical index + WHO API + LLM. Without --csv "
                    "it processes dataset-run1.csv .. dataset-run10.csv."
    )
    parser.add_argument("--product1", default=None,
                        help="Path to the Orphanet product1 JSON (English). If "
                             "omitted, en_product1.json is searched next to the "
                             "script and in its parent folders.")
    parser.add_argument("--csv", default=None,
                        help="Single CSV to process in place. If omitted, "
                             "processes dataset-run1.csv .. dataset-run10.csv "
                             "next to the script.")
    parser.add_argument("--model", default="medgemma-27b-it",
                        help="Model name in LM Studio")
    parser.add_argument("--lms-host",
                        default=os.environ.get("LMS_HOST", "10.8.0.45:1234"),
                        help="LM Studio server host:port (or env LMS_HOST). "
                             "Default: localhost:1234. Local network example: "
                             "192.168.1.50:1234")
    parser.add_argument("--who-client-id",
                        default=os.environ.get("WHO_CLIENT_ID", "00da56ea-0fe1-465e-b830-61cb0add2173_2741cb3a-0cd6-4977-af95-6258be8bd99a"),
                        help="WHO ICD-11 API client ID (or env WHO_CLIENT_ID)")
    parser.add_argument("--who-client-secret",
                        default=os.environ.get("WHO_CLIENT_SECRET", "h1MrlyMGBnGt7Q6kAnSpq8/1s18FkkzPboT7MIaim7o="),
                        help="WHO ICD-11 API client secret (or env WHO_CLIENT_SECRET)")
    parser.add_argument("--threshold", type=int, default=85,
                        help="Fuzzy threshold (default: 85)")
    parser.add_argument("--rows", type=int, default=None,
                        help="Process only the first N rows of each CSV (test mode)")
    parser.add_argument("--log", default=None,
                        help="Session log file (default: experiment-runs.log "
                             "next to the script). Appended, never truncated.")
    parser.add_argument("--no-log", action="store_true",
                        help="Do not write the session log")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Load-time seed (default: {DEFAULT_SEED}). Has no "
                             f"effect at temperature 0, which is greedy.")
    args = parser.parse_args()

    product1_path = find_product1(args.product1)
    if not product1_path:
        if args.product1:
            print(f"[ERROR] File not found --product1: {args.product1}")
        else:
            print(f"[ERROR] {PRODUCT1_FILENAME} not found starting from "
                  f"{SCRIPT_DIR} nor in its {PRODUCT1_SEARCH_DEPTH} parent "
                  f"folders (nor in their direct subfolders).")
            print(f"        Copy the file next to the script or pass "
                  f"--product1 PATH.")
        return
    print(f"[Lexical index] product1 detected: {product1_path}\n")

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

    asyncio.run(run_pipeline(
        product1_path=product1_path,
        csv_paths=csv_paths,
        model_name=args.model,
        lms_host=args.lms_host,
        who_client_id=args.who_client_id,
        who_client_secret=args.who_client_secret,
        threshold=args.threshold,
        max_rows=args.rows,
        log_path=args.log,
        logging_enabled=not args.no_log,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
