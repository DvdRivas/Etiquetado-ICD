"""
icd-experiment.py
-----------------
Asigna códigos ICD-11 (capítulo + categoría) a diagnósticos clínicos usando:
  1. KGraph desde JSON Orphanet (product1) — nombres + sinónimos → ICD-11
  2. WHO ICD-11 API /search              — candidatos validados semánticamente por LLM
  3. NF dictionary                       — último recurso (NF-XXXX secuencial)

Cascada de matching (toda en INGLÉS, sobre diagnosis_en):
    0. Enfermedad  — el LLM responde "¿qué enfermedad es?" y da su nombre
                     estándar (todas las filas)
    1. Caché       — evita recomputar diagnósticos repetidos
    2. Exacto      — KGraph, frase completa y luego nombre
    3. Fuzzy       — KGraph ≥85 + validación LLM, completa y luego nombre
    4. WHO /search — AMBAS consultas, pool unificado, el LLM elige
    5. NF          — no encontrado, se registra en nf_dictionary.json

Frase completa vs nombre de la enfermedad:
  Los diagnósticos traen la causa y el contexto adosados ("Acute intermittent
  porphyria due to HMBS mutation"), lo que impide el match aunque la entidad
  exista en la ontología. Identificar la enfermedad ("Acute intermittent
  porphyria") amplía la RECUPERACIÓN.

  No es un recorte de texto sino una identificación clínica: el nombre puede
  usar vocabulario ausente en la entrada ("Cryptococcus neoformans
  meningoencephalitis" → "Cryptococcosis").

  La validación semántica se hace SIEMPRE contra la frase completa, porque la
  causa puede cambiar la categoría ICD-11 asignada (diabetes inducida por
  fármacos vs diabetes tipo 2). La frase completa se prueba siempre primero.

Por qué inglés:
  - ICD-11 se redacta en inglés; las traducciones tienen menos términos y
    sinónimos indexados, tanto en la API como en Orphanet.
  - en_product1.json cubre 11645 trastornos con 6553 mapeados a ICD-11,
    contra 11456/6143 de la versión española.
  - La columna diagnosis_es se conserva para lectura humana y para la
    evaluación de coherencia posterior, pero no participa en el matching.

Todos los códigos se truncan a capítulo + categoría (lo que está a la
izquierda del punto): "5A11.2" → "5A11", "BA00.0Z" → "BA00".

CSV entrada/salida (mismo archivo, se sobreescribe in-place):
    filename | diagnosis_en | diagnosis_es | core_diagnosis | clinical_summary
            | icd11_code | icd11_lookup | match_type | match_source | consistency

Columnas que escribe este script:
    icd11_code      — código truncado a capítulo + categoría
    core_diagnosis  — nombre de la enfermedad identificada por el LLM
                      (igual a diagnosis_en si ya era el nombre estándar)
    match_type      — paso de la cascada que resolvió
    match_source    — "complete" | "core" | "none"
                      ("core" = resuelto gracias al nombre de la enfermedad)

Este script solo llena `icd11_code`. La búsqueda inversa (`icd11_lookup`)
se implementará por separado.

Dependencias:
    pip install lmstudio rapidfuzz tqdm requests

Uso:
    python icd-experiment.py --csv "../dataset-run1.csv"

    # Modo prueba (5 filas)
    python icd-experiment.py --csv "../dataset-run1.csv" --rows 5

    # Servidor LM Studio en otra máquina de la red local
    python icd-experiment.py --csv "../dataset-run1.csv" \
        --lms-host 192.168.1.50:1234

Servidor LM Studio: por defecto localhost:1234. Para usar otra máquina,
pasa --lms-host IP:PUERTO o define la variable de entorno LMS_HOST.
En el equipo servidor hay que habilitar el acceso por red en LM Studio:
Settings > Developer > "Serve on Local Network".

Credenciales WHO: se leen de las variables de entorno
WHO_CLIENT_ID y WHO_CLIENT_SECRET, o vía --who-client-id / --who-client-secret.
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

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ─────────────────────────────────────────────────────────────────────────────
# 1. UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\-–—]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def truncate_code(code: str) -> str:
    """
    Reduce un código ICD-11 a capítulo + categoría: todo lo que está
    a la izquierda del primer punto.

        "5A11.2"  → "5A11"
        "BA00.0Z" → "BA00"
        "1C62"    → "1C62"
        "NF-0001" → "NF-0001"   (los códigos NF no se truncan)
    """
    code = (code or "").strip()
    if not code or code.startswith("NF-"):
        return code
    return code.split(".", 1)[0]


def _parse_channel_response(raw: str) -> str:
    """Extrae el canal final si el modelo usa formato de canales."""
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
    """Elimina bloques markdown ```json ... ``` si el modelo los incluye."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2. KNOWLEDGE GRAPH desde product1 JSON (ICD-11)
# ─────────────────────────────────────────────────────────────────────────────

def build_knowledge_graph(product1_path: str) -> dict:
    """
    Construye el KGraph desde product1 de Orphanet.

    Estructura de cada nodo:
        {
            "canonical_name": str,
            "orpha_code":     str,
            "icd11_codes":    [str, ...],
        }

    Las claves son nombres normalizados (canónico + todos los sinónimos).
    """
    print(f"[KGraph] Cargando {product1_path} ...")

    with open(product1_path, encoding="utf-8") as f:
        data = json.load(f)

    disorders = (
        data.get("JDBOR", [{}])[0]
            .get("DisorderList", [{}])[0]
            .get("Disorder", [])
    )

    kg    = {}
    total = 0
    con   = 0

    for disorder in disorders:
        total += 1
        orpha_code = disorder.get("OrphaCode", "")
        names      = disorder.get("Name", [])
        canonical  = _pick_label(names, preferred_lang="en")
        if not canonical:
            continue

        icd11_codes = _extract_icd11(disorder)
        if icd11_codes:
            con += 1

        node = {
            "canonical_name": canonical,
            "orpha_code":     orpha_code,
            "icd11_codes":    icd11_codes,
        }

        # Nombre canónico tiene prioridad
        kg[_normalize(canonical)] = node

        # Sinónimos no sobreescriben al canónico
        for syn_block in disorder.get("SynonymList", []):
            for syn in syn_block.get("Synonym", []):
                label = syn.get("label", "").strip()
                if label:
                    key = _normalize(label)
                    if key not in kg:
                        kg[key] = node

    print(f"[KGraph] {total} trastornos | {con} con ICD-11 | {len(kg)} entradas totales\n")
    return kg


def _pick_label(names: list, preferred_lang: str = "en") -> str:
    for n in names:
        if n.get("lang") == preferred_lang:
            return n["label"].strip()
    return names[0]["label"].strip() if names else ""


def _extract_icd11(disorder: dict) -> list:
    """Extrae códigos ICD-11 desde ExternalReferenceList."""
    codes = []
    for ref_block in disorder.get("ExternalReferenceList", []):
        for ref in ref_block.get("ExternalReference", []):
            if ref.get("Source") != "ICD-11":
                continue
            code = ref.get("Reference", "").strip()
            if code:
                codes.append(code)
    return codes


# ─────────────────────────────────────────────────────────────────────────────
# 3. WHO ICD-11 API
# ─────────────────────────────────────────────────────────────────────────────

class WHOClient:
    """Cliente WHO ICD-11 API con manejo de token OAuth2."""

    TOKEN_URL    = "https://icdaccessmanagement.who.int/connect/token"
    BASE_URL     = "https://id.who.int/icd/release/11/2024-01/mms"
    SEARCH_URL   = f"{BASE_URL}/search"

    def __init__(self, client_id: str, client_secret: str):
        self.client_id     = client_id
        self.client_secret = client_secret
        self.token         = None

    def authenticate(self) -> bool:
        if not self.client_id or not self.client_secret:
            print("[WHO] Sin credenciales — WHO API deshabilitada\n")
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
            print("[WHO] Token obtenido correctamente\n")
            return True
        except Exception as e:
            print(f"[WHO] Error de autenticación: {e}\n")
            return False

    def _headers(self, lang: str = "en") -> dict:
        return {
            "Authorization":   f"Bearer {self.token}",
            "Accept":          "application/json",
            "Accept-Language": lang,
            "API-Version":     "v2",
        }

    @staticmethod
    def _strip_highlight(text: str) -> str:
        """La API marca coincidencias con etiquetas <em>; se limpian."""
        return re.sub(r"</?em[^>]*>", "", text or "").strip()

    def search(self, text: str, lang: str = "en", limit: int = 5) -> list:
        """
        Busca candidatos ICD-11 para un texto diagnóstico libre.
        Retorna [{"icd11_code": str, "title": str}, ...] (máx. `limit`).
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
                headers=self._headers(lang),
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


# ─────────────────────────────────────────────────────────────────────────────
# 4. NF DICTIONARY PERSISTENTE
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
    Retorna el código NF existente para el diagnóstico, o crea uno nuevo.
    Formato secuencial global: NF-0001, NF-0002, ...
    """
    key = _normalize(diagnosis)
    if key in nf_dict:
        return nf_dict[key]["code"]

    # Siguiente número: máximo existente + 1 (robusto ante huecos)
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
# 5. PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

SEMANTIC_MATCH_SYSTEM_PROMPT = """Eres un especialista en terminología médica en español.
Tu función es identificar si alguno de los términos candidatos de una ontología médica
es equivalente clínico al diagnóstico de entrada.

Reglas de equivalencia:
- Acepta equivalencias clínicas generales: "Tuberculosis pulmonar" es equivalente a
  "Tuberculosis pulmonar primaria" porque en un DDx clínico se refieren al mismo proceso.
- Acepta cuando el candidato es la forma más específica del mismo concepto.
- Rechaza (null) cuando el candidato es un concepto claramente diferente,
  aunque comparta palabras.
- Si varios candidatos son válidos, elige el clínicamente más preciso.
- Responde ÚNICAMENTE con el término candidato exacto, o null si ninguno es equivalente.
- No uses comillas, no des explicaciones, solo el término o null.

Ejemplos:
  Entrada: "Tuberculosis pulmonar"
  Candidatos: ["Tuberculosis pulmonar primaria", "Tuberculosis miliar", "Tuberculosis"]
  Respuesta: Tuberculosis pulmonar primaria

  Entrada: "Hemoptisis idiopática"
  Candidatos: ["Hemosiderosis pulmonar idiopática", "Hemorragia pulmonar neonatal"]
  Respuesta: null

  Entrada: "Micobacterias no tuberculosas"
  Candidatos: ["Infección pulmonar por micobacterias no tuberculosas", "Tuberculosis multifocal"]
  Respuesta: Infección pulmonar por micobacterias no tuberculosas"""


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


SEMANTIC_MATCH_SYSTEM_PROMPT_EN = """You are a specialist in medical terminology.
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


# ─────────────────────────────────────────────────────────────────────────────
# 6. LLAMADAS AL LLM
# ─────────────────────────────────────────────────────────────────────────────

async def semantic_match_with_llm(diagnosis: str, candidates: list, model,
                                  lang: str = "es") -> str | None:
    """
    El LLM elige el candidato equivalente, o None si ninguno lo es.

    `lang` selecciona el prompt: "es" para candidatos del KGraph Orphanet
    (en español) y "en" para candidatos de la WHO API (en inglés). Mantener
    prompt y candidatos en el mismo idioma mejora la precisión del modelo.
    """
    if not candidates:
        return None

    if lang == "en":
        chat = lms.Chat(SEMANTIC_MATCH_SYSTEM_PROMPT_EN)
        chat.add_user_message(
            f'Input diagnosis: "{diagnosis}"\n'
            f'Candidates: {json.dumps(candidates, ensure_ascii=False)}\n'
            f'Which candidate is equivalent? Answer only the exact term or null.'
        )
    else:
        chat = lms.Chat(SEMANTIC_MATCH_SYSTEM_PROMPT)
        chat.add_user_message(
            f'Diagnóstico de entrada: "{diagnosis}"\n'
            f'Candidatos: {json.dumps(candidates, ensure_ascii=False)}\n'
            f'¿Cuál candidato es equivalente? Responde solo el término exacto o null.'
        )

    raw_text = ""
    try:
        result   = await model.respond(chat)
        raw_text = _parse_channel_response(result.content)
        raw_text = _clean_json_response(raw_text).strip().strip('"').strip("'")

        if raw_text.lower() == "null" or not raw_text:
            return None

        for candidate in candidates:
            if raw_text.lower() == candidate.lower():
                return candidate

        # Tolerancia a variaciones menores en la respuesta del modelo
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

# Signos de que el modelo devolvió prosa en vez de un nombre de enfermedad.
_PROSE_RE = re.compile(r"[.;:]\s|\b(the|this|is|are|refers|means|should)\b",
                       re.IGNORECASE)


def _is_valid_disease_name(raw: str) -> bool:
    """
    ¿La respuesta del modelo es un nombre de enfermedad utilizable?

    No se valida por longitud de caracteres —nombrar la entidad puede dar una
    frase más larga que la entrada ("Jejunal variceal bleeding" → "Bleeding
    intestinal varices")— sino por forma: entre 1 y 10 palabras, sin
    puntuación de oración ni muletillas explicativas.
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
    Pregunta al LLM qué enfermedad es el diagnóstico y devuelve su nombre
    estándar, tal como aparecería en una clasificación médica.

    Es una tarea de identificación clínica, no de recorte de texto: el nombre
    resultante puede usar vocabulario ausente en la entrada ("Cryptococcus
    neoformans meningoencephalitis" → "Cryptococcosis"). Por eso se consulta
    para TODAS las filas: no hay heurística textual capaz de decidir de
    antemano cuáles lo necesitan.

    Ante error o respuesta inválida se devuelve el diagnóstico original, con
    lo que la cascada degrada al comportamiento previo en vez de romperse.
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
        result = await model.respond(chat)
        raw    = _parse_channel_response(result.content)
        raw    = _clean_json_response(raw).strip().strip('"').strip("'")
        raw    = raw.split("\n")[0].strip()
        raw    = re.sub(r"^Answer:\s*", "", raw, flags=re.IGNORECASE).strip()

        if _is_valid_disease_name(raw):
            name = raw
        elif raw:
            print(f"\n    [LLM-disease] Respuesta descartada: {raw[:80]!r}")
    except Exception as e:
        print(f"\n    [LLM-disease] Error: {e}")

    _disease_cache[key] = name
    return name


# ─────────────────────────────────────────────────────────────────────────────
# 7. MATCHING EN CASCADA
# ─────────────────────────────────────────────────────────────────────────────

_lookup_cache: dict = {}


def _make_node_result(diagnosis: str, node: dict, match_type: str, score: float) -> dict:
    return {
        "input":        diagnosis,
        "matched_name": node["canonical_name"],
        "orpha_code":   node["orpha_code"],
        "icd11_codes":  node["icd11_codes"],
        "match_type":   match_type,
        "score":        score,
        "has_icd11":    bool(node["icd11_codes"]),
    }


def lookup_exact(diagnosis: str, kg: dict) -> dict | None:
    normalized = _normalize(diagnosis)
    if normalized in kg:
        return _make_node_result(diagnosis, kg[normalized], "exact", 100.0)
    return None


def lookup_fuzzy(diagnosis: str, kg: dict, threshold: int) -> dict | None:
    normalized = _normalize(diagnosis)
    result = process.extractOne(
        normalized,
        list(kg.keys()),
        scorer=fuzz.token_sort_ratio,
        score_cutoff=threshold,
    )
    if result:
        best_key, score, _ = result
        return _make_node_result(diagnosis, kg[best_key], "fuzzy", round(score, 1))
    return None


def _to_nf(diagnosis: str, nf_dict: dict) -> dict:
    """Diagnóstico sin match en ninguna fuente → código NF secuencial."""
    return {
        "input":        diagnosis,
        "matched_name": None,
        "orpha_code":   None,
        "icd11_codes":  [get_or_create_nf_code(diagnosis, nf_dict)],
        "match_type":   "nf",
        "score":        0.0,
        "has_icd11":    True,
    }


async def _who_lookup(complete: str, core: str, model,
                      who_client: WHOClient) -> dict | None:
    """
    Paso WHO API con recuperación ampliada.

    Consulta /search DOS veces —con la frase completa y con el nombre de la
    enfermedad identificada— y une los candidatos en un solo pool deduplicado
    por código. El LLM elige después usando SIEMPRE la frase completa como
    referencia: el nombre sólo sirvió para que aparecieran candidatos que la
    frase larga no recuperaba, pero la causa es la que decide cuál es el
    correcto (p. ej. "diabetes inducida por fármacos" en vez de "tipo 2").

    Cada candidato recuerda de qué consulta vino, para poder reportar
    `match_source` = complete | core.

    No hay fallback a /autocode: en la corrida de referencia (201 casos) ese
    endpoint no resolvió ni un solo diagnóstico. /search con flexisearch rara
    vez devuelve lista vacía, y cuando el LLM rechaza todos los candidatos,
    /autocode suele proponer uno de esos mismos códigos ya descartados.
    """
    if not who_client.token:
        return None

    queries = [("complete", complete)]
    if core and _normalize(core) != _normalize(complete):
        queries.append(("core", core))

    # requests es bloqueante: se ejecuta en hilo aparte para no frenar el loop
    pool: dict = {}          # icd11_code -> {title, source}
    for source, query in queries:
        if not query:
            continue
        for cand in await asyncio.to_thread(who_client.search, query):
            code = cand["icd11_code"]
            if code not in pool:      # la consulta completa tiene prioridad
                pool[code] = {"title": cand["title"], "source": source}

    if pool:
        by_title = {}
        for code, info in pool.items():
            by_title.setdefault(info["title"], (code, info["source"]))

        chosen = await semantic_match_with_llm(
            complete, list(by_title.keys()), model, lang="en"
        )
        if chosen and chosen in by_title:
            code, source = by_title[chosen]
            return {
                "input":        complete,
                "matched_name": chosen,
                "orpha_code":   None,
                "icd11_codes":  [code],
                "match_type":   "who_search",
                "match_source": source,
                "score":        0.0,
                "has_icd11":    True,
            }

    return None


async def lookup_icd11(
    diagnosis_es: str,
    diagnosis_en: str,
    kg:           dict,
    model,
    who_client:   WHOClient,
    nf_dict:      dict,
    threshold:    int,
) -> dict:
    """
    Cascada con doble consulta, toda en INGLÉS:

        0. Identificar la enfermedad (LLM) → nombre estándar
        1. Caché
        2. KGraph exacto      · frase completa → nombre
        3. KGraph fuzzy + LLM · frase completa → nombre
        4. WHO /search + LLM  · pool unificado de ambas consultas
        5. NF                 · etiqueta en español

    Precedencia: la frase COMPLETA se prueba siempre antes que el nombre,
    porque es más específica. El nombre sólo amplía la recuperación cuando
    la completa falla.

    La validación semántica se hace SIEMPRE contra la frase completa, incluso
    cuando el candidato provino del nombre: así la causa sigue pesando en la
    decisión final. Si el modelo identificó mal la enfermedad, el candidato
    recuperado no validará y la fila caerá a NF en vez de recibir un código
    incorrecto: falla visible en lugar de silenciosa.

    En los pasos 2 y 3, si hay match pero el nodo no tiene ICD-11, se
    continúa al siguiente paso.
    """
    if not diagnosis_en:
        return _to_nf(diagnosis_es or "", nf_dict)

    # 0. ¿Qué enfermedad es? (devuelve la frase original si ya es el nombre)
    core = await identify_disease(diagnosis_en, model)
    has_core = _normalize(core) != _normalize(diagnosis_en)

    cache_key = _normalize(diagnosis_en)

    # 1. Caché
    if cache_key in _lookup_cache:
        return _lookup_cache[cache_key].copy()

    def _cache_and_return(result: dict) -> dict:
        result.setdefault("match_source", "complete")
        result["core_diagnosis"] = core
        _lookup_cache[cache_key] = result
        return result

    # Variantes a probar, en orden de especificidad
    variants = [("complete", diagnosis_en)]
    if has_core:
        variants.append(("core", core))

    # 2. Exacto — confiable, sin validación adicional
    for source, term in variants:
        result = lookup_exact(term, kg)
        if result and result["has_icd11"]:
            result["match_type"]   = f"exact_{source}"
            result["match_source"] = source
            return _cache_and_return(result)

    # 3. Fuzzy + validación semántica LLM (contra la frase completa)
    for source, term in variants:
        result = lookup_fuzzy(term, kg, threshold)
        if result and result["has_icd11"]:
            confirmed = await semantic_match_with_llm(
                diagnosis_en, [result["matched_name"]], model, lang="en"
            )
            if confirmed:
                result["match_type"]   = f"fuzzy_{source}"
                result["match_source"] = source
                return _cache_and_return(result)

    # 4. WHO ICD-11 API: pool unificado de ambas consultas
    who_result = await _who_lookup(diagnosis_en, core, model, who_client)
    if who_result:
        return _cache_and_return(who_result)

    # 5. NF — etiqueta en español para lectura humana
    result = _to_nf(diagnosis_es or diagnosis_en, nf_dict)
    result["match_source"] = "none"
    return _cache_and_return(result)


# ─────────────────────────────────────────────────────────────────────────────
# 8. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = ["diagnosis_es", "diagnosis_en", "core_diagnosis",
                    "clinical_summary", "icd11_code", "icd11_lookup",
                    "match_type", "match_source"]


def read_csv_robust(path: str) -> tuple[list, list]:
    """
    Lee el CSV tolerando encoding mixto (filas UTF-8 y filas cp1252 en el
    mismo archivo, típico cuando se editó parcialmente en Excel).

    Decodifica línea por línea: intenta UTF-8 y cae a cp1252 solo en las
    líneas que fallan. Al reescribir, todo queda normalizado a UTF-8.

    Retorna (filas, nombres_de_columna) preservando el orden original de
    las columnas y agregando las requeridas que falten.
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
        print(f"[CSV] Encoding mixto: {fallbacks} líneas decodificadas como "
              f"cp1252. Se normalizará todo a UTF-8 al guardar.")

    reader     = csv.DictReader(decoded)
    rows       = list(reader)
    fieldnames = list(reader.fieldnames or [])

    for col in REQUIRED_COLUMNS:
        if col not in fieldnames:
            fieldnames.append(col)

    return rows, fieldnames


async def process_row(
    row:        dict,
    kg:         dict,
    model,
    who_client: WHOClient,
    nf_dict:    dict,
    threshold:  int,
) -> dict:
    """
    Asigna icd11_code a una fila. Preserva el resto de columnas.

    Usa diagnosis_es para el KGraph y diagnosis_en para la WHO API.
    """
    diagnosis_es = (row.get("diagnosis_es") or "").strip()
    diagnosis_en = (row.get("diagnosis_en") or "").strip()

    if not diagnosis_es and not diagnosis_en:
        row["icd11_code"]     = ""
        row["core_diagnosis"] = ""
        row["match_type"]     = ""
        row["match_source"]   = ""
        return row

    result = await lookup_icd11(
        diagnosis_es, diagnosis_en, kg, model, who_client, nf_dict, threshold
    )

    raw_code = result["icd11_codes"][0] if result["icd11_codes"] else ""
    row["icd11_code"]     = truncate_code(raw_code)
    row["core_diagnosis"] = result.get("core_diagnosis", "")
    row["match_type"]     = result["match_type"]
    row["match_source"]   = result.get("match_source", "")
    return row


async def run_pipeline(
    product1_path:     str,
    csv_path:          str,
    model_name:        str,
    lms_host:          str,
    who_client_id:     str,
    who_client_secret: str,
    threshold:         int,
    max_rows:          int | None,
):
    # KGraph
    kg = build_knowledge_graph(product1_path)

    # WHO client
    who_client = WHOClient(who_client_id, who_client_secret)
    who_client.authenticate()

    # NF dictionary
    nf_dict   = load_nf_dictionary()
    nf_before = len(nf_dict)
    print(f"[NF]  Diccionario cargado: {nf_before} entradas\n")

    # Leer CSV
    rows_in, fieldnames = read_csv_robust(csv_path)

    if not rows_in:
        print(f"[CSV] {csv_path} no tiene filas de datos. Nada que procesar.\n")
        return

    if max_rows is not None:
        rows_to_process = rows_in[:max_rows]
        print(f"[CSV] Modo prueba: {len(rows_to_process)} de {len(rows_in)} filas")
    else:
        rows_to_process = rows_in
        print(f"[CSV] {len(rows_in)} filas cargadas")

    # diagnosis_en alimenta todo el matching; diagnosis_es solo etiqueta NF.
    con_es = sum(1 for r in rows_to_process if (r.get("diagnosis_es") or "").strip())
    con_en = sum(1 for r in rows_to_process if (r.get("diagnosis_en") or "").strip())
    print(f"[CSV] diagnosis_es: {con_es} | diagnosis_en: {con_en}")

    if con_en == 0:
        print("[!] Ninguna fila tiene diagnosis_en: el matching completo "
              "(KGraph y WHO API) quedará inactivo y todo caerá a NF.")
    print()

    BATCH = 20

    async with lms.AsyncClient(lms_host) as client:
        print(f"[LLM] Host: {lms_host}")
        print(f"[LLM] Conectando con modelo: {model_name}")
        try:
            model = await client.llm.model(model_name)
        except Exception as e:
            print(f"[LLM] ERROR al conectar con {lms_host}: {e}")
            print("[LLM] Verifica que LM Studio esté sirviendo en esa IP:puerto")
            print("      (Settings > Developer > Serve on Local Network) y que el")
            print("      firewall del equipo servidor permita el puerto.\n")
            return
        print(f"[LLM] Conexión exitosa\n")

        with tqdm(total=len(rows_to_process), desc="Procesando filas", unit="fila") as pbar:
            for i in range(0, len(rows_to_process), BATCH):
                batch = rows_to_process[i:i + BATCH]
                tasks = [
                    process_row(row, kg, model, who_client, nf_dict, threshold)
                    for row in batch
                ]
                await asyncio.gather(*tasks)
                pbar.update(len(batch))

    # Estadísticas antes de limpiar el campo auxiliar
    _print_stats(rows_to_process)
    # Sobreescribir el mismo CSV preservando todas sus columnas originales
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_in:
            writer.writerow({k: (row.get(k) or "") for k in fieldnames})

    nf_after = len(nf_dict)
    save_nf_dictionary(nf_dict)

    print(f"[CSV] Actualizado in-place: {csv_path}")
    print(f"[NF]  {nf_before} → {nf_after} entradas (+{nf_after - nf_before})")
    print(f"[NF]  Guardado en: {NF_DICT_PATH}\n")


def _print_stats(rows: list):
    LABELS = [
        ("exact_complete",  "Exacto KGraph (completa)"),
        ("exact_core",      "Exacto KGraph (nombre)"),
        ("fuzzy_complete",  "Fuzzy KGraph (completa)"),
        ("fuzzy_core",      "Fuzzy KGraph (nombre)"),
        ("who_search",      "WHO /search + LLM"),
        ("nf",              "NF (no encontrado)"),
    ]
    counts   = {k: 0 for k, _ in LABELS}
    sources  = {"complete": 0, "core": 0, "none": 0}
    n_core   = 0
    total    = 0

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
    print(f"\n── Estadísticas de matching ────────────────────────────")
    print(f"  Diagnósticos procesados    : {total}")
    for key, label in LABELS:
        print(f"  {label:<27}: {counts.get(key, 0):>3} ({counts.get(key, 0)/t*100:5.1f}%)")

    resueltos = total - counts.get("nf", 0)
    print(f"  {'':-<27}   {'':->11}")
    print(f"  {'Con código ICD-11':<27}: {resueltos:>3} ({resueltos/t*100:5.1f}%)")
    print(f"────────────────────────────────────────────────────────")
    print(f"  Nombre distinto de la frase : {n_core} de {total}")
    print(f"  Resueltos por frase completa: {sources['complete']}")
    print(f"  Resueltos por nombre        : {sources['core']}   <- ganancia del cambio")
    print(f"────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# 9. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CSV = str(Path(__file__).resolve().parent.parent / "dataset-run1.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Asigna códigos ICD-11 (capítulo + categoría) usando "
                    "KGraph Orphanet + WHO API + LLM"
    )
    parser.add_argument("--product1", default="en_product1.json",
                        help="Ruta al JSON product1 de Orphanet (inglés)")
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help=f"CSV a procesar in-place (default: {DEFAULT_CSV})")
    parser.add_argument("--model", default="medgemma-27b-it",
                        help="Nombre del modelo en LM Studio")
    parser.add_argument("--lms-host",
                        default=os.environ.get("LMS_HOST", "10.8.0.45:1234"),
                        help="Host:puerto del servidor LM Studio "
                             "(o env LMS_HOST). Default: localhost:1234. "
                             "Ejemplo red local: 192.168.1.50:1234")
    parser.add_argument("--who-client-id",
                        default=os.environ.get("WHO_CLIENT_ID", "00da56ea-0fe1-465e-b830-61cb0add2173_2741cb3a-0cd6-4977-af95-6258be8bd99a"),
                        help="Client ID WHO ICD-11 API (o env WHO_CLIENT_ID)")
    parser.add_argument("--who-client-secret",
                        default=os.environ.get("WHO_CLIENT_SECRET", "h1MrlyMGBnGt7Q6kAnSpq8/1s18FkkzPboT7MIaim7o="),
                        help="Client Secret WHO ICD-11 API (o env WHO_CLIENT_SECRET)")
    parser.add_argument("--threshold", type=int, default=85,
                        help="Umbral fuzzy (default: 85)")
    parser.add_argument("--rows", type=int, default=None,
                        help="Procesar solo las primeras N filas (modo prueba)")
    args = parser.parse_args()

    for path, name in [(args.product1, "--product1"), (args.csv, "--csv")]:
        if not Path(path).exists():
            print(f"[ERROR] Archivo no encontrado {name}: {path}")
            return

    asyncio.run(run_pipeline(
        product1_path=args.product1,
        csv_path=args.csv,
        model_name=args.model,
        lms_host=args.lms_host,
        who_client_id=args.who_client_id,
        who_client_secret=args.who_client_secret,
        threshold=args.threshold,
        max_rows=args.rows,
    ))


if __name__ == "__main__":
    main()
