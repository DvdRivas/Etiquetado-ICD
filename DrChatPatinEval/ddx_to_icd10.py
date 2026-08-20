"""
ddx_to_icd10.py
---------------
Convierte diagnósticos diferenciales (DDx) a códigos ICD-10 usando:
  1. LM Studio (modelo local, async) — extracción DDx y matching semántico
  2. Knowledge Graph desde JSON Orphanet — sinónimos + nombre canónico → ICD-10
  3. Diccionario NR persistente — diagnósticos no encontrados en Orphanet

Estrategia de matching (cascada):
    1. Exacto      — normalización directa contra claves del KGraph
    2. Fuzzy       — rapidfuzz token_sort_ratio (default threshold: 75)
    3. Semántico   — LLM elige entre top-5 candidatos fuzzy (~60%) o retorna null
    4. NR          — no encontrado en Orphanet, se registra en nr_dictionary.json

CSV entrada : query_id | iteration | ddx
CSV salida  : query_id | iteration | icd10_codes | ddx_parsed | match_details

Dependencias:
    pip install lmstudio rapidfuzz tqdm

Uso:
    # Modo prueba
    python ddx_to_icd10.py --orphanet es_product1.json --csv evaluaciones.csv --output resultado.csv --rows 5

    # Corrida completa
    python ddx_to_icd10.py --orphanet es_product1.json --csv evaluaciones.csv --output resultado.csv

    # Opciones avanzadas
    python ddx_to_icd10.py --orphanet es_product1.json --csv evaluaciones.csv --output resultado.csv \
        --model "openai/gpt-oss-120b" --threshold 75 --semantic-threshold 60
"""

import json
import csv
import re
import argparse
import asyncio
from pathlib import Path
from datetime import date

import lmstudio as lms
from rapidfuzz import process, fuzz
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# 1. KNOWLEDGE GRAPH desde JSON Orphanet
# ─────────────────────────────────────────────────────────────────────────────

def build_knowledge_graph(orphanet_path: str) -> dict:
    """
    Construye el KGraph desde el JSON de Orphanet.

    Estructura de cada nodo:
        {
            "canonical_name": str,
            "orpha_code":     str,
            "icd10_codes":    [str, ...]  # exactMatch > NTBT > BTNT
        }

    Las claves del dict son nombres normalizados (canónico + todos los sinónimos).
    """
    print(f"[KGraph] Cargando {orphanet_path} ...")

    with open(orphanet_path, encoding="utf-8") as f:
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

        names     = disorder.get("Name", [])
        canonical = _pick_label(names, preferred_lang="es")
        if not canonical:
            continue

        icd10_codes = _extract_icd10(disorder)
        if icd10_codes:
            con += 1

        node = {
            "canonical_name": canonical,
            "orpha_code":     orpha_code,
            "icd10_codes":    icd10_codes,
        }

        # Indexar por nombre canónico (prioridad alta)
        kg[_normalize(canonical)] = node

        # Indexar por cada sinónimo (no sobreescribe el canónico)
        for syn_block in disorder.get("SynonymList", []):
            for syn in syn_block.get("Synonym", []):
                label = syn.get("label", "").strip()
                if label:
                    key = _normalize(label)
                    if key not in kg:
                        kg[key] = node

    print(f"[KGraph] {total} trastornos | {con} con ICD-10 | {len(kg)} entradas totales\n")
    return kg


def _pick_label(names: list, preferred_lang: str = "es") -> str:
    for n in names:
        if n.get("lang") == preferred_lang:
            return n["label"].strip()
    return names[0]["label"].strip() if names else ""


def _extract_icd10(disorder: dict) -> list:
    """
    Extrae códigos ICD-10 aceptando las tres relaciones.
    Orden de prioridad: exactMatch > NTBT > BTNT
    """
    ACCEPTED = ("correspondencia exacta", "ntbt", "btnt")
    buckets  = {"correspondencia exacta": [], "ntbt": [], "btnt": []}

    for ref_block in disorder.get("ExternalReferenceList", []):
        for ref in ref_block.get("ExternalReference", []):
            if ref.get("Source") != "ICD-10":
                continue
            rel_list  = ref.get("DisorderMappingRelation", [])
            if not rel_list:
                continue
            rel_label = rel_list[0].get("Name", [{}])[0].get("label", "").lower()
            code      = ref.get("Reference", "").strip()
            if not code:
                continue
            for key in ACCEPTED:
                if key in rel_label:
                    buckets[key].append(code)
                    break

    return (
        buckets["correspondencia exacta"] +
        buckets["ntbt"] +
        buckets["btnt"]
    )


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\-–—]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _parse_channel_response(raw: str) -> str:
    """
    Extrae el canal final si el modelo usa formato de canales.
    Maneja correctamente saltos de linea y canales parciales.

    Formato esperado:
        <|channel|>analysis<|message|>...<|end|>
        <|start|>assistant<|channel|>final<|message|>[JSON aqui]
    """
    # Intentar extraer canal final explicito
    match = re.search(
        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|\Z)",
        raw, re.DOTALL,
    )
    if match:
        return match.group(1).strip()

    # Si hay canales pero ninguno es final, descartar bloque analysis
    # y tomar solo lo que viene despues del ultimo <|end|>
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
# 2. DICCIONARIO NR PERSISTENTE
# ─────────────────────────────────────────────────────────────────────────────

NR_DICT_PATH = Path(__file__).parent / "nr_dictionary.json"

# Categorías clínicas para los códigos NR
NR_CATEGORIES = {
    "respiratorio":    "RESP",
    "cardiovascular":  "CARD",
    "infeccioso":      "INFEC",
    "neurológico":     "NEUR",
    "gastrointestinal":"GAST",
    "renal":           "REN",
    "hematológico":    "HEMAT",
    "endocrino":       "ENDOC",
    "musculoesquelético": "MUSC",
    "dermatológico":   "DERM",
    "oncológico":      "ONCOL",
    "otro":            "OTR",
}


def load_nr_dictionary() -> dict:
    """Carga el diccionario NR desde disco. Si no existe, retorna vacío."""
    if NR_DICT_PATH.exists():
        with open(NR_DICT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_nr_dictionary(nr_dict: dict) -> None:
    """Guarda el diccionario NR en disco."""
    with open(NR_DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(nr_dict, f, ensure_ascii=False, indent=2)


def get_or_create_nr_code(
    diagnosis: str,
    category:  str,
    nr_dict:   dict,
) -> str:
    """
    Retorna el código NR existente para el diagnóstico,
    o crea uno nuevo si no existe.

    Formato: NR-{CATEGORIA}-{NNN}
    Ejemplo: NR-RESP-001, NR-INFEC-002
    """
    key = _normalize(diagnosis)

    # Ya existe en el diccionario
    if key in nr_dict:
        return nr_dict[key]["code"]

    # Determinar prefijo de categoría
    cat_normalized = _normalize(category)
    prefix = "OTR"
    for cat_name, cat_code in NR_CATEGORIES.items():
        if cat_name in cat_normalized:
            prefix = cat_code
            break

    # Calcular siguiente número para esta categoría
    existing_in_cat = [
        v for v in nr_dict.values()
        if v["code"].startswith(f"NR-{prefix}-")
    ]
    next_num = len(existing_in_cat) + 1
    code     = f"NR-{prefix}-{next_num:03d}"

    # Registrar en el diccionario
    nr_dict[key] = {
        "code":       code,
        "label":      diagnosis,
        "category":   category,
        "first_seen": date.today().isoformat(),
    }

    return code


# ─────────────────────────────────────────────────────────────────────────────
# 3. PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """Eres un extractor especializado de diagnósticos clínicos en español.
Tu única función es leer un texto de diagnóstico diferencial médico y extraer
los nombres de las enfermedades o condiciones diagnósticas propuestas.

Reglas estrictas:
- Extrae TODOS los diagnósticos presentes, máximo 5.
- Respeta el orden de probabilidad tal como aparecen en el texto.
- Usa el nombre clínico estándar en español, sin especificadores extra.
- NO agregues adjetivos como "crónica", "activa", "primaria", "aguda" a menos que
  sean parte indivisible del nombre (ej: "Fiebre reumática aguda").
- NO uses nombres en inglés. Si el texto tiene "Q-fever" escríbelo como "Fiebre Q".
- NO incluyas procedimientos, estudios ni recomendaciones de tratamiento.
- NO incluyas texto adicional, explicaciones ni comentarios.
- Responde ÚNICAMENTE con un array JSON válido de strings.

Ejemplos de corrección:
  ❌ "Tuberculosis pulmonar activa"   ✅ "Tuberculosis pulmonar"
  ❌ "Q-fever crónica"               ✅ "Fiebre Q"
  ❌ "Brucelosis bacteriana"         ✅ "Brucelosis"
  ❌ "Neumonía atypical"             ✅ "Neumonía atípica"

Ejemplo de respuesta correcta:
["Tuberculosis pulmonar", "Fiebre Q", "Psitacosis", "Brucelosis", "Neumonía atípica"]"""


SEMANTIC_MATCH_SYSTEM_PROMPT = """Eres un especialista en terminología médica en español.
Tu función es identificar si alguno de los términos candidatos de una ontología médica
es equivalente clínico al diagnóstico de entrada.

Reglas de equivalencia:
- Acepta equivalencias clínicas generales: "Tuberculosis pulmonar" es equivalente a
  "Tuberculosis pulmonar primaria" porque en un DDx clínico se refieren al mismo proceso.
- Acepta cuando el candidato es la forma más específica del mismo concepto.
- Rechaza (null) cuando el candidato es un concepto claramente diferente, aunque comparta palabras.
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


NR_CATEGORY_SYSTEM_PROMPT = """Eres un clasificador de diagnósticos médicos.
Dado un nombre de diagnóstico, responde ÚNICAMENTE con la categoría clínica
que mejor lo describe, eligiendo entre estas opciones exactas:

respiratorio, cardiovascular, infeccioso, neurológico, gastrointestinal,
renal, hematológico, endocrino, musculoesquelético, dermatológico, oncológico, otro

Responde solo con una palabra de la lista. Sin explicaciones."""


# ─────────────────────────────────────────────────────────────────────────────
# 4. LLAMADAS AL LLM
# ─────────────────────────────────────────────────────────────────────────────

async def extract_ddx_with_llm(ddx_text: str, model) -> list:
    """Extrae lista ordenada de diagnósticos desde el texto DDx."""
    if not ddx_text or not isinstance(ddx_text, str) or not ddx_text.strip():
        return []

    chat = lms.Chat(EXTRACTION_SYSTEM_PROMPT)
    chat.add_user_message(
        f"Extrae los diagnósticos diferenciales del siguiente texto:\n\n{ddx_text.strip()}"
    )

    raw_text = ""
    try:
        result   = await model.respond(chat)
        raw_text = _parse_channel_response(result.content)
        raw_text = _clean_json_response(raw_text)
        parsed   = json.loads(raw_text)

        if isinstance(parsed, list):
            return [str(d).strip() for d in parsed if str(d).strip()][:5]
        return []

    except json.JSONDecodeError as e:
        print(f"\n    [LLM-extract] JSON inválido: {e} | raw: {raw_text[:200]}")
        return []
    except Exception as e:
        print(f"\n    [LLM-extract] Error: {e}")
        return []


async def semantic_match_with_llm(
    diagnosis:  str,
    candidates: list[str],
    model,
) -> str | None:
    """LLM elige el candidato equivalente o retorna null."""
    if not candidates:
        return None

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

        # Verificar que sea uno de los candidatos (exacto o fuzzy de seguridad)
        for candidate in candidates:
            if raw_text.lower() == candidate.lower():
                return candidate

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


async def classify_nr_category(diagnosis: str, model) -> str:
    """Clasifica el diagnóstico NR en una categoría clínica."""
    chat = lms.Chat(NR_CATEGORY_SYSTEM_PROMPT)
    chat.add_user_message(f'Diagnóstico: "{diagnosis}"')

    try:
        result   = await model.respond(chat)
        raw_text = _parse_channel_response(result.content).strip().lower()

        # Verificar que sea una categoría válida
        for cat in NR_CATEGORIES:
            if cat in raw_text:
                return cat
        return "otro"

    except Exception:
        return "otro"


# ─────────────────────────────────────────────────────────────────────────────
# 5. MATCHING EN CASCADA
# ─────────────────────────────────────────────────────────────────────────────

# Caché en memoria: evita repetir lookups para el mismo diagnóstico
_lookup_cache: dict = {}


def _make_node_result(diagnosis: str, node: dict, match_type: str, score: float) -> dict:
    """Helper que construye el dict de resultado desde un nodo del KGraph."""
    return {
        "input":        diagnosis,
        "matched_name": node["canonical_name"],
        "orpha_code":   node["orpha_code"],
        "icd10_codes":  node["icd10_codes"],
        "match_type":   match_type,
        "score":        score,
        "has_icd10":    bool(node["icd10_codes"]),
    }


def lookup_exact(diagnosis: str, kg: dict) -> dict | None:
    """Match exacto por clave normalizada."""
    normalized = _normalize(diagnosis)
    if normalized in kg:
        return _make_node_result(diagnosis, kg[normalized], "exact", 100.0)
    return None


def lookup_fuzzy(
    diagnosis: str,
    kg:        dict,
    threshold: int,
) -> dict | None:
    """Fuzzy match con threshold dado. Retorna None si no supera el umbral."""
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


def get_semantic_candidates(
    diagnosis:          str,
    kg:                 dict,
    semantic_threshold: int,
    top_n:              int = 5,
) -> list[str]:
    """Top-N nombres canónicos candidatos con threshold bajo para el LLM."""
    normalized = _normalize(diagnosis)
    results    = process.extract(
        normalized,
        list(kg.keys()),
        scorer=fuzz.token_sort_ratio,
        score_cutoff=semantic_threshold,
        limit=top_n,
    )
    seen  = set()
    names = []
    for key, _, _ in results:
        canonical = kg[key]["canonical_name"]
        if canonical not in seen:
            seen.add(canonical)
            names.append(canonical)
    return names


async def _to_nr(diagnosis: str, model, nr_dict: dict) -> dict:
    """Convierte un diagnóstico sin match (o con match pero sin ICD-10) a NR."""
    category = await classify_nr_category(diagnosis, model)
    nr_code  = get_or_create_nr_code(diagnosis, category, nr_dict)
    return {
        "input":        diagnosis,
        "matched_name": None,
        "orpha_code":   None,
        "icd10_codes":  [nr_code],
        "match_type":   "nr",
        "score":        0.0,
        "has_icd10":    True,
    }


async def lookup_icd10(
    diagnosis:          str,
    kg:                 dict,
    model,
    nr_dict:            dict,
    threshold:          int,
    semantic_threshold: int,
) -> dict:
    """
    Matching en cascada con caché:
        1. Caché          — evita recomputar diagnósticos ya vistos
        2. Exacto         — normalización directa, sin validación (100% confiable)
        3. Fuzzy          — todos los matches fuzzy pasan por validación semántica LLM
        4. Semántico LLM  — top-5 candidatos al 60%
        5. NR             — no encontrado en Orphanet

    En cualquier paso donde haya match pero icd10_codes esté vacío → NR.
    """
    cache_key = _normalize(diagnosis)

    # 1. Caché
    if cache_key in _lookup_cache:
        cached = _lookup_cache[cache_key].copy()
        cached["input"] = diagnosis
        return cached

    async def _cache_and_return(result: dict) -> dict:
        _lookup_cache[cache_key] = result
        return result

    # 2. Exacto — confiable al 100%, sin validación adicional
    result = lookup_exact(diagnosis, kg)
    if result:
        if not result["has_icd10"]:
            return await _cache_and_return(await _to_nr(diagnosis, model, nr_dict))
        return await _cache_and_return(result)

    # 3. Fuzzy — todos los matches pasan por validación semántica LLM
    fuzzy_result = lookup_fuzzy(diagnosis, kg, threshold)
    if fuzzy_result:
        confirmed = await semantic_match_with_llm(
            diagnosis, [fuzzy_result["matched_name"]], model
        )
        if confirmed:
            if not fuzzy_result["has_icd10"]:
                return await _cache_and_return(await _to_nr(diagnosis, model, nr_dict))
            fuzzy_result["match_type"] = "fuzzy_validated"
            return await _cache_and_return(fuzzy_result)
        # LLM rechazó el match fuzzy → continuar al semántico

    # 4. Semántico LLM con candidatos al semantic_threshold (60%)
    candidates   = get_semantic_candidates(diagnosis, kg, semantic_threshold, top_n=5)
    matched_name = await semantic_match_with_llm(diagnosis, candidates, model)
    if matched_name:
        node = kg.get(_normalize(matched_name))
        if node:
            sem_result = _make_node_result(diagnosis, node, "semantic", 0.0)
            if not sem_result["has_icd10"]:
                return await _cache_and_return(await _to_nr(diagnosis, model, nr_dict))
            return await _cache_and_return(sem_result)

    # 5. NR — no encontrado en Orphanet
    return await _cache_and_return(await _to_nr(diagnosis, model, nr_dict))


# ─────────────────────────────────────────────────────────────────────────────
# 6. PIPELINE PRINCIPAL (async, paralelo)
# ─────────────────────────────────────────────────────────────────────────────

async def process_row(
    row:                dict,
    kg:                 dict,
    model,
    nr_dict:            dict,
    threshold:          int,
    semantic_threshold: int,
) -> dict:
    """Procesa una fila completa en paralelo para todos sus diagnósticos."""
    query_id  = row.get("query_id",  "")
    iteration = row.get("iteration", "")
    ddx_text  = row.get("ddx",       "")

    diagnoses = await extract_ddx_with_llm(ddx_text, model)

    # Matching en paralelo para los diagnósticos de la fila
    lookup_tasks = [
        lookup_icd10(diag, kg, model, nr_dict, threshold, semantic_threshold)
        for diag in diagnoses
    ]
    results = await asyncio.gather(*lookup_tasks)

    icd10_list    = []
    match_details = []

    for result in results:
        code = result["icd10_codes"][0] if result["icd10_codes"] else "N/A"
        icd10_list.append(code)
        match_details.append(
            f"{result['input']} → {result['matched_name'] or result['icd10_codes'][0] if result['icd10_codes'] else 'None'}"
            f" [{result['match_type']}, {result['score']}%]"
            f" → {code}"
        )

    return {
        "query_id":      query_id,
        "iteration":     iteration,
        "icd10_codes":   json.dumps(icd10_list, ensure_ascii=False),
        "ddx_parsed":    " | ".join(diagnoses),
        "match_details": " ;; ".join(match_details),
    }


async def run_pipeline(
    csv_path:           str,
    kg:                 dict,
    output_path:        str,
    model_name:         str,
    threshold:          int,
    semantic_threshold: int,
    max_rows:           int | None,
):
    # Leer CSV
    rows_in = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_in.append(row)

    if max_rows is not None:
        rows_in = rows_in[:max_rows]
        print(f"[CSV] Modo prueba: procesando {len(rows_in)} filas\n")
    else:
        print(f"[CSV] {len(rows_in)} filas cargadas\n")

    # Cargar diccionario NR
    nr_dict = load_nr_dictionary()
    nr_before = len(nr_dict)
    print(f"[NR] Diccionario cargado: {nr_before} entradas existentes\n")

    async with lms.AsyncClient() as client:
        print(f"[LLM] Conectando con modelo: {model_name}")
        model = await client.llm.model(model_name)
        print(f"[LLM] Conexión exitosa\n")

        rows_out  = []
        BATCH     = 20   # filas por lote — ajustar según RAM del servidor

        with tqdm(total=len(rows_in), desc="Procesando filas", unit="fila") as pbar:
            for i in range(0, len(rows_in), BATCH):
                batch   = rows_in[i:i + BATCH]
                tasks   = [
                    process_row(row, kg, model, nr_dict, threshold, semantic_threshold)
                    for row in batch
                ]
                results = await asyncio.gather(*tasks)
                rows_out.extend(results)
                pbar.update(len(batch))

    # Reordenar por query_id + iteration
    rows_out.sort(key=lambda r: (
        int(r["query_id"])  if str(r["query_id"]).isdigit()  else r["query_id"],
        int(r["iteration"]) if str(r["iteration"]).isdigit() else r["iteration"],
    ))

    # Guardar CSV output
    fieldnames = ["query_id", "iteration", "icd10_codes", "ddx_parsed", "match_details"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    # Guardar diccionario NR actualizado
    nr_after = len(nr_dict)
    save_nr_dictionary(nr_dict)

    print(f"\n[CSV] Output guardado en: {output_path}")
    print(f"[NR]  Diccionario actualizado: {nr_before} → {nr_after} entradas (+{nr_after - nr_before})")
    print(f"[NR]  Guardado en: {NR_DICT_PATH}\n")
    _print_stats(rows_out)


def _print_stats(rows: list):
    total = exact = fuzzy_val = semantic = nr = none_ = 0
    for row in rows:
        for detail in row["match_details"].split(" ;; "):
            if not detail:
                continue
            total += 1
            if   "[exact,"           in detail: exact     += 1
            elif "[fuzzy_validated," in detail: fuzzy_val += 1
            elif "[semantic,"        in detail: semantic  += 1
            elif "[nr,"              in detail: nr        += 1
            elif "[none,"            in detail: none_     += 1

    t = max(total, 1)
    print(f"── Estadísticas de matching ──────────────────────────")
    print(f"  Diagnósticos procesados    : {total}")
    print(f"  Exacto                     : {exact}     ({exact/t*100:.1f}%)")
    print(f"  Fuzzy validado (LLM)       : {fuzzy_val} ({fuzzy_val/t*100:.1f}%)")
    print(f"  Semántico (LLM)            : {semantic}  ({semantic/t*100:.1f}%)")
    print(f"  NR (no en Orphanet)        : {nr}        ({nr/t*100:.1f}%)")
    print(f"  Sin match                  : {none_}     ({none_/t*100:.1f}%)")
    print(f"──────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convierte DDx a ICD-10 usando LM Studio + KGraph Orphanet"
    )
    parser.add_argument("--orphanet",           required=True,
                        help="Ruta al JSON de Orphanet (ej: es_product1.json)")
    parser.add_argument("--csv",                required=True,
                        help="CSV de entrada: query_id, iteration, ddx")
    parser.add_argument("--output",             required=True,
                        help="Ruta del CSV de salida")
    parser.add_argument("--model",              default="openai/gpt-oss-120b",
                        help="Nombre del modelo en LM Studio (default: openai/gpt-oss-120b)")
    parser.add_argument("--threshold",          type=int, default=85,
                        help="Umbral fuzzy alto (default: 85); zona 75-85 pasa por validación LLM")
    parser.add_argument("--semantic-threshold", type=int, default=60,
                        help="Umbral fuzzy para candidatos semánticos (default: 60)")
    parser.add_argument("--rows",               type=int, default=None,
                        help="Procesar solo las primeras N filas (modo prueba)")
    args = parser.parse_args()

    for path, name in [(args.orphanet, "--orphanet"), (args.csv, "--csv")]:
        if not Path(path).exists():
            print(f"[ERROR] Archivo no encontrado para {name}: {path}")
            return

    kg = build_knowledge_graph(args.orphanet)

    asyncio.run(run_pipeline(
        csv_path=args.csv,
        kg=kg,
        output_path=args.output,
        model_name=args.model,
        threshold=args.threshold,
        semantic_threshold=args.semantic_threshold,
        max_rows=args.rows,
    ))


if __name__ == "__main__":
    main()