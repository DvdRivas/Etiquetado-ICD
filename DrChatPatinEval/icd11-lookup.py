"""
icd11-lookup.py
---------------
Proceso INVERSO de icd-experiment.py.

Lee la columna `icd11_code` del CSV y, para cada código, consulta la WHO
ICD-11 API para obtener el título oficial de esa entidad. El resultado se
escribe en la columna `icd11_lookup`.

Después pregunta a un modelo de LM Studio si ese título es coherente con el
diagnóstico original, y escribe "yes" o "no" en la columna `consistency`.

FASE 1 — Búsqueda inversa (determinista, sin LLM):
  1. Códigos NF-XXXX  → se resuelven contra nf_dictionary.json (no son ICD-11)
  2. Códigos ICD-11   → WHO API /codeinfo → stemId → título oficial
  3. Fallback         → WHO API /search restringido al código

FASE 2 — Evaluación de coherencia (LLM):
  Pregunta "¿el lookup es coherente con el diagnóstico?" → yes / no
  - Criterio flexible: el título siempre será más general que el diagnóstico
    porque los códigos se truncaron a capítulo + categoría. Que englobe
    correctamente al diagnóstico cuenta como "yes".
  - Filas con código NF se omiten por defecto (su lookup es el propio
    diagnóstico, siempre daría "yes"). Usa --eval-nf para incluirlas.
  - Filas cuyo código no resolvió a ningún título se marcan "no".

CSV entrada/salida (mismo archivo, se sobreescribe in-place):
    filename | diagnosis_en | diagnosis_es | clinical_summary
            | icd11_code | icd11_lookup | consistency

Dependencias:
    pip install requests tqdm lmstudio

Uso:
    python icd11-lookup.py

    # Modo prueba (5 filas)
    python icd11-lookup.py --rows 5

    # LM Studio en otra máquina de la red local
    python icd11-lookup.py --lms-host 192.168.1.50:1234

    # Solo búsqueda inversa, sin evaluar coherencia
    python icd11-lookup.py --skip-consistency

    # Rehacer filas ya procesadas
    python icd11-lookup.py --overwrite

Credenciales WHO: variables de entorno WHO_CLIENT_ID y WHO_CLIENT_SECRET,
o vía --who-client-id / --who-client-secret.
Servidor LM Studio: env LMS_HOST o --lms-host (default localhost:1234).
"""

import os
import csv
import json
import re
import argparse
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import requests
import urllib3
import lmstudio as lms
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LECTURA ROBUSTA DE CSV
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = ["diagnosis_es", "diagnosis_en", "clinical_summary",
                    "icd11_code", "icd11_lookup", "consistency"]


def read_csv_robust(path: str) -> tuple[list, list]:
    """
    Lee el CSV tolerando encoding mixto (filas UTF-8 y filas cp1252 en el
    mismo archivo). Al reescribir, todo queda normalizado a UTF-8.

    Retorna (filas, nombres_de_columna) preservando el orden original.
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
        print(f"[CSV] Encoding mixto: {fallbacks} líneas decodificadas como "
              f"cp1252. Se normalizará todo a UTF-8 al guardar.")

    reader     = csv.DictReader(decoded)
    rows       = list(reader)
    fieldnames = list(reader.fieldnames or [])

    for col in REQUIRED_COLUMNS:
        if col not in fieldnames:
            fieldnames.append(col)

    return rows, fieldnames


# ─────────────────────────────────────────────────────────────────────────────
# 2. DICCIONARIO NF
# ─────────────────────────────────────────────────────────────────────────────

NF_DICT_PATH = Path(__file__).parent / "nf_dictionary.json"


def load_nf_labels() -> dict:
    """
    Carga nf_dictionary.json y lo invierte a { "NF-0001": "etiqueta", ... }
    para poder resolver un código NF a su texto original.
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
# 3. WHO ICD-11 API — BÚSQUEDA INVERSA
# ─────────────────────────────────────────────────────────────────────────────

class WHOLookupClient:
    """Cliente WHO ICD-11 API para resolver código → título oficial."""

    TOKEN_URL    = "https://icdaccessmanagement.who.int/connect/token"
    BASE_URL     = "https://id.who.int/icd/release/11/2024-01/mms"
    CODEINFO_URL = f"{BASE_URL}/codeinfo"
    SEARCH_URL   = f"{BASE_URL}/search"

    def __init__(self, client_id: str, client_secret: str, lang: str = "es"):
        self.client_id     = client_id
        self.client_secret = client_secret
        self.lang          = lang
        self.token         = None
        self.session       = requests.Session()
        self._cache        = {}
        self._cache_lock   = Lock()

    # ── autenticación ────────────────────────────────────────────────────
    def authenticate(self) -> bool:
        if not self.client_id or not self.client_secret:
            print("[WHO] Sin credenciales. Define WHO_CLIENT_ID y "
                  "WHO_CLIENT_SECRET o usa --who-client-id/--who-client-secret\n")
            return False
        try:
            r = self.session.post(
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

    def _headers(self) -> dict:
        return {
            "Authorization":   f"Bearer {self.token}",
            "Accept":          "application/json",
            "Accept-Language": self.lang,
            "API-Version":     "v2",
        }

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _extract_title(payload: dict) -> str:
        """El título viene como {"@value": "..."} o como string plano."""
        title = payload.get("title")
        if isinstance(title, dict):
            title = title.get("@value", "")
        title = title or ""
        return re.sub(r"</?em[^>]*>", "", title).strip()

    def _get(self, url: str, **kwargs):
        return self.session.get(
            url, headers=self._headers(), verify=False, timeout=15, **kwargs
        )

    # ── resolución ───────────────────────────────────────────────────────
    def _via_codeinfo(self, code: str) -> str:
        """
        /codeinfo/{code} devuelve el stemId (URI de la entidad). Se sigue
        esa URI para obtener el título oficial.
        """
        r = self._get(f"{self.CODEINFO_URL}/{code}")
        if r.status_code == 404:
            return ""
        r.raise_for_status()

        stem_id = (r.json() or {}).get("stemId", "")
        if not stem_id:
            return ""

        # El stemId apunta a id.who.int; se consulta directamente.
        r2 = self._get(stem_id)
        if r2.status_code == 404:
            return ""
        r2.raise_for_status()
        return self._extract_title(r2.json())

    def _via_search(self, code: str) -> str:
        """Fallback: buscar el código como texto y tomar coincidencia exacta."""
        r = self._get(
            self.SEARCH_URL,
            params={
                "q":                    code,
                "useFlexisearch":       "false",
                "flatResults":          "true",
                "highlightingEnabled":  "false",
                "includeKeywordResult": "false",
            },
        )
        r.raise_for_status()
        for entity in (r.json().get("destinationEntities") or []):
            if (entity.get("theCode") or "").strip().upper() == code.upper():
                return self._extract_title(entity)
        return ""

    def lookup(self, code: str) -> str:
        """
        Resuelve un código ICD-11 a su título oficial.
        Retorna "" si no se pudo resolver.
        """
        code = (code or "").strip()
        if not code or not self.token:
            return ""

        with self._cache_lock:
            if code in self._cache:
                return self._cache[code]

        title = ""
        try:
            title = self._via_codeinfo(code)
            if not title:
                title = self._via_search(code)
        except Exception as e:
            print(f"\n    [WHO-lookup] {code}: {e}")

        with self._cache_lock:
            self._cache[code] = title
        return title


# ─────────────────────────────────────────────────────────────────────────────
# 4. EVALUACIÓN DE CONSISTENCIA (LLM)
# ─────────────────────────────────────────────────────────────────────────────

CONSISTENCY_SYSTEM_PROMPT = """Eres un auditor de codificación clínica.

Recibes un DIAGNÓSTICO clínico y el TÍTULO OFICIAL de la categoría ICD-11 que
se le asignó. Debes responder una sola pregunta:

    ¿El título de la categoría es coherente con el diagnóstico?

Contexto importante: los códigos fueron truncados a nivel de capítulo y
categoría, por lo que el título siempre será MÁS GENERAL que el diagnóstico.
Eso NO es un error.

Responde "yes" cuando:
- La categoría engloba correctamente al diagnóstico, aunque sea más amplia.
  Ej: diagnóstico "Amiloidosis AL" / título "Amiloidosis" → yes
  Ej: diagnóstico "Carcinoma hepatocelular" / título "Tumores malignos del
      hígado o de las vías biliares intrahepáticas" → yes
- El título es sinónimo o variante terminológica del diagnóstico.
- El título nombra la misma entidad clínica con otra denominación.

Responde "no" cuando:
- La categoría corresponde a un sistema, órgano o proceso patológico distinto.
  Ej: diagnóstico "Malacoplaquia renal" / título "Trastornos del esófago" → no
- La categoría comparte palabras con el diagnóstico pero designa otra entidad.
  Ej: diagnóstico "Fiebre Q" / título "Fiebre amarilla" → no
- El título es tan inespecífico que no aporta clasificación real.
  Ej: título "Ciertas afecciones especificadas" o "Otros trastornos" → no

Responde ÚNICAMENTE con una palabra: yes o no.
Sin comillas, sin puntuación, sin explicaciones."""


async def check_consistency(diagnosis: str, lookup: str, model) -> str:
    """
    Pregunta al LLM si el lookup es coherente con el diagnóstico.
    Retorna "yes" o "no". Ante error, retorna "" (fila queda sin evaluar).
    """
    chat = lms.Chat(CONSISTENCY_SYSTEM_PROMPT)
    chat.add_user_message(
        f'DIAGNÓSTICO: "{diagnosis}"\n'
        f'TÍTULO ICD-11: "{lookup}"\n\n'
        f'¿El título es coherente con el diagnóstico? Responde yes o no.'
    )

    raw = ""
    try:
        result = await model.respond(chat)
        raw    = _parse_channel_response(result.content).strip().lower()
        raw    = re.sub(r"[^a-z]", "", raw)

        if raw.startswith("yes"):
            return "yes"
        if raw.startswith("no"):
            return "no"

        print(f"\n    [LLM-consistency] Respuesta inesperada: {raw[:60]!r}")
        return ""
    except Exception as e:
        print(f"\n    [LLM-consistency] Error: {e} | raw: {raw[:100]}")
        return ""


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


async def run_consistency_phase(
    rows:      list,
    model_name: str,
    lms_host:  str,
    eval_nf:   bool,
    batch:     int = 10,
) -> dict:
    """
    Evalúa la coherencia diagnosis vs icd11_lookup en las filas dadas.
    Escribe la columna `consistency` in-place. Retorna estadísticas.
    """
    stats = {"yes": 0, "no": 0, "omitido_nf": 0, "sin_lookup": 0, "error": 0}

    pending = []
    for row in rows:
        code   = (row.get("icd11_code")   or "").strip()
        lookup = (row.get("icd11_lookup") or "").strip()

        # Códigos NF: no hubo etiquetado ICD-11 real que auditar.
        # Su lookup es el propio diagnóstico, así que compararlo daría
        # siempre "yes" e inflaría la métrica.
        if code.startswith("NF-") and not eval_nf:
            row["consistency"] = ""
            stats["omitido_nf"] += 1
            continue

        # Código que no resolvió a ninguna entidad: no puede ser coherente.
        if code and not lookup:
            row["consistency"] = "no"
            stats["sin_lookup"] += 1
            continue

        if not lookup or not (row.get("diagnosis_en") or "").strip():
            row["consistency"] = ""
            continue

        pending.append(row)

    if not pending:
        print("[LLM] No hay filas que evaluar.\n")
        return stats

    async with lms.AsyncClient(lms_host) as client:
        print(f"[LLM] Host: {lms_host}")
        print(f"[LLM] Conectando con modelo: {model_name}")
        try:
            model = await client.llm.model(model_name)
        except Exception as e:
            print(f"[LLM] ERROR al conectar con {lms_host}: {e}")
            print("[LLM] Verifica que LM Studio esté sirviendo en esa IP:puerto")
            print("      (Settings > Developer > Serve on Local Network).\n")
            stats["error"] = len(pending)
            return stats
        print("[LLM] Conexión exitosa\n")

        with tqdm(total=len(pending), desc="Evaluando coherencia", unit="fila") as pbar:
            for i in range(0, len(pending), batch):
                chunk   = pending[i:i + batch]
                verdicts = await asyncio.gather(*[
                    check_consistency(
                        (r.get("diagnosis_en") or "").strip(),
                        (r.get("icd11_lookup") or "").strip(),
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

def resolve_row(row: dict, who: WHOLookupClient, nf_labels: dict) -> str:
    """
    Determina el valor de icd11_lookup para una fila.
    Retorna la categoría del resultado, para estadísticas.
    """
    code = (row.get("icd11_code") or "").strip()

    if not code:
        row["icd11_lookup"] = ""
        return "sin_codigo"

    # Los códigos NF no existen en ICD-11: se resuelven localmente
    if code.startswith("NF-"):
        label = nf_labels.get(code, "")
        row["icd11_lookup"] = f"[NF] {label}" if label else "[NF] sin etiqueta"
        return "nf"

    title = who.lookup(code)
    if title:
        row["icd11_lookup"] = title
        return "resuelto"

    row["icd11_lookup"] = ""
    return "no_resuelto"


def run(csv_path: str, who: WHOLookupClient, overwrite: bool,
        max_rows: int | None, workers: int,
        model_name: str, lms_host: str,
        skip_consistency: bool, eval_nf: bool):
    rows_in, fieldnames = read_csv_robust(csv_path)

    if not rows_in:
        print(f"[CSV] {csv_path} no tiene filas de datos.\n")
        return

    nf_labels = load_nf_labels()
    print(f"[NF]  Diccionario cargado: {len(nf_labels)} códigos\n")

    # Selección de filas a procesar
    candidates = rows_in if max_rows is None else rows_in[:max_rows]
    if overwrite:
        pending = [r for r in candidates if (r.get("icd11_code") or "").strip()]
    else:
        pending = [
            r for r in candidates
            if (r.get("icd11_code") or "").strip()
            and not (r.get("icd11_lookup") or "").strip()
        ]

    total_con_codigo = sum(
        1 for r in candidates if (r.get("icd11_code") or "").strip()
    )
    print(f"[CSV] {len(rows_in)} filas | {total_con_codigo} con icd11_code | "
          f"{len(pending)} por resolver")
    if not overwrite and total_con_codigo > len(pending):
        print(f"[CSV] {total_con_codigo - len(pending)} ya tenían "
              f"icd11_lookup (usa --overwrite para rehacerlas)")
    print()

    # ── FASE 1: búsqueda inversa código → título ─────────────────────────
    stats = {"resuelto": 0, "nf": 0, "no_resuelto": 0, "sin_codigo": 0}

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(tqdm(
                pool.map(lambda r: resolve_row(r, who, nf_labels), pending),
                total=len(pending),
                desc="Resolviendo códigos",
                unit="fila",
            ))
        for outcome in results:
            stats[outcome] = stats.get(outcome, 0) + 1
    else:
        print("[CSV] Búsqueda inversa: nada por hacer.\n")

    # ── FASE 2: evaluación de coherencia con LLM ─────────────────────────
    cstats = None
    if not skip_consistency:
        if overwrite:
            to_eval = [r for r in candidates if (r.get("icd11_code") or "").strip()]
        else:
            to_eval = [
                r for r in candidates
                if (r.get("icd11_code") or "").strip()
                and not (r.get("consistency") or "").strip()
            ]

        if to_eval:
            print(f"\n[LLM] {len(to_eval)} filas a evaluar\n")
            cstats = asyncio.run(run_consistency_phase(
                to_eval, model_name, lms_host, eval_nf
            ))
        else:
            print("\n[LLM] Coherencia: nada por hacer "
                  "(usa --overwrite para reevaluar).\n")

    # ── Escritura única, preservando todas las columnas originales ───────
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_in:
            writer.writerow({k: (row.get(k) or "") for k in fieldnames})

    print(f"\n[CSV] Actualizado in-place: {csv_path}\n")

    # ── Reportes ─────────────────────────────────────────────────────────
    if pending:
        t = max(len(pending), 1)
        print("── Estadísticas de búsqueda inversa ────────────────────")
        print(f"  Procesados                 : {len(pending)}")
        print(f"  Resueltos (WHO API)        : {stats['resuelto']}    ({stats['resuelto']/t*100:.1f}%)")
        print(f"  NF (diccionario local)     : {stats['nf']}          ({stats['nf']/t*100:.1f}%)")
        print(f"  No resueltos               : {stats['no_resuelto']} ({stats['no_resuelto']/t*100:.1f}%)")
        print("────────────────────────────────────────────────────────\n")

        if stats["no_resuelto"]:
            print(f"[!] {stats['no_resuelto']} códigos no se pudieron resolver.")
            print("    Suelen ser códigos truncados que no existen como entidad")
            print("    propia en ICD-11, o códigos mal formados.\n")

    if cstats:
        evaluados = cstats["yes"] + cstats["no"]
        e = max(evaluados, 1)
        print("── Coherencia diagnosis_en vs icd11_lookup ─────────────")
        print(f"  Evaluados por el LLM       : {evaluados}")
        print(f"  Coherentes    (yes)        : {cstats['yes']} ({cstats['yes']/e*100:.1f}%)")
        print(f"  No coherentes (no)         : {cstats['no']}  ({cstats['no']/e*100:.1f}%)")
        if cstats["sin_lookup"]:
            print(f"  Marcados \"no\" sin lookup   : {cstats['sin_lookup']}")
        if cstats["omitido_nf"]:
            print(f"  Omitidos por ser NF        : {cstats['omitido_nf']} "
                  f"(usa --eval-nf para incluirlos)")
        if cstats["error"]:
            print(f"  Errores del LLM            : {cstats['error']}")
        print("────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CSV = str(Path(__file__).resolve().parent.parent / "dataset-run1.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Búsqueda inversa ICD-11: código → título oficial"
    )
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help=f"CSV a procesar in-place (default: {DEFAULT_CSV})")
    parser.add_argument("--who-client-id",
                        default=os.environ.get("WHO_CLIENT_ID", "00da56ea-0fe1-465e-b830-61cb0add2173_2741cb3a-0cd6-4977-af95-6258be8bd99a"),
                        help="Client ID WHO ICD-11 API (o env WHO_CLIENT_ID)")
    parser.add_argument("--who-client-secret",
                        default=os.environ.get("WHO_CLIENT_SECRET", "h1MrlyMGBnGt7Q6kAnSpq8/1s18FkkzPboT7MIaim7o="),
                        help="Client Secret WHO ICD-11 API (o env WHO_CLIENT_SECRET)")
    parser.add_argument("--lang", default="es",
                        help="Idioma de los títulos (default: es)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Peticiones concurrentes a la WHO API (default: 8)")
    parser.add_argument("--model", default="medgemma-27b-it",
                        help="Modelo en LM Studio para evaluar coherencia")
    parser.add_argument("--lms-host",
                        default=os.environ.get("LMS_HOST", "10.8.0.45:1234"),
                        help="Host:puerto del servidor LM Studio "
                             "(o env LMS_HOST). Default: localhost:1234")
    parser.add_argument("--skip-consistency", action="store_true",
                        help="Solo hacer la búsqueda inversa, sin evaluar coherencia")
    parser.add_argument("--eval-nf", action="store_true",
                        help="Evaluar también las filas con código NF "
                             "(por defecto se omiten: su lookup es el propio "
                             "diagnóstico y siempre daría \"yes\")")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rehacer las filas que ya tienen icd11_lookup/consistency")
    parser.add_argument("--rows", type=int, default=None,
                        help="Procesar solo las primeras N filas (modo prueba)")
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"[ERROR] Archivo no encontrado --csv: {args.csv}")
        return

    who = WHOLookupClient(args.who_client_id, args.who_client_secret, args.lang)
    if not who.authenticate():
        return

    run(args.csv, who, args.overwrite, args.rows, args.workers,
        args.model, args.lms_host, args.skip_consistency, args.eval_nf)


if __name__ == "__main__":
    main()
