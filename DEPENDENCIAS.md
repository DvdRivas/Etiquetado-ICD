# Dependencias y guía de reproducción

**Proyecto:** Etiquetado automático ICD-11 sobre corpus NEJM
**Entorno verificado:** Ubuntu (WSL2) · Anaconda · Python 3.13.5
**Fecha de verificación:** 10 de agosto de 2026

Este documento lista todo lo necesario para reejecutar el pipeline y el análisis. Las versiones indicadas son las **efectivamente instaladas en el entorno donde se produjeron las 10 corridas**, obtenidas con `importlib.metadata`, no estimadas.

---

## 1. Intérprete

| Requisito | Valor |
|---|---|
| Python | **3.13.5** (verificado) |
| Mínimo teórico | 3.10 — los scripts usan anotaciones `dict[k, v]`, `list[str]`, `tuple[list, list]` |
| Distribución usada | Anaconda (`/home/rivas/anaconda3`) |

---

## 2. Paquetes de terceros

### 2.1 Versiones verificadas

| Paquete | Versión instalada | Usado por | Función |
|---|---|---|---|
| `requests` | **2.32.3** | experiment, lookup | Llamadas HTTP a la WHO ICD-11 API |
| `urllib3` | **2.3.0** | experiment, lookup | Supresión de avisos TLS |
| `lmstudio` | **1.5.0** | experiment, lookup | SDK cliente de LM Studio (MedGemma) |
| `rapidfuzz` | **3.14.5** | experiment | Emparejamiento difuso contra el KGraph |
| `tqdm` | **4.67.1** | experiment, lookup | Barras de progreso |
| `numpy` | **2.1.3** | deep_analysis | Estadística agregada |
| `pandas` | **2.2.3** | deep_analysis | Carga y cruce de los 10 CSV |
| `tabulate` | **0.9.0** | deep_analysis | Tablas en consola |

### 2.2 `requirements.txt` — reproducción exacta

```
requests==2.32.3
urllib3==2.3.0
lmstudio==1.5.0
rapidfuzz==3.14.5
tqdm==4.67.1
numpy==2.1.3
pandas==2.2.3
tabulate==0.9.0
```

### 2.3 `requirements.txt` — versiones mínimas compatibles

Si la reproducción exacta no es necesaria:

```
requests>=2.31
urllib3>=2.0
lmstudio>=1.5
rapidfuzz>=3.0
tqdm>=4.65
numpy>=1.26
pandas>=2.0
tabulate>=0.9
```

> **Nota sobre `lmstudio`:** es la dependencia más sensible. El SDK cambió de API entre las series 0.x y 1.x. Los scripts usan `lms.Chat(...)`, `client.llm.model(...)` y `await model.respond(chat)`, propios de la serie 1.x. Con `lmstudio<1.0` los scripts fallan.

### 2.4 Instalación

```bash
# conda
conda create -n icd11 python=3.13
conda activate icd11
pip install -r requirements.txt

# o venv
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Biblioteca estándar (sin instalación)

Se listan para documentar el alcance del código, no requieren acción.

`os` · `sys` · `json` · `csv` · `re` · `argparse` · `asyncio` · `pathlib` · `datetime` · `collections` · `concurrent.futures` · `threading`

---

## 4. Servicios externos

### 4.1 LM Studio + MedGemma

| Parámetro | Valor usado |
|---|---|
| Modelo | `medgemma-27b-it` |
| Servidor | LM Studio en modo servidor |
| Host por defecto | `10.8.0.45:1234` |
| Variable de entorno | `LMS_HOST` |
| Flag CLI | `--lms-host IP:PUERTO` |
| Flag de modelo | `--model <nombre>` |

Requisitos: el modelo debe estar descargado y cargado en LM Studio, y el servidor local activo antes de lanzar los scripts. Ambos scripts verifican la conexión al inicio y abortan si falla.

**Parámetros de inferencia:** los scripts invocan `model.respond(chat)` **sin especificar `temperature`, `seed`, `top_p` ni `max_tokens`**, por lo que se aplican los valores por defecto de LM Studio. Esta es la causa de la no-determinación documentada en el análisis (2 casos de 201 con código variable entre corridas). Quien desee reproducción bit-a-bit debe fijar `temperature=0` y una semilla explícita, lo que modifica el código.

### 4.2 WHO ICD-11 API

| Parámetro | Valor |
|---|---|
| Endpoint base | `https://id.who.int/icd/release/11/2024-01/mms` |
| Release | **2024-01** (fijada, no `latest`) |
| Idioma | `es` (configurable con `--lang`) |
| Autenticación | OAuth2 client credentials |
| Variables de entorno | `WHO_CLIENT_ID`, `WHO_CLIENT_SECRET` |
| Flags CLI | `--who-client-id`, `--who-client-secret` |

Registro de credenciales: <https://icd.who.int/icdapi> (gratuito).

> ### ⚠ Advertencia de seguridad
>
> En la versión actual de los scripts, un `client_id` y un `client_secret` reales figuran **hardcodeados como valores por defecto** de `argparse`:
>
> - `icd-experiment.py`, líneas 1093 y 1096
> - `icd11-lookup.py`, líneas 592 y 595
>
> Antes de compartir, publicar o subir estos scripts a cualquier repositorio:
>
> 1. Sustituir esos defaults por `""` y dejar únicamente la lectura de `os.environ`.
> 2. **Rotar el `client_secret`** en el portal de la WHO. Ha estado en texto plano en disco y en el historial de git, por lo que debe considerarse comprometido.
> 3. Verificar que ningún commit anterior lo contenga (`git log -p -S "client_secret"`).

Configuración recomendada mediante `.env` (no versionado):

```bash
# .env — NO subir a git
WHO_CLIENT_ID=tu_client_id
WHO_CLIENT_SECRET=tu_client_secret
LMS_HOST=10.8.0.45:1234
```

Cargar con `set -a && source .env && set +a` antes de ejecutar. Los scripts **no** usan `python-dotenv`; leen directamente de `os.environ`.

---

## 5. Archivos de datos

| Archivo | Tamaño | Origen | Obligatorio |
|---|---|---|---|
| `en_product1.json` | 37 MB | Orphanet — *product1*, nomenclatura en inglés con mapeo a ICD-11 | **Sí** (`icd-experiment.py`) |
| `es_product1.json` | 39 MB | Orphanet — equivalente en español | No (presente, no usado por defecto) |
| `dataset-raw.csv` | 225 KB | Corpus base: 201 casos NEJM | **Sí** — entrada del pipeline |
| `nf_dictionary.json` | 6 KB | Generado por el pipeline | Se crea solo si no existe |

**Orphanet `product1`:** descargable desde <https://www.orphadata.com>. Contiene 11 645 trastornos, de los cuales 6 553 tienen mapeo a ICD-11 (26 908 entradas totales incluyendo sinónimos). El pipeline lo carga en memoria como grafo de conocimiento al inicio.

**`nf_dictionary.json`:** persiste entre ejecuciones y asigna identificadores secuenciales `NF-0001`, `NF-0002`… a los diagnósticos no encontrados.

> **Importante para la reproducción:** este archivo es estado compartido entre corridas. Para reproducir corridas verdaderamente independientes debe **borrarse o vaciarse antes de cada ejecución**. Si se conserva, 25 de los 201 casos quedan fijados como NF y los intervalos de confianza de cobertura subestiman la varianza real.

---

## 6. Estructura de directorios esperada

Las rutas por defecto son relativas a la ubicación de los scripts. Debe respetarse esta disposición o pasar rutas explícitas por CLI.

```
Etiquetado ICD11/
├── dataset-raw.csv              # corpus de entrada
├── dataset-run1.csv … run10.csv # salidas por corrida
├── METODOLOGIA.md
├── DEPENDENCIAS.md              # este documento
└── DrChatPatinEval/
    ├── icd-experiment.py        # fase 1 — asignación
    ├── icd11-lookup.py          # fase 2 — resolución + evaluación
    ├── deep_analysis.py         # análisis de las 10 corridas
    ├── en_product1.json         # KGraph Orphanet
    ├── nf_dictionary.json       # estado NF (generado)
    ├── deep_analysis_report.json # salida del análisis
    └── unstable_cases.csv        # salida del análisis
```

Rutas por defecto codificadas:

- `DEFAULT_CSV` = `<parent_dir>/dataset-run1.csv`
- `NF_DICT_PATH` = `<script_dir>/nf_dictionary.json`
- `--product1` = `en_product1.json` (relativo al directorio de trabajo)

---

## 7. Secuencia de ejecución

Las dos fases son secuenciales: la segunda consume las columnas que produce la primera, sobre el **mismo CSV, modificado in-place**.

```bash
cd "Etiquetado ICD11/DrChatPatinEval"

# Reproducción independiente: limpiar estado NF
rm -f nf_dictionary.json

# Fase 1 — asignación de código
python icd-experiment.py --csv ../dataset-run1.csv

# Fase 2 — resolución de título + evaluación de coherencia
python icd11-lookup.py --csv ../dataset-run1.csv

# Repetir para runs 2..10, luego:
python deep_analysis.py
```

### 7.1 Argumentos — `icd-experiment.py`

| Flag | Default | Descripción |
|---|---|---|
| `--csv` | `../dataset-run1.csv` | CSV a procesar in-place |
| `--product1` | `en_product1.json` | Ruta al JSON de Orphanet |
| `--model` | `medgemma-27b-it` | Modelo en LM Studio |
| `--lms-host` | `10.8.0.45:1234` | Host del servidor LM Studio |
| `--who-client-id` | *(env)* | Credencial WHO |
| `--who-client-secret` | *(env)* | Credencial WHO |
| `--threshold` | `85` | Umbral de similitud para match difuso |
| `--rows` | `None` | Limitar número de filas (pruebas) |

### 7.2 Argumentos — `icd11-lookup.py`

| Flag | Default | Descripción |
|---|---|---|
| `--csv` | `../dataset-run1.csv` | CSV a procesar in-place |
| `--model` | `medgemma-27b-it` | Modelo en LM Studio |
| `--lms-host` | `10.8.0.45:1234` | Host del servidor LM Studio |
| `--lang` | `es` | Idioma del título oficial ICD-11 |
| `--workers` | `8` | Hilos para las llamadas a la WHO API |
| `--who-client-id` | *(env)* | Credencial WHO |
| `--who-client-secret` | *(env)* | Credencial WHO |
| `--skip-consistency` | `False` | Omitir la evaluación con LLM |
| `--eval-nf` | `False` | Incluir códigos NF en la evaluación |
| `--overwrite` | `False` | Recalcular filas ya resueltas |
| `--rows` | `None` | Limitar número de filas |

### 7.3 Argumentos — `deep_analysis.py`

| Flag | Default | Descripción |
|---|---|---|
| `--runs-dir` | directorio padre | Ubicación de `dataset-run{1..N}.csv` |
| `--n-runs` | `10` | Número de corridas a analizar |

---

## 8. Notas de compatibilidad

**Encoding.** Los CSV originales presentan codificación mixta cp1252/UTF-8. `icd-experiment.py` detecta e informa la mezcla y normaliza todo a UTF-8 al guardar. `deep_analysis.py` intenta UTF-8, cp1252 y latin-1 en ese orden. No requiere intervención.

**TLS.** Ambos scripts importan `urllib3` para suprimir avisos de certificado en las llamadas a la WHO API.

**Concurrencia.** `icd11-lookup.py` usa `ThreadPoolExecutor` (8 hilos por defecto) para la resolución de títulos y `asyncio` para las llamadas al LLM. La fase de asignación es secuencial.

**Sin GPU requerida por los scripts.** El cómputo del modelo ocurre en el servidor de LM Studio, que puede residir en otra máquina. Los scripts son clientes HTTP.

---

## 9. Tiempos de referencia

Medidos en el entorno descrito, 201 filas por corrida:

| Etapa | Duración observada |
|---|---|
| `icd-experiment.py` (completo) | 2 min 30 s – 5 min |
| `icd11-lookup.py` — resolución de títulos | ~8 s (22–25 filas/s) |
| `icd11-lookup.py` — evaluación de coherencia | 42 s – 1 min 26 s (2.0–4.1 filas/s) |
| `deep_analysis.py` | < 5 s |

La variación entre corridas obedece a la carga del servidor de inferencia, no al volumen de datos.

---

## 10. Verificación del entorno

Comprobación rápida antes de ejecutar:

```bash
python3 -c "
import importlib.metadata as md
for p in ['requests','urllib3','lmstudio','rapidfuzz','tqdm','numpy','pandas','tabulate']:
    try:
        print(f'{p:12} {md.version(p)}')
    except Exception:
        print(f'{p:12} FALTA')
"
```

Comprobación de servicios:

```bash
# LM Studio accesible
curl -s http://10.8.0.45:1234/v1/models | head -c 200

# Credenciales WHO definidas
[ -n "$WHO_CLIENT_ID" ] && echo "WHO_CLIENT_ID definido" || echo "WHO_CLIENT_ID FALTA"
```

---

## 11. Puntos abiertos

Si algo de lo siguiente resulta relevante para quien reproduzca el trabajo, conviene aclararlo antes:

1. **`es_product1.json` no se usa.** Está presente en el directorio pero el default es `en_product1.json`. Si la intención era usar la nomenclatura española del KGraph, es una discrepancia a resolver.
2. **Versión de Orphanet product1.** El archivo no lleva marca de versión ni fecha de descarga. Orphanet publica actualizaciones periódicas; una descarga posterior puede alterar los resultados de las ramas `exact_*` y `fuzzy_*`. Conviene archivar el JSON exacto junto con el código.
3. **Determinismo.** Mientras no se fijen `temperature` y semilla, ninguna reejecución reproducirá exactamente las cifras publicadas. La diferencia esperada es de 1–2 casos sobre 201.
