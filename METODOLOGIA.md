# Metodología: Etiquetado ICD-11 automatizado y evaluación de coherencia

## 1. Objetivo

Evaluar la capacidad de un sistema automatizado para asignar códigos ICD-11 a
diagnósticos clínicos, y medir la coherencia del etiquetado resultante mediante
una verificación por búsqueda inversa.

La hipótesis operativa es que un código correctamente asignado, al resolverse de
vuelta a su título oficial en la clasificación, debe describir una entidad
clínicamente compatible con el diagnóstico de origen.

---

## 2. Corpus

**Fuente**: 201 casos clínicos del *New England Journal of Medicine*, en dos series:

| Serie | Prefijo | Descripción |
|---|---|---|
| Case Records of the MGH | `NEJMcpc` | Casos clínico-patológicos con diagnóstico final confirmado |
| Images in Clinical Medicine | `NEJMicm` | Casos ilustrados con diagnóstico establecido |

**Criterio de inclusión**: cada PDF aporta exactamente un caso con diagnóstico
final explícito. Los casos sin diagnóstico confirmado se excluyen.

---

## 3. Construcción del dataset

Cada caso se convierte en una fila con las siguientes columnas:

| Columna | Contenido |
|---|---|
| `filename` | PDF de origen (trazabilidad) |
| `diagnosis_es` | Diagnóstico final en español |
| `diagnosis_en` | Traducción a terminología médica estándar en inglés |
| `core_diagnosis` | Entidad codificable extraída (ver §4.1) |
| `clinical_summary` | Resumen clínico narrado experto-a-experto |
| `icd11_code` | Código asignado, truncado a capítulo + categoría |
| `icd11_lookup` | Título oficial recuperado por búsqueda inversa |
| `match_type` | Paso de la cascada que resolvió la asignación |
| `match_source` | Variante de consulta que produjo el match |
| `consistency` | Veredicto de coherencia (`yes` / `no`) |

### 3.1 Diagnóstico

Se registra el diagnóstico final del caso, no el diferencial discutido en el
desarrollo. Se incluye un **elemento diferenciador de la causa únicamente
cuando es necesario** para distinguir entidades homónimas —por ejemplo,
diabetes gestacional frente a diabetes mellitus tipo 2—. Cuando el nombre no es
ambiguo, no se agregan especificadores.

### 3.2 Resumen clínico

Narración en prosa, en registro experto-a-experto, como en una solicitud de
segunda opinión. Incluye datos demográficos, presentación, hallazgos de
exploración, laboratorio e imagen, evolución y desenlace.

**El resumen no menciona el diagnóstico final**, para no contaminar
evaluaciones posteriores que pudieran usar este campo como entrada.

### 3.3 Idioma de trabajo

El matching opera sobre `diagnosis_en`. Justificación:

- ICD-11 se redacta originalmente en inglés; las traducciones tienen menos
  términos y sinónimos indexados, tanto en la API de la OMS como en Orphanet.
- La versión inglesa de Orphanet (`en_product1.json`) cubre 11 645 trastornos
  con 6 553 mapeados a ICD-11, frente a 11 456 / 6 143 de la versión española.

`diagnosis_es` se conserva como referencia legible y como término de
comparación en la evaluación de coherencia.

> **Limitación reconocida**: al traducir, el traductor pasa a formar parte del
> sistema evaluado. Un error de traducción se contabilizaría como error de
> etiquetado. Se mitigó usando terminología médica estándar (no traducción
> literal) y conservando ambas columnas para auditoría.

---

## 4. Asignación de código (proceso directo)

Implementado en `icd-experiment.py`.

### 4.1 Paso 0 — Extracción del núcleo codificable

Los diagnósticos clínicos suelen presentarse como entidad + causa + contexto:

```
Acute intermittent porphyria due to HMBS mutation
└────── entidad codificable ──────┘└─── causa ───┘
```

La frase completa no coincide con ninguna entrada de la ontología, aunque la
entidad sí exista en ella. Se verificó empíricamente:

| Consulta | Resultado en KGraph |
|---|---|
| `Acute intermittent porphyria` | match exacto → `5C58.1Y` |
| `Acute intermittent porphyria due to HMBS mutation` | sin match |
| `Takayasu arteritis` | match exacto → `4A44.1` |
| `Pediatric Takayasu arteritis` | sin match |

Para resolverlo se extrae el **núcleo codificable** mediante un modelo de
lenguaje local (medgemma vía LM Studio). El prompt especifica qué eliminar
(agente causal, fármaco desencadenante, contexto clínico, comorbilidad,
circunstancia) y qué conservar (subtipo de la enfermedad, sitio anatómico,
calificadores intrínsecos, epónimos completos).

**Filtro previo**: solo se invoca al modelo cuando la frase contiene conectores
causales o contextuales (`due to`, `secondary to`, `in`, `with`, `following`,
`-induced`, etc.). En el corpus actual, 142 de 201 filas requieren la llamada;
59 se procesan sin coste adicional.

**Salvaguardas**: si la respuesta es vacía, más larga que la entrada o menor a
tres caracteres, se descarta y se conserva la frase original.

### 4.2 Cascada de asignación

Se prueban en orden dos variantes de consulta —**frase completa** y
**núcleo**— con precedencia siempre para la completa, por ser más específica.

| # | Paso | Fuente | Variantes | `match_type` |
|---|---|---|---|---|
| 1 | Caché | memoria | — | — |
| 2 | Match exacto | KGraph Orphanet | completa → núcleo | `exact_complete` / `exact_core` |
| 3 | Match difuso ≥85 + validación LLM | KGraph Orphanet | completa → núcleo | `fuzzy_complete` / `fuzzy_core` |
| 4 | Búsqueda + selección LLM | WHO ICD-11 API | pool unificado | `who_search` / `who_autocode` |
| 5 | No encontrado | — | — | `nf` |

**Paso 3** usa `rapidfuzz` con `token_sort_ratio` y umbral 85. Todo match difuso
pasa por validación semántica del LLM antes de aceptarse: la similitud textual
no garantiza equivalencia clínica.

**Paso 4** consulta `/search` con ambas variantes, une los candidatos en un
único conjunto deduplicado por código y deja que el LLM seleccione. Si no hay
resultados, recurre a `/autocode`.

### 4.3 Principio de separación recuperación / selección

> El núcleo amplía la **recuperación** de candidatos.
> La frase completa gobierna la **selección** del candidato correcto.

La validación semántica se realiza **siempre contra la frase completa**, incluso
cuando el candidato provino de la consulta con núcleo. Esto preserva la
información causal en la decisión final, lo cual es necesario porque la causa
puede determinar una categoría ICD-11 distinta:

| Diagnóstico | Categoría correcta | Categoría si se ignora la causa |
|---|---|---|
| Diabetes inducida por pembrolizumab | Diabetes inducida por fármacos | Diabetes tipo 2 |
| Ginecomastia inducida por espironolactona | Efecto adverso de fármaco | Ginecomastia idiopática |
| Linfohistiocitosis hemofagocítica secundaria a COVID-19 | HLH secundaria | HLH primaria |

Descartar la causa por completo aumentaría la cobertura a costa de introducir
**errores silenciosos**: un código no asignado es visible; un código incorrecto
no lo es.

### 4.4 Truncamiento de códigos

Todo código se reduce a **capítulo + categoría**, descartando lo que sigue al
punto decimal:

```
5A11.2  → 5A11
BA00.0Z → BA00
1C62    → 1C62
```

Justificación: la evaluación mide si el sistema ubica correctamente la entidad
en la taxonomía, no si acierta el nivel máximo de especificidad.

### 4.5 Diagnósticos no encontrados (NF)

Los diagnósticos que agotan la cascada reciben un identificador interno
secuencial `NF-XXXX`, persistido en `nf_dictionary.json`. Estos códigos:

- no pertenecen a ICD-11 y se excluyen de las métricas de coherencia;
- permiten cuantificar la cobertura del sistema;
- conservan la etiqueta en español para revisión manual.

---

## 5. Búsqueda inversa (proceso inverso)

Implementado en `icd11-lookup.py`. Proceso **determinista, sin LLM**.

Para cada `icd11_code` se recupera el título oficial de la entidad:

1. `GET /codeinfo/{code}` → devuelve el `stemId` (URI de la entidad)
2. `GET {stemId}` → devuelve el título oficial
3. Fallback: `/search` con coincidencia exacta de `theCode`

Los códigos `NF-XXXX` se resuelven contra el diccionario local y se marcan con
el prefijo `[NF]` para distinguirlos.

El resultado se escribe en `icd11_lookup`.

> **Nota operativa**: esta columna debe regenerarse (`--overwrite`) cada vez que
> se reasignan códigos. De lo contrario queda desincronizada respecto a
> `icd11_code` y cualquier medición posterior carece de validez.

---

## 6. Evaluación de coherencia

Segunda fase de `icd11-lookup.py`. Un modelo de lenguaje local responde una
pregunta binaria por cada fila:

> ¿El título oficial recuperado es coherente con el diagnóstico original?

Resultado: `yes` o `no` en la columna `consistency`.

### 6.1 Criterio de juicio

El criterio es **flexible respecto a la especificidad**, porque el truncamiento
garantiza que el título sea más general que el diagnóstico. Esa diferencia de
granularidad no constituye error.

**Se considera coherente (`yes`)** cuando:
- la categoría engloba correctamente al diagnóstico, aunque sea más amplia
  (`Amiloidosis AL` → *Amiloidosis*);
- el título es sinónimo o variante terminológica de la entidad.

**Se considera incoherente (`no`)** cuando:
- la categoría corresponde a otro sistema, órgano o proceso patológico;
- el título comparte vocabulario pero designa otra entidad
  (`Fiebre Q` → *Fiebre amarilla*);
- el título es tan inespecífico que no aporta clasificación real
  (*Otros trastornos especificados*).

### 6.2 Tratamiento de casos límite

| Situación | Tratamiento | Justificación |
|---|---|---|
| Código `NF-XXXX` | `consistency` vacío | No hubo etiquetado ICD-11 real que auditar; su lookup es el propio diagnóstico y produciría `yes` sistemáticamente, inflando la métrica |
| Código sin título resoluble | `consistency = no` | Un código que no apunta a ninguna entidad no puede ser coherente |
| Error del modelo | `consistency` vacío | Se reporta por separado; no contamina la métrica |

---

## 7. Métricas

### 7.1 Cobertura

```
Cobertura = (filas con código ICD-11) / (total de filas)
```

Mide qué proporción del corpus el sistema logra ubicar en la clasificación.
El complemento son los casos `NF`.

### 7.2 Coherencia

```
Coherencia = (veredictos "yes") / (veredictos "yes" + "no")
```

Calculada únicamente sobre filas con código ICD-11 real. Mide la calidad de las
asignaciones efectuadas.

> Cobertura y coherencia son métricas **independientes y en tensión**. Un
> sistema permisivo aumenta cobertura degradando coherencia. Deben reportarse
> siempre en conjunto.

### 7.3 Atribución del efecto

La columna `match_source` permite cuantificar el aporte de la extracción de
núcleo:

| Valor | Interpretación |
|---|---|
| `complete` | Resuelto por la frase completa (comportamiento base) |
| `core` | Resuelto por el núcleo — **ganancia neta del método** |
| `none` | No resuelto |

La columna `core_diagnosis` permite distinguir dos modos de fallo distintos:
extracción incorrecta del núcleo frente a ausencia real de la entidad en la
clasificación.

### 7.4 Distribución por paso

`match_type` permite reportar qué proporción resolvió cada fuente (KGraph
exacto, KGraph difuso, WHO API), útil para caracterizar la contribución de cada
recurso al desempeño global.

---

## 8. Diseño experimental

El corpus se procesa en cinco corridas independientes (`dataset-run1.csv` …
`dataset-run5.csv`) para evaluar la **estabilidad** de las asignaciones. Los
pasos que involucran al LLM son no deterministas, por lo que la varianza entre
corridas es en sí misma un resultado: un sistema que asigna códigos distintos
al mismo diagnóstico en corridas sucesivas no es confiable, aunque su
coherencia promedio sea alta.

---

## 9. Herramientas

| Componente | Implementación |
|---|---|
| Modelo de lenguaje | medgemma-27b-it, servido localmente vía LM Studio |
| Ontología de enfermedades raras | Orphanet `en_product1.json` |
| Clasificación de referencia | WHO ICD-11 API, release 2024-01, linearización MMS |
| Coincidencia difusa | `rapidfuzz` (`token_sort_ratio`) |
| Traducción es→en | Revisión asistida con terminología médica estándar |

Todo el procesamiento con modelos de lenguaje se ejecuta **localmente**, sin
envío de datos a servicios externos más allá de las consultas terminológicas a
la API de la OMS.

---

## 10. Limitaciones

1. **El traductor forma parte del sistema evaluado.** Los errores de traducción
   son indistinguibles de los errores de etiquetado en las métricas actuales.

2. **La extracción de núcleo puede sobre-generalizar.** Si el modelo reduce
   `Vascular Ehlers-Danlos syndrome` a `Ehlers-Danlos syndrome`, gana cobertura
   perdiendo precisión. La columna `core_diagnosis` existe para auditar esto,
   pero requiere revisión manual.

3. **El juez es un modelo de lenguaje.** La evaluación de coherencia no ha sido
   validada contra criterio humano experto. Sin ese contraste, la métrica mide
   el acuerdo del modelo consigo mismo.

4. **El detector de conectores es heurístico.** Diagnósticos con calificadores
   antepuestos (`Pediatric Takayasu arteritis`, `Late-onset multiple
   sclerosis`) no activan la extracción de núcleo y siguen fallando.

5. **Cobertura de Orphanet.** Es una base de enfermedades raras; la mayoría de
   los casos del corpus se resuelven vía la API de la OMS, no vía el KGraph.

6. **Corpus de un solo origen.** Los casos del NEJM están redactados con un
   estilo particular y sesgados hacia presentaciones inusuales. Los resultados
   no son extrapolables directamente a notas clínicas de rutina.
