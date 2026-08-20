# Respuesta a las observaciones metodológicas sobre el pipeline de etiquetado ICD-11

**Proyecto:** Etiquetado automático ICD-11 sobre corpus NEJM (201 casos)
**Fecha:** 6 de agosto de 2026
**Alcance:** Este documento responde exclusivamente a las dudas y objeciones planteadas en los documentos *Parte 1* y *Parte 2*. Los scripts (`icd-experiment.py`, `icd11-lookup.py`) se entregan por separado; aquí se explica el **porqué** de las decisiones, no el **cómo** de la implementación.

---

## 0. Resumen ejecutivo

De las observaciones planteadas, la evaluación honesta es la siguiente:

| # | Observación | Veredicto |
|---|---|---|
| 1 | La regla de selección de qué codificar no está documentada | **Válida.** La regla existe y es explícita en el prompt, pero no fue elevada a la documentación metodológica. Se corrige. |
| 2 | `core_diagnosis` elimina información antes de codificar | **Parcialmente inexacta.** No es recorte de texto sino normalización de entidad, y la frase completa se prueba *primero*. Se aporta evidencia cuantitativa. |
| 3 | `consistency = yes` no demuestra que el código sea completo ni óptimo | **Enteramente válida y ya asumida por diseño.** El error está en cómo se ha reportado la métrica, no en la métrica misma. Se propone renombrarla. |
| 4 | El mismo modelo participa en selección y evaluación | **Válida.** Es la limitación más seria del diseño actual. Se aportan mediciones que acotan su magnitud, pero no la eliminan. |

Adicionalmente, la revisión interna motivada por estas observaciones detectó **tres defectos no señalados en los documentos** que se declaran en la §7: ausencia de control de determinismo en el LLM, pérdida irreversible del código pre-truncamiento, y no-independencia entre corridas por estado compartido.

---

## 1. La pregunta central: ¿qué unidad se está codificando?

> *"¿El objetivo del pipeline es asignar un solo código a la condición principal de cada caso, o representar todas las entidades que integran el Final Diagnosis del NEJM?"*

**Respuesta: un solo código a la condición focal principal.** No se pretende representar el diagnóstico compuesto.

### 1.1 Justificación

El objeto de evaluación de este pipeline no es la codificación clínica de un episodio asistencial, sino la **capacidad diagnóstica de DrChatPatin**. El sistema evaluado emite un diagnóstico único por caso; el etiquetado ICD-11 existe para hacer comparables esos diagnósticos contra una referencia normalizada, no para producir un registro de codificación apto para facturación o estadística sanitaria.

Esto tiene tres consecuencias que fijan el diseño:

1. **El corpus lo permite.** Los *Case Records of the Massachusetts General Hospital* del NEJM están construidos precisamente para converger en un diagnóstico final único. La comorbilidad aparece como contexto que explica la presentación, no como entidad co-principal del ejercicio.

2. **La unidad de comparación debe ser conmensurable.** Si el pipeline emitiera conjuntos de códigos de cardinalidad variable, comparar la salida de DrChatPatin (un diagnóstico) contra la referencia (n códigos) exigiría métricas de solapamiento parcial —Jaccard, F1 por conjunto— que introducen decisiones arbitrarias sobre ponderación y que no responden a la pregunta de investigación.

3. **La codificación de episodio completo es otra tarea.** Requiere reglas de secuenciación de diagnóstico principal, criterios de "condición presente al ingreso", y un gold standard producido por codificadores certificados. Es un proyecto distinto, no una extensión de este.

### 1.2 La regla sí está implementada explícitamente

La objeción de que la regla "no está documentada" es correcta respecto de la documentación, pero no respecto del sistema. La instrucción que gobierna la selección es explícita y textual:

> *"It must be the disease actually present in the patient, not the trigger, the comorbidity or the setting."*
> *"Prefer the most specific disease name that is a real classification entry."*

Es decir: ante `MASH cirrhosis + Hepatocellular carcinoma`, la regla ordena retener la enfermedad presente —el carcinoma— y descartar el contexto etiológico —la cirrosis— exactamente como se observó. El comportamiento no es emergente ni accidental; es la regla ejecutándose.

**Acción correctiva:** esta regla se incorpora textualmente a `METODOLOGIA.md` como criterio de inclusión/exclusión declarado *a priori*.

### 1.3 Delimitación honesta del resultado

Se acepta sin reservas la formulación del documento *Parte 1*:

- **Correcto** para: codificar el carcinoma hepatocelular como condición focal principal.
- **Incompleto** para: representar el diagnóstico compuesto *"MASH cirrhosis complicated by hepatocellular carcinoma"*.

Lo segundo nunca fue el objetivo, pero al no haberse declarado, la crítica es procedente.

---

## 2. Sobre `core_diagnosis`: normalización, no recorte

La observación afirma que el pipeline "transforma *Hepatocellular carcinoma arising in MASH cirrhosis* en *Hepatocellular carcinoma*, y después encuentra el código a partir de esta versión reducida". Esto requiere dos precisiones.

### 2.1 No es eliminación de palabras; es reidentificación de entidad

La instrucción al modelo prohíbe explícitamente el recorte textual:

> *"Use your medical knowledge to name the entity; do not merely delete words from the input. The standard name often uses vocabulary that never appears in the input."*

Los ejemplos que gobiernan la tarea lo confirman —en varios casos el nombre resultante **introduce vocabulario ausente** en la entrada:

| Entrada | `core_diagnosis` |
|---|---|
| Cryptococcus neoformans meningoencephalitis | Cryptococcosis |
| Amoxicillin-induced rash in EBV infectious mononucleosis | Drug eruption |
| Spur cell hemolytic anemia in advanced alcoholic cirrhosis | Acquired haemolytic anaemia |
| Jejunal variceal bleeding secondary to noncirrhotic portal hypertension | Bleeding intestinal varices |

Ninguna de estas transformaciones es obtenible borrando tokens. La operación es de mapeo léxico a la nomenclatura de clasificación, que es el paso que un codificador humano ejecuta mentalmente antes de abrir el índice alfabético.

El caso del carcinoma hepatocelular es engañoso porque en él la normalización *coincide* con un recorte. Es la excepción, no el mecanismo.

### 2.2 La frase completa se prueba primero, y resuelve la mayoría de los casos

La cascada intenta el emparejamiento con la **frase diagnóstica íntegra** antes de recurrir al nombre normalizado. La columna `match_source` registra cuál de las dos vías produjo el resultado. Distribución observada en las 10 corridas:

| Corrida | `complete` | `core` | `none` (NF) |
|---|---|---|---|
| 1 | 104 | 70 | 27 |
| 2 | 103 | 71 | 27 |
| 3 | 105 | 70 | 26 |
| 4 | 103 | 70 | 28 |
| 5 | 103 | 70 | 28 |
| 6 | 105 | 70 | 26 |
| 7 | 104 | 70 | 27 |
| 8 | 106 | 69 | 26 |
| 9 | 103 | 71 | 27 |
| 10 | 102 | 72 | 27 |
| **Media** | **103.8 (59.7 %)** | **70.3 (40.3 %)** | **26.9** |

**El 59.7 % de los casos resueltos se codifican a partir de la frase diagnóstica completa, sin intervención de `core_diagnosis`.** La versión normalizada actúa como mecanismo de recuperación cuando la frase literal no tiene entrada en la clasificación —que es su función declarada.

Por tanto la caracterización "reduce y luego codifica" no describe el flujo real. La descripción exacta es: *intenta la frase completa; si falla, intenta el nombre normalizado; si falla, marca NF*.

### 2.3 Lo que sí se concede

- La información descartada **no se pierde del dataset** —`diagnosis_es` y `diagnosis_en` conservan la frase original íntegra en todas las filas—, pero **sí se pierde para el emparejamiento** en el 40.3 % de casos que caen a la vía `core`.
- En esos casos la comorbilidad no queda representada en ningún código. El resultado es, en la terminología del documento, **parcial**, no incorrecto.
- No existe actualmente una columna que registre qué entidades fueron descartadas al normalizar. Sería trivial añadirla y haría auditable esta pérdida. Se acepta como recomendación.

---

## 3. Qué mide y qué no mide `consistency`

Esta objeción se acepta íntegramente. Es correcta y, además, el diseño ya la contemplaba —el problema es que la métrica se ha estado **reportando bajo un nombre que promete más de lo que mide**.

### 3.1 La definición operativa es explícita

El criterio entregado al evaluador declara de antemano que la generalidad no es error:

> *"Contexto importante: los códigos fueron truncados a nivel de capítulo y categoría, por lo que el título siempre será MÁS GENERAL que el diagnóstico. Eso NO es un error."*

Y fija como positivo el caso exacto que motiva la objeción:

> *"Ej: diagnóstico 'Carcinoma hepatocelular' / título 'Tumores malignos del hígado o de las vías biliares intrahepáticas' → yes"*

### 3.2 Enunciado exacto de lo que la métrica significa

`consistency = yes` significa, y solo significa:

> La categoría ICD-11 asignada **no contradice** al diagnóstico: pertenece al mismo sistema/proceso patológico y lo subsume.

**No significa** —y aquí se suscribe punto por punto el documento *Parte 1*—:

- que sea el código más específico disponible;
- que la comorbilidad esté codificada;
- que el diagnóstico compuesto esté representado;
- que un especialista haya confirmado el código.

Es una **condición necesaria, no suficiente**, para la corrección. Funciona como *detector de errores groseros*, no como certificado de exactitud. Los criterios negativos del prompt confirman ese alcance: rechaza discordancia de sistema orgánico, confusión por homonimia y categorías vacuas del tipo *"Otros trastornos especificados"*.

### 3.3 Consecuencia para el reporte de resultados

La equiparación de `consistency` con *precisión* en las tablas agregadas es indefendible. Se corrige la nomenclatura:

| Nombre actual | Nombre correcto |
|---|---|
| Precisión / Coherencia | **Tasa de compatibilidad categorial** (o *tasa de no-contradicción*) |

Y se acompaña de la advertencia: *es un límite superior de la exactitud real, no una estimación de ella.* El valor observado —91.0 % [90.8 %, 91.3 %]— debe leerse como "en el 9 % de los casos el sistema comete un error tan evidente que su propio evaluador lo detecta", no como "el sistema acierta el 91 % de las veces".

### 3.4 Sobre el gold standard

No existe actualmente referencia validada por codificadores humanos. Esta es la carencia que impide convertir la compatibilidad categorial en exactitud. La vía de corrección es una submuestra estratificada —del orden de 50 casos, con sobrerrepresentación de los `who_search` y de los 16 casos sistemáticamente incoherentes— codificada a doble ciego por dos revisores clínicos, con reporte de kappa de Cohen. Sin ese anclaje, ninguna cifra de este trabajo puede presentarse como exactitud diagnóstica.

---

## 4. Riesgo de autoevaluación: el mismo modelo es juez y parte

Esta observación es correcta y constituye la debilidad más seria del diseño. MedGemma-27B interviene en cuatro puntos: generación de `core_diagnosis`, validación de candidatos difusos, selección entre candidatos de la WHO API, y evaluación final de `consistency`.

No se puede refutar. Sí se puede **acotar empíricamente**, y las 10 corridas permiten hacerlo.

### 4.1 El evaluador no ratifica sistemáticamente

Si operara como sello de goma, la tasa de rechazo tendería a cero. No es lo que se observa: **rechaza 15–16 de 174 asignaciones por corrida (≈ 9.2 %)**, y en todos los casos rechaza asignaciones producidas por él mismo. Existe capacidad de autocontradicción.

### 4.2 Los errores son sistemáticos, no ruido complaciente

El análisis por caso a través de las 10 corridas arroja:

| Categoría | Casos | % |
|---|---|---|
| Estable y compatible | 154 | 76.6 % |
| **Estable e incompatible** | **16** | **8.0 %** |
| Código inestable entre corridas | 2 | 1.0 % |
| NF persistente | 25 | 12.4 % |
| NF intermitente | 4 | 2.0 % |

Los **16 casos "estable e incompatible"** son diagnósticos donde el modelo asigna el mismo código en las 10 corridas y en las 10 lo rechaza. Es un error reproducible y auditable, no una fluctuación. Constituyen la cola de fallo real del sistema y son el primer lote que debe ir a revisión humana.

### 4.3 La inestabilidad del juez es medible y baja

Casos con código idéntico en las 10 corridas pero veredicto variable: **exactamente 1 de 201 (0.5 %)**.

| Caso | Código | Veredicto |
|---|---|---|
| `NEJMcpc2412521.pdf` — Reninoma | `2F98&XA6KU8` | `yes` en 4/10, `no` en 6/10 |

Un único caso limítrofe. La varianza del evaluador es, por tanto, despreciable frente a la magnitud de las métricas reportadas.

### 4.4 El sesgo de nivel absoluto persiste

Lo anterior acota la **estabilidad** del juez, no su **sesgo**. Un evaluador puede ser perfectamente consistente y perfectamente indulgente. La afirmación honesta es:

> El nivel absoluto de `consistency` **no es interpretable como exactitud** por estar contaminado de autoevaluación. Su **varianza entre corridas sí es interpretable** como medida de estabilidad del pipeline, porque el sesgo, sea cual sea su magnitud, es constante a través de las corridas y se cancela en la comparación.

### 4.5 Mitigación propuesta

1. **Juez independiente.** Reevaluar el corpus con un modelo de familia distinta (p. ej. Qwen-2.5-72B-Instruct o GPT-4o) y reportar el acuerdo inter-juez mediante kappa. La discrepancia entre jueces acota el sesgo de autoevaluación.
2. **Anclaje humano.** La submuestra de la §3.4 permite calibrar ambos jueces contra referencia clínica.
3. **Reporte dual, mientras tanto.** Presentar siempre `consistency` acompañada de la declaración de que es autoevaluación.

---

## 5. Respuestas a las dudas técnicas verificables

Estas son las respuestas a los puntos del documento *Parte 2*. Los scripts permiten verificar cada una.

### 5.1 Fase de asignación (`icd-experiment.py`)

| Duda | Respuesta |
|---|---|
| Qué recibe el modelo para producir `core_diagnosis` | Únicamente `diagnosis_en`. **No** recibe `clinical_summary` ni el texto del caso. Es una tarea de normalización terminológica sin contexto clínico. |
| Prompt exacto | `DISEASE_IDENTIFICATION_SYSTEM_PROMPT`: instrucción + 20 ejemplos *few-shot*. Reproducido parcialmente en §2.1. |
| Ejecución del modelo | MedGemma-27B-IT vía LM Studio, servidor local en `10.8.0.45:1234`, SDK `lmstudio-python`. |
| Temperatura, semilla, top_p, límite de tokens | **No se fijan.** Se usan los valores por defecto de LM Studio. Véase §7.1 — es un defecto. |
| Orden de la cascada | `exact_complete` → `exact_core` → `fuzzy_complete` → `fuzzy_core` → `who_search` + LLM → `nf`. Se detiene en el primer acierto. |
| Truncamiento | Corte en el primer punto: `5A11.2 → 5A11`, `BA00.0Z → BA00`. Los códigos `NF-` no se truncan. Los *clusters* de postcoordinación con `&` **sobreviven** al truncamiento (p. ej. `2F98&XA6KU8`). |
| ¿Se conserva el código completo antes de truncar? | **No.** Véase §7.2 — es un defecto. |
| Identificadores NF | Secuenciales globales `NF-0001`, `NF-0002`… Clave: diagnóstico normalizado. Persistidos en `nf_dictionary.json`, que **sobrevive entre corridas**. Véase §7.3. |

### 5.2 Fase de resolución y evaluación (`icd11-lookup.py`)

| Duda | Respuesta |
|---|---|
| Endpoint y versión | `https://id.who.int/icd/release/11/2024-01/mms` — release MMS 2024-01, fijada explícitamente, no `latest`. |
| Idioma del título | Español (`Accept-Language: es`). |
| Mecanismo de resolución | `/codeinfo/{code}` → `stemId` → petición a la URI de la entidad → título oficial. |
| Fallo de `/codeinfo` | *Fallback* automático al endpoint `/search`. Ambas vías cachean resultados en memoria. |
| Qué recibe el evaluador | Solo dos cadenas: `diagnosis_es` y `icd11_lookup`. **No** recibe el código, ni el `match_type`, ni el resumen clínico. El veredicto se emite sobre texto contra texto. |
| ¿Acepta categorías más generales? | Sí, **explícitamente declarado** en el prompt. Véase §3.1. |
| Manejo de salidas ambiguas | Normalización a minúsculas, eliminación de no-alfabéticos, aceptación solo si empieza por `yes` o `no`; cualquier otra salida → cadena vacía (fila no evaluada) y registro en consola. |
| ¿Ocurrió alguna vez? | **No.** En las 10 corridas, el número de filas sin evaluar coincide exactamente con el número de NF en todas ellas (27/27, 26/26, 28/28…). El parser no falló ni una vez en 1 741 invocaciones. |
| Por qué se excluyen los NF | Un código `NF-XXXX` no tiene título oficial que recuperar; no hay nada contra qué evaluar coherencia. Se excluyen del denominador de compatibilidad y se contabilizan aparte como fallo de cobertura. |
| ¿Misma configuración de modelo? | Sí, mismo modelo y mismo servidor. Con la misma ausencia de control de determinismo. |

### 5.3 Postcoordinación

El pipeline opera con **stem codes truncados a categoría**. No implementa postcoordinación deliberada. Los *clusters* con `&` que aparecen en la salida provienen de códigos que la WHO API devuelve ya postcoordinados y que atraviesan el truncamiento sin modificación —no de una construcción del sistema. Es una inconsistencia menor de tratamiento que conviene declarar.

---

## 6. Materiales entregados

Se entregan por separado: `icd-experiment.py`, `icd11-lookup.py`, `nf_dictionary.json`, `deep_analysis.py`, `deep_analysis_report.json`, `unstable_cases.csv`, `METODOLOGIA.md` y la lista de dependencias.

Los prompts no residen en fichero aparte: están embebidos como constantes en los scripts (`DISEASE_IDENTIFICATION_SYSTEM_PROMPT`, `SEMANTIC_MATCH_SYSTEM_PROMPT`, `SEMANTIC_MATCH_SYSTEM_PROMPT_EN`, `CONSISTENCY_SYSTEM_PROMPT`), lo que permite auditarlos directamente.

No se comparten credenciales de la WHO API ni ningún token. La configuración del modelo en LM Studio se documenta en §5.1.

---

## 7. Defectos adicionales detectados en la revisión interna

La revisión motivada por estas observaciones detectó tres problemas que los documentos no señalaban. Se declaran aquí porque afectan a la validez de las cifras publicadas.

### 7.1 Ausencia de control de determinismo (crítico)

Las llamadas al modelo se realizan **sin fijar temperatura ni semilla**. El pipeline es por tanto no determinista por construcción.

*Efecto medido:* 2 casos de 201 (1.0 %) reciben códigos distintos entre corridas.

| Caso | Códigos observados |
|---|---|
| `161-NEJMcps2108991.pdf` — Neumonía por *Legionella pneumophila* serogrupo 6 nosocomial | `1C19` (6/10) vs `CA40` (4/10) |
| `NEJMicm2412464.pdf` — Variz vaginal en el embarazo | `BD75` (9/10) vs `JA61` (1/10) |

El efecto es cuantitativamente pequeño, pero impide la reproducción exacta del experimento por terceros. **Corrección:** fijar `temperature=0` y semilla explícita, y reejecutar.

*Nota favorable:* las ramas `exact_complete` y `exact_core` resultaron **100 % deterministas** en las 10 corridas, como debe ser. La no-determinación se localiza íntegramente en las ramas que invocan al LLM (`who_search`, `fuzzy_core`), que es donde cabía esperarla.

### 7.2 El código pre-truncamiento no se conserva

`truncate_code()` descarta el sufijo sin guardarlo. La granularidad original se pierde de forma irreversible y no puede recuperarse sin reejecutar el pipeline completo. Ello impide, entre otras cosas, medir *a posteriori* cuánta especificidad se sacrificó al truncar. **Corrección:** añadir columna `icd11_code_full`.

### 7.3 Las corridas no son plenamente independientes (crítico para los intervalos de confianza)

`nf_dictionary.json` **persiste entre ejecuciones**. Un diagnóstico marcado NF queda registrado y su identificador se reutiliza en corridas posteriores.

Alcance medido:

- **25 de 201 casos (12.4 %) son NF en las 10 corridas** — quedan efectivamente fijados por el diccionario.
- **4 casos presentan *churn* real** (entran y salen de NF entre corridas), lo que confirma que el pipeline sí reintenta y que la variabilidad observada en la cobertura es genuina, no artefactual.
- El 86.6 % restante del dataset —todos los casos resueltos— **sí varía de forma independiente** entre corridas.

**Consecuencia:** los intervalos de confianza reportados asumen independencia entre corridas, supuesto que se viola parcialmente. El efecto es una **subestimación de la varianza real de la cobertura**: el 12.4 % del dataset está clavado y no contribuye a ella. Los intervalos de cobertura —86.6 % [86.4 %, 86.9 %]— deben leerse como *condicionados al estado del diccionario NF*, no como intervalos de un muestreo independiente.

La métrica de compatibilidad categorial está menos afectada, ya que se calcula sobre los casos resueltos, que sí varían libremente.

**Corrección:** reejecutar las 10 corridas partiendo de `nf_dictionary.json` vacío en cada una, o bien declarar explícitamente el condicionamiento al presentar los intervalos.

---

## 8. Síntesis

1. **La unidad de codificación es un código por caso, la condición focal.** La regla estaba implementada pero no documentada; se documenta.
2. **`core_diagnosis` no es recorte sino normalización**, y la frase completa resuelve el 59.7 % de los casos. La caracterización de "reducir antes de codificar" no describe el flujo real, aunque la pérdida de comorbilidad en la vía `core` es real y se acepta.
3. **`consistency` mide compatibilidad categorial, no exactitud.** Se renombra y se acompaña de la advertencia correspondiente. Sin gold standard humano no puede convertirse en exactitud.
4. **La autoevaluación es una limitación real y no subsanable con el diseño actual.** Se acota: el juez rechaza el 9.2 %, los errores son sistemáticos (16 casos reproducibles) y su inestabilidad es de 1 caso en 201. El nivel absoluto no es interpretable como exactitud; la varianza entre corridas sí lo es como estabilidad.
5. **Tres defectos adicionales se declaran voluntariamente:** falta de control de determinismo, pérdida del código completo y no-independencia entre corridas por el diccionario NF acumulativo.

Ninguna de las observaciones planteadas invalida los resultados. Todas ellas delimitan correctamente qué se ha medido: **estabilidad y compatibilidad categorial de un pipeline de normalización terminológica**, no exactitud de codificación clínica. Presentar las cifras bajo el segundo rótulo sería incorrecto; bajo el primero, son defendibles.
