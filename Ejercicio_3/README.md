# Ejercicio 3: Sistema RAG + Agente Critico para Q&A de Laptops

Sistema de Retrieval-Augmented Generation (RAG) con Agente Critico para responder preguntas sobre especificaciones tecnicas de laptops, verificando la fidelidad de las respuestas generadas.

## Descripcion del Proyecto

Este proyecto implementa un pipeline RAG completo que:

1. **FASE 1: Ingesta y Normalizacion** - Procesa dataset de laptops, filtra y normaliza datos
2. **FASE 2: Chunking e Indexacion** - Crea chunks por campo y genera embeddings
3. **FASE 3: Retrieval** - Busqueda semantica usando similitud coseno
4. **FASE 4: Generation** - Genera respuestas con LLM (Groq) incluyendo citas
5. **FASE 5: Agente Critico** - Verifica respuestas y detecta alucinaciones (OBLIGATORIO)
6. **FASE 6: Evaluacion** - Ejecuta queries de prueba y genera metricas

## Pre-requisitos Importantes

### API Key de Groq (Obligatorio)

**IMPORTANTE**: Para ejecutar las fases 4, 5 y 6, necesitas una API key de Groq.

1. **Obtener API Key**:
   - Registrate en: https://console.groq.com/
   - Genera una API key en la seccion "API Keys"
   - Copia la key (empieza con `gsk_...`)

2. **Configurar API Key**:

   **Opcion Recomendada - Archivo .env**:
   ```bash
   # Copiar ejemplo y editar
   cp .env.example .env
   # Editar .env con tu API key
   ```

   El archivo `.env` debe contener SOLO esta linea (sin espacios ni comentarios):
   ```
   GROQ_API_KEY=gsk_tu_api_key_aqui
   ```

3. **Verificar Configuracion**:
   ```bash
   python test_groq.py
   ```

> **Nota**: Sin la API key, solo podras ejecutar FASE 1 y FASE 2. Las fases 3-6 requieren la API de Groq.

## Estructura del Proyecto

```
Ejercicio_3/
├── config/
│   └── config.yaml              # Configuracion del proyecto
├── data/
│   ├── raw/                     # Dataset original (CSV)
│   │   └── Laptops_with_technical_specifications.csv
│   └── processed/               # Datos procesados (JSON)
│       ├── laptops_normalized.json
│       ├── chunks.json
│       └── normalization_report.json
├── outputs/
│   ├── models/                  # Embeddings generados
│   │   └── embeddings.pkl
│   ├── logs/                    # Logs de ejecucion
│   │   ├── pipeline_completo.log
│   │   └── logs_agente_critico.json
│   └── results/                 # Resultados y metricas
│       ├── ejemplos_completos.json
│       ├── metricas.json
│       └── reporte_evaluacion.txt
├── src/
│   ├── fase1_ingesta.py         # FASE 1: Ingesta y normalizacion
│   ├── fase2_chunking.py        # FASE 2: Chunking e indexacion
│   ├── fase3_retrieval.py       # FASE 3: Busqueda semantica
│   ├── fase4_generation.py      # FASE 4: Generacion con LLM
│   ├── fase5_agente_critico.py  # FASE 5: Verificacion de respuestas
│   └── fase6_evaluacion.py      # FASE 6: Evaluacion completa
├── main.py                      # Orquestador principal (6 fases)
├── test_groq.py                 # Test de conexion con Groq
├── requirements.txt             # Dependencias Python
├── Dockerfile                   # Imagen Docker
├── docker-compose.yml           # Docker Compose
├── Makefile                     # Comandos simplificados
├── README_DOCKER.md             # Documentacion Docker
├── .env.example                 # Ejemplo de configuracion
└── README.md                    # Este archivo
```

## Instalacion

### Opcion 1: Instalacion Local

#### Pre-requisitos

- Python 3.11+
- pip

#### Pasos

1. Ir al directorio del ejercicio:
```bash
cd Ejercicio_3
```

2. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Configurar API key de Groq:
```bash
cp .env.example .env
# Editar .env con tu API key
```

5. Verificar conexion con Groq:
```bash
python test_groq.py
```

### Opcion 2: Docker (Recomendado)

Ver [README_DOCKER.md](README_DOCKER.md) para instrucciones detalladas.

#### Inicio Rapido con Docker

1. Crear archivo `.env`:
```bash
cp .env.example .env
# Editar .env con tu GROQ_API_KEY
```

2. Construir imagen:
```bash
make build
```

3. Ejecutar pipeline completo:
```bash
make run
```

## Uso

### Pipeline Completo (6 Fases)

```bash
python main.py
```

Esto ejecutara secuencialmente:
- FASE 1: Ingesta y Normalizacion
- FASE 2: Chunking e Indexacion
- FASE 3: Retrieval (test interno)
- FASE 4: Generation (test interno)
- FASE 5: Agente Critico (test interno)
- FASE 6: Evaluacion completa (10 queries)

### Ejecutar Fases Individuales

```bash
# FASE 1: Ingesta y Normalizacion (no requiere API)
python src/fase1_ingesta.py

# FASE 2: Chunking e Indexacion (no requiere API)
python src/fase2_chunking.py

# FASE 3: Retrieval - test de busqueda
python src/fase3_retrieval.py

# FASE 4: Generation - test de generacion
python src/fase4_generation.py

# FASE 5: Agente Critico - test de verificacion
python src/fase5_agente_critico.py

# FASE 6: Evaluacion completa
python src/fase6_evaluacion.py
```

### Comandos Docker (Makefile)

```bash
make help       # Ver todos los comandos disponibles
make build      # Construir imagen Docker
make run        # Ejecutar pipeline completo
make fase1      # Ejecutar FASE 1
make fase2      # Ejecutar FASE 2
make fase3      # Ejecutar FASE 3
make fase4      # Ejecutar FASE 4
make fase5      # Ejecutar FASE 5
make fase6      # Ejecutar FASE 6
make test-groq  # Probar conexion con Groq
make shell      # Abrir shell en contenedor
make clean      # Limpiar outputs
```

## Arquitectura del Sistema

### Flujo del Pipeline RAG

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE RAG + AGENTE CRITICO                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ FASE 1: Ingesta y Normalizacion                                 │
│ - Cargar CSV (1000 laptops)                                     │
│ - Filtrar y normalizar (192 laptops con datos completos)        │
│ - Output: data/processed/laptops_normalized.json                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ FASE 2: Chunking e Indexacion                                   │
│ - Crear chunks field-based (1 chunk por campo de laptop)        │
│ - Generar embeddings (384 dimensiones)                          │
│ - Modelo: sentence-transformers/all-MiniLM-L6-v2                │
│ - Output: data/processed/chunks.json, outputs/models/embeddings │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ FASE 3: Retrieval                                               │
│ - Busqueda semantica por similitud coseno                       │
│ - Retorna top-5 chunks mas relevantes                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ FASE 4: Generation                                              │
│ - LLM: Groq (Llama 3.3 70B Versatile)                          │
│ - Genera respuesta con citas [laptop_id:campo]                  │
│ - Maximo 120 palabras por respuesta                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ FASE 5: Agente Critico (OBLIGATORIO)                            │
│ - Verifica cada afirmacion contra los chunks originales         │
│ - Detecta alucinaciones y citas incorrectas                     │
│ - Decision: APROBAR / REHACER (max 2 reintentos)                │
│ - Output: logs_agente_critico.json                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ FASE 6: Evaluacion                                              │
│ - Ejecuta 10 queries de prueba                                  │
│ - Calcula metricas de fidelidad                                 │
│ - Genera ejemplos completos y reportes                          │
│ - Output: outputs/results/, outputs/logs/                       │
└─────────────────────────────────────────────────────────────────┘
```

## Configuracion

El archivo `config/config.yaml` contiene todos los parametros configurables:

### Datos
- `min_laptops`: Minimo de laptops a procesar (default: 200)
- `max_laptops`: Maximo de laptops a procesar (default: 400)
- `min_non_null_fields`: Minimo de campos no nulos requeridos (default: 15)

### Chunking (FASE 2)
- `chunking_strategy`: Estrategia de chunking (default: field_based)
- `embedding_model`: Modelo de embeddings (default: all-MiniLM-L6-v2)

### Retrieval (FASE 3)
- `top_k`: Numero de chunks a recuperar (default: 5)
- `min_score`: Score minimo de similitud (default: 0.1)

### Generation (FASE 4)
- `model_name`: Modelo LLM (default: llama-3.3-70b-versatile)
- `temperature`: Temperatura del modelo (default: 0.0)
- `max_words_response`: Maximo de palabras por respuesta (default: 120)

### Agente Critico (FASE 5)
- `max_retries`: Numero maximo de reintentos si se detectan problemas (default: 2)

### Evaluacion (FASE 6)
- `n_test_queries`: Numero de queries de prueba (default: 10)

## Componentes Principales

### Modelo de Embeddings

- **Modelo**: sentence-transformers/all-MiniLM-L6-v2
- **Dimensiones**: 384
- **Normalizacion**: Embeddings normalizados para similitud coseno
- **Uso**: Indexacion de chunks y busqueda semantica

### LLM (Large Language Model)

- **Proveedor**: Groq
- **Modelo**: llama-3.3-70b-versatile
- **Temperatura**: 0.0 (respuestas deterministas)
- **Formato de citas**: [laptop_id:campo]

### Agente Critico

El Agente Critico es un componente **OBLIGATORIO** que:

1. **Extrae afirmaciones** de la respuesta generada
2. **Verifica cada afirmacion** contra los chunks originales
3. **Detecta problemas**:
   - Alucinaciones (informacion no presente en chunks)
   - Citas incorrectas (referencias a datos erroneos)
   - Contradicciones con los datos
4. **Emite veredicto**: APROBAR o REHACER
5. **Proporciona feedback** para mejorar la respuesta si es necesario

## Resultados Generados

Despues de ejecutar el pipeline completo, encontraras:

### Archivos de Resultados
- `outputs/results/ejemplos_completos.json`: 10 ejemplos Query -> Chunks -> Respuesta
- `outputs/results/metricas.json`: Metricas consolidadas (incluye metricas avanzadas)
- `outputs/results/reporte_evaluacion.txt`: Reporte legible

### Logs
- `outputs/logs/pipeline_completo.log`: Log completo de ejecucion
- `outputs/logs/logs_agente_critico.json`: Decisiones del agente critico

### Modelos
- `outputs/models/embeddings.pkl`: Index de embeddings generados

> **Nota**: Los archivos se sobrescriben en cada ejecucion (sin timestamps).

## Metricas de Evaluacion

El sistema calcula las siguientes metricas avanzadas de RAG:

### Metricas de Retrieval
- **Precision**: Proporcion de chunks recuperados que son relevantes para la query
- **Recall**: Proporcion de informacion relevante que fue recuperada
- **F1-Score**: Media armonica de Precision y Recall

### Metricas de Generacion
- **Faithfulness**: Fidelidad de la respuesta al contexto (evaluada por LLM)
- **Answer Coverage**: Proporcion de informacion esperada presente en la respuesta

### Metricas del Agente Critico
- **Tasa de Aprobacion**: Porcentaje de respuestas aprobadas en primer intento
- **Promedio de Intentos**: Numero promedio de intentos para aprobar una respuesta

### Metricas de Tiempo
- **Tiempo Total**: Tiempo promedio end-to-end por query
- **Tiempo de Retrieval**: Tiempo promedio de busqueda semantica
- **Tiempo de Generation**: Tiempo promedio de generacion LLM
- **Tiempo de Verificacion**: Tiempo promedio del agente critico

## Queries de Prueba

El sistema incluye 10 queries de prueba con ground truth para evaluacion:

1. Que procesador tiene el HP 15?
2. Que laptops tienen tarjeta grafica NVIDIA?
3. Cual es el tamano de pantalla del Lenovo ThinkPad?
4. Que laptop tiene mejor rendimiento de CPU?
5. Cuales laptops tienen WiFi 6?
6. Que GPU tiene el ASUS ROG?
7. Laptops con pantalla de 15.6 pulgadas
8. Que laptops tienen puerto Ethernet?
9. Cual es el peso del Dell XPS?
10. Que resolucion de pantalla tiene el Lenovo IdeaPad?

## Troubleshooting

### Error: "GROQ_API_KEY no configurada"

**Solucion**:
1. Verificar que existe el archivo `.env`:
   ```bash
   cat .env
   ```
2. Verificar formato (sin espacios ni comentarios):
   ```
   GROQ_API_KEY=gsk_tu_api_key_aqui
   ```
3. Ejecutar test:
   ```bash
   python test_groq.py
   ```

### Error: "Invalid API Key" (401)

**Solucion**:
1. Verificar que la API key sea valida
2. Debe empezar con `gsk_`
3. Debe tener ~56 caracteres
4. Sin espacios ni saltos de linea
5. Generar nueva key en: https://console.groq.com/keys

### Error: "Rate limit exceeded" (429)

**Solucion**:
- Esperar hasta que se resetee el limite
- Reducir `n_test_queries` en config.yaml
- Ejecutar fases individuales en lugar del pipeline completo

### Error: "No such file or directory: data/raw/..."

**Solucion**:
```bash
ls data/raw/Laptops_with_technical_specifications.csv
```
Si no existe, asegurate de tener el dataset en la ubicacion correcta.

### Los embeddings tardan mucho

**Nota**: La primera ejecucion descarga el modelo (~80MB). Las siguientes seran mas rapidas.

### Error de memoria

**Solucion**:
- Ejecutar fases individuales en lugar del pipeline completo
- Si usas Docker, aumentar memoria en docker-compose.yml

## Tecnologias Utilizadas

- **Python 3.11+**
- **sentence-transformers**: Embeddings semanticos
- **numpy**: Operaciones vectoriales
- **scikit-learn**: Similitud coseno
- **groq**: API del LLM
- **python-dotenv**: Manejo de variables de entorno
- **PyYAML**: Configuracion
- **Docker**: Containerizacion

## Autor

Danilo Melo

## Fecha

2026-01-12

## Licencia

Este proyecto es parte de un ejercicio tecnico para Mercado Libre.
