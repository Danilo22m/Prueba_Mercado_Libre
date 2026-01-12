# Docker - Ejercicio 3: Sistema RAG + Agente Critico

Este documento explica como ejecutar el proyecto usando Docker.

## Requisitos Previos

- Docker instalado (version 20.10+)
- Docker Compose instalado (version 2.0+)
- Al menos 8GB de RAM disponible
- 10GB de espacio en disco
- API Key de Groq (https://console.groq.com/)

## Estructura del Proyecto

```
Ejercicio_3/
├── Dockerfile              # Definicion de la imagen Docker
├── docker-compose.yml      # Orquestacion de servicios
├── .dockerignore          # Archivos excluidos del build
├── Makefile               # Comandos simplificados
├── main.py                # Pipeline completo
├── .env                   # Variables de entorno (API keys)
├── .env.example           # Ejemplo de configuracion
├── config/
│   └── config.yaml        # Configuracion del proyecto
├── src/                   # Scripts de cada fase
│   ├── fase1_ingesta.py
│   ├── fase2_chunking.py
│   ├── fase3_retrieval.py
│   ├── fase4_generation.py
│   ├── fase5_agente_critico.py
│   └── fase6_evaluacion.py
├── data/
│   ├── raw/              # Dataset original (CSV)
│   └── processed/        # Datos procesados (JSON)
└── outputs/
    ├── results/          # Reportes y metricas
    ├── logs/             # Logs del pipeline
    └── models/           # Embeddings generados
```

## Inicio Rapido

### 1. Configurar API Key

Crea el archivo `.env` con tu API key de Groq:

```bash
cp .env.example .env
# Edita .env y agrega tu GROQ_API_KEY
```

El archivo `.env` debe verse asi:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 2. Verificar Dataset

Asegurate de tener el dataset en `data/raw/`:

```bash
ls -lh data/raw/Laptops_with_technical_specifications.csv
```

### 3. Construir la Imagen Docker

```bash
# Opcion 1: Usando Makefile (recomendado)
make build

# Opcion 2: Usando docker-compose directamente
docker-compose build
```

### 4. Ejecutar el Pipeline Completo

```bash
# Opcion 1: Usando Makefile (recomendado)
make run

# Opcion 2: Usando docker-compose
docker-compose run --rm rag-pipeline python main.py
```

Esto ejecutara todas las fases en secuencia:
1. FASE 1: Ingesta y Normalizacion de datos
2. FASE 2: Chunking e Indexacion con Embeddings
3. FASE 3: Retrieval (busqueda semantica)
4. FASE 4: Generation (respuestas con LLM)
5. FASE 5: Agente Critico (verificacion)
6. FASE 6: Evaluacion y reportes

## Comandos Disponibles (Makefile)

### Ejecucion

```bash
# Ver ayuda
make help

# Ejecutar pipeline completo
make run

# Ejecutar fases individuales
make fase1    # FASE 1: Ingesta y Normalizacion
make fase2    # FASE 2: Chunking e Indexacion
make fase3    # FASE 3: Retrieval (test)
make fase4    # FASE 4: Generation
make fase5    # FASE 5: Agente Critico (test simulado)
make fase6    # FASE 6: Evaluacion completa

# Probar conexion con Groq
make test-groq
```

### Utilidades

```bash
# Abrir shell interactivo en el contenedor
make shell

# Ver logs en tiempo real
make logs

# Detener contenedores
make stop

# Verificar requisitos
make verify
```

### Limpieza

```bash
# Limpiar outputs generados
make clean

# Limpiar todo (outputs + imagenes Docker)
make clean-all
```

## Comandos Docker Avanzados

### Sin Makefile

Si prefieres usar Docker directamente:

```bash
# Construir imagen
docker-compose build rag-pipeline

# Ejecutar pipeline completo
docker-compose run --rm rag-pipeline python main.py

# Ejecutar fase especifica
docker-compose run --rm rag-pipeline python src/fase1_ingesta.py
docker-compose run --rm rag-pipeline python src/fase2_chunking.py
docker-compose run --rm rag-pipeline python src/fase6_evaluacion.py

# Shell interactivo
docker-compose run --rm rag-pipeline /bin/bash

# Probar API key
docker-compose run --rm rag-pipeline python test_groq.py

# Detener todo
docker-compose down
```

## Volumenes y Persistencia

Los siguientes directorios estan montados como volumenes:

- `./data` → `/app/data` (Dataset de entrada y procesados)
- `./outputs` → `/app/outputs` (Resultados, logs, modelos)
- `./config` → `/app/config` (Configuracion)

Esto significa que:
- Los resultados se guardan en tu maquina local
- Puedes modificar la configuracion sin reconstruir la imagen
- El dataset no se duplica dentro del contenedor
- Los logs persisten entre ejecuciones

## Outputs Generados

Despues de ejecutar el pipeline completo, encontraras:

```
outputs/
├── logs/
│   ├── pipeline_completo_TIMESTAMP.log      # Log completo del flujo
│   └── logs_agente_critico_TIMESTAMP.json   # Logs del agente critico
├── results/
│   ├── ejemplos_completos_TIMESTAMP.json    # 10 ejemplos Query→Respuesta
│   ├── metricas_TIMESTAMP.json              # Metricas consolidadas
│   └── reporte_evaluacion_TIMESTAMP.txt     # Reporte legible
└── models/
    └── embeddings.pkl                       # Index de embeddings
```

## Troubleshooting

### Error: "GROQ_API_KEY no configurada"

**Solucion**: Crea el archivo `.env` con tu API key:

```bash
echo "GROQ_API_KEY=gsk_tu_api_key_aqui" > .env
```

### Error: "Invalid API Key"

**Solucion**: Verifica que tu API key sea valida:
- Debe empezar con `gsk_`
- Debe tener ~56 caracteres
- No debe tener espacios ni comillas
- Genera una nueva en https://console.groq.com/keys

### Error: "No such file or directory: data/raw/..."

**Solucion**: Asegurate de tener el dataset en el lugar correcto:

```bash
ls data/raw/Laptops_with_technical_specifications.csv
```

### Error: "Out of memory"

**Solucion**: Aumenta la memoria disponible para Docker en la configuracion de Docker Desktop, o ejecuta fases individuales:

```bash
make fase1
make fase2
# etc.
```

### Los embeddings tardan mucho

**Nota**: La primera ejecucion descarga el modelo `sentence-transformers/all-MiniLM-L6-v2` (~80MB). Las siguientes ejecuciones seran mas rapidas.

## Limpieza

```bash
# Limpiar solo outputs
make clean

# Limpiar todo (outputs + imagenes Docker)
make clean-all

# O manualmente:
docker-compose down --rmi all --volumes
rm -rf outputs/results/* outputs/logs/* outputs/models/*
rm -rf data/processed/*
```

## Verificacion de la Instalacion

Para verificar que Docker esta correctamente configurado:

```bash
# Verificar versiones
docker --version
docker-compose --version

# Verificar requisitos
make verify

# Construir imagen
make build

# Probar conexion con Groq
make test-groq

# Ejecutar solo FASE 1 (rapida, no requiere API)
make fase1

# Verificar outputs
ls -lh data/processed/
ls -lh outputs/
```

## Flujo del Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE RAG + AGENTE CRITICO                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ FASE 1: Ingesta y Normalizacion                                 │
│ - Cargar CSV (1000 laptops)                                     │
│ - Filtrar y normalizar (192 laptops)                            │
│ - Output: data/processed/laptops_normalized.json                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ FASE 2: Chunking e Indexacion                                   │
│ - Crear chunks field-based (1056 chunks)                        │
│ - Generar embeddings (384 dims)                                 │
│ - Output: data/processed/chunks.json, outputs/models/embeddings │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ FASE 3: Retrieval                                               │
│ - Busqueda semantica por similitud coseno                       │
│ - Retorna top-5 chunks relevantes                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ FASE 4: Generation                                              │
│ - LLM: Groq (Llama 3.3 70B)                                     │
│ - Genera respuesta con citas [laptop_id:campo]                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ FASE 5: Agente Critico (OBLIGATORIO)                            │
│ - Verifica cada afirmacion contra los chunks                    │
│ - Detecta alucinaciones y citas incorrectas                     │
│ - Decision: APROBAR / REHACER (max 2 reintentos)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ FASE 6: Evaluacion                                              │
│ - Ejecuta 10 queries de prueba                                  │
│ - Genera metricas y reportes                                    │
│ - Output: outputs/results/, outputs/logs/                       │
└─────────────────────────────────────────────────────────────────┘
```

## Soporte

Para mas informacion, consulta:
- Documentacion de Docker: https://docs.docker.com/
- Documentacion de Groq: https://console.groq.com/docs
- API de Sentence Transformers: https://www.sbert.net/
