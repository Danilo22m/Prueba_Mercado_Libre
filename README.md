# Prueba Tecnica - Mercado Libre

**Autor:** Jose Danilo Melo Fuquene
**Fecha:** Enero 2026

## Descripcion General

Este repositorio contiene la solucion a la prueba tecnica de Mercado Libre, compuesta por 3 ejercicios independientes que abordan diferentes desafios de ciencia de datos e inteligencia artificial.

## Estructura del Repositorio

```
Prueba_Mercado_Libre/
├── Ejercicio_1/          # Deteccion de Anomalias: LLM vs Modelo Clasico
├── Ejercicio_2/          # Analisis de Grafos de URLs
├── Ejercicio_3/          # Sistema RAG + Agente Critico
└── README.md             # Este archivo
```

---

## Ejercicio 1: Comparacion LLM vs Modelo Clasico para Deteccion de Anomalias

### Objetivo
Comparar el rendimiento de un modelo basado en LLM (Large Language Model) versus un modelo clasico de Machine Learning (Isolation Forest) para la deteccion de anomalias en series temporales de precios.

### Componentes Principales
- **Modelo A (LLM):** Groq API con Llama 3.3 70B - Zero-shot classification
- **Modelo B (Clasico):** Isolation Forest (scikit-learn)
- **Evaluacion:** Metricas de precision, recall, F1-score, PR-AUC
- **A/B Testing:** Bootstrap estratificado con 1,000 iteraciones

### Pipeline
1. Preprocesamiento de datos historicos de precios
2. Generacion de features (lags, rolling stats, z-scores)
3. Entrenamiento y prediccion con ambos modelos
4. Evaluacion comparativa
5. A/B Testing estadistico
6. Visualizaciones de series temporales

### Ejecucion
```bash
cd Ejercicio_1
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Configurar GROQ_API_KEY en .env
python main.py
```

### Documentacion
- [README Ejercicio 1](Ejercicio_1/README.md)
- [Docker](Ejercicio_1/DOCKER.md)

---

## Ejercicio 2: Analisis de Grafos de URLs

### Objetivo
Analizar la estructura de un grafo de URLs web para identificar nodos influyentes, simular propagacion de informacion y generar recomendaciones estrategicas para e-commerce.

### Componentes Principales
- **Dataset:** web-Stanford (Stanford Large Network Dataset Collection)
- **Metricas:** PageRank, Betweenness, Closeness, HITS (Hubs/Authorities)
- **Simulacion:** Modelo de propagacion epidemiologica (SI/SIR)
- **Visualizacion:** Grafos interactivos con PyVis

### Pipeline (7 Fases)
1. Seleccion de subgrafo representativo
2. Analisis estadistico del grafo
3. Calculo de metricas de centralidad
4. Ranking Top-20 de nodos influyentes
5. Visualizacion interactiva
6. Simulacion de propagacion de informacion
7. Generacion de recomendaciones para e-commerce

### Ejecucion
```bash
cd Ejercicio_2
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Documentacion
- [README Ejercicio 2](Ejercicio_2/README.md)
- [Docker](Ejercicio_2/README_DOCKER.md)

---

## Ejercicio 3: Sistema RAG + Agente Critico para Q&A de Laptops

### Objetivo
Implementar un sistema de Retrieval-Augmented Generation (RAG) con un Agente Critico que verifica la fidelidad de las respuestas generadas, detectando alucinaciones y citas incorrectas.

### Componentes Principales
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2, 384 dims)
- **LLM:** Groq API con Llama 3.3 70B
- **Agente Critico:** Verificacion de afirmaciones contra chunks originales
- **Dataset:** Especificaciones tecnicas de laptops

### Pipeline (6 Fases)
1. **Ingesta y Normalizacion:** Procesar CSV, filtrar datos de calidad
2. **Chunking e Indexacion:** Crear chunks por campo, generar embeddings
3. **Retrieval:** Busqueda semantica con similitud coseno (top-5)
4. **Generation:** Generar respuestas con citas [laptop_id:campo]
5. **Agente Critico (OBLIGATORIO):** Verificar respuestas, detectar alucinaciones
6. **Evaluacion:** Ejecutar queries de prueba, generar metricas

### Ejecucion
```bash
cd Ejercicio_3
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Configurar GROQ_API_KEY en .env
python main.py
```

### Documentacion
- [README Ejercicio 3](Ejercicio_3/README.md)
- [Docker](Ejercicio_3/README_DOCKER.md)

---

## Requisitos Generales

### Software
- Python 3.11+
- Docker 20.10+ (opcional)
- Docker Compose 2.0+ (opcional)

### API Keys
Los ejercicios 1 y 3 requieren una API key de Groq:
1. Registrarse en: https://console.groq.com/
2. Generar API key (formato: `gsk_...`)
3. Configurar en archivo `.env`:
   ```
   GROQ_API_KEY=gsk_tu_api_key_aqui
   ```

### Instalacion con Docker

Cada ejercicio incluye soporte completo para Docker:

```bash
# Ejercicio 1
cd Ejercicio_1 && make build && make run

# Ejercicio 2
cd Ejercicio_2 && make build && make run

# Ejercicio 3
cd Ejercicio_3 && make build && make run
```

---

## Resumen de Tecnologias

| Tecnologia | Ejercicio 1 | Ejercicio 2 | Ejercicio 3 |
|------------|:-----------:|:-----------:|:-----------:|
| Python 3.11+ | X | X | X |
| Pandas | X | X | X |
| NumPy | X | X | X |
| Scikit-learn | X | X | X |
| Groq API (LLM) | X | | X |
| NetworkX | | X | |
| PyVis | | X | |
| Sentence-Transformers | | | X |
| Matplotlib | X | X | |
| Docker | X | X | X |

---

## Outputs Generados

Cada ejercicio genera sus resultados en la carpeta `outputs/`:

### Ejercicio 1
- `outputs/results/` - Metricas y predicciones
- `outputs/plots/` - Graficos comparativos
- `outputs/logs/` - Logs de ejecucion

### Ejercicio 2
- `outputs/results/` - Analisis y rankings
- `outputs/visualizaciones/` - Grafos interactivos
- `outputs/propagacion/` - Simulaciones
- `outputs/recomendaciones/` - Estrategias e-commerce
- `outputs/logs/` - Logs de ejecucion

### Ejercicio 3
- `outputs/results/` - Ejemplos y metricas
- `outputs/models/` - Embeddings
- `outputs/logs/` - Logs de ejecucion y agente critico

---

## Licencia

Este proyecto es parte de una prueba tecnica para Mercado Libre.
