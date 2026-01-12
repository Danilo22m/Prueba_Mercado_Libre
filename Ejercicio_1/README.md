# Ejercicio 1: Comparación LLM vs Modelo Clásico para Detección de Anomalías

Proyecto que compara el rendimiento de un modelo basado en LLM (Large Language Model) versus un modelo clásico de Machine Learning (Isolation Forest) para la detección de anomalías en series temporales de precios.

## Descripción del Proyecto

Este proyecto implementa un pipeline completo de detección de anomalías que:

1. **Preprocesa** datos históricos de precios
2. **Genera features** de ingeniería (lags, rolling stats, z-scores, etc.)
3. **Crea ground truth** usando método de Z-score
4. **Entrena y evalúa** dos modelos:
   - **Modelo A**: LLM usando Groq API (llama-3.3-70b-versatile)
   - **Modelo B**: Isolation Forest (scikit-learn)
5. **Compara** ambos modelos usando métricas estadísticas
6. **Realiza A/B testing** con bootstrap estadístico
7. **Genera visualizaciones** comparativas

## ⚠️ Pre-requisitos Importantes

### API Key de Groq (Obligatorio para Modelo A - LLM)

**IMPORTANTE**: Para ejecutar el Modelo A (LLM), necesitas una API key de Groq.

1. **Obtener API Key**:
   - Regístrate en: https://console.groq.com/
   - Genera una API key en la sección "API Keys"
   - Copia la key (empieza con `gsk_...`)

2. **Configurar API Key**:

   **Opción 1 - Variable de Entorno** (Recomendado):
   ```bash
   export GROQ_API_KEY="gsk_tu_api_key_aqui"
   ```

   **Opción 2 - Archivo .env**:
   ```bash
   # Crear archivo .env en la raíz del proyecto
   echo "GROQ_API_KEY=gsk_tu_api_key_aqui" > .env
   ```

3. **Verificar Configuración**:
   ```bash
   echo $GROQ_API_KEY  # Debe mostrar tu API key
   ```

> **Nota**: Sin la API key, solo podrás ejecutar el Modelo B (Isolation Forest). El Modelo A (LLM) fallará con error de autenticación.

## Estructura del Proyecto

```
Ejercicio_1/
├── config/
│   └── config.yaml              # Configuración del proyecto
├── data/
│   ├── raw/                     # Datos originales
│   └── processed/               # Datos procesados
├── outputs/
│   ├── models/                  # Modelos entrenados
│   ├── plots/                   # Gráficos generados
│   └── results/                 # Resultados y métricas
├── src/
│   ├── preprocessing.py         # Preprocesamiento de datos
│   ├── model_a_llm.py          # Modelo A (LLM)
│   ├── model_b_classic.py      # Modelo B (Isolation Forest)
│   ├── evaluation.py           # Evaluación comparativa
│   ├── ab_test.py              # A/B Testing estadístico
│   └── visualizaciones_series.py  # Visualizaciones
├── main.py                      # Orquestador principal
├── requirements.txt             # Dependencias Python
├── Dockerfile                   # Imagen Docker
├── docker-compose.yml          # Docker Compose
├── run_docker.sh               # Script de ayuda Docker
├── DOCKER.md                   # Documentación Docker
└── README.md                   # Este archivo
```

## Instalación

### Opción 1: Instalación Local

#### Pre-requisitos

- Python 3.11+
- pip

#### Pasos

1. Clonar el repositorio:
```bash
cd Ejercicio_1
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

4. **⚠️ Configurar API key de Groq** (Ver sección "Pre-requisitos Importantes" arriba):
```bash
export GROQ_API_KEY="gsk_tu_api_key_aqui"
# O crear archivo .env con: GROQ_API_KEY=gsk_tu_api_key_aqui
```

### Opción 2: Docker (Recomendado)

Ver [DOCKER.md](DOCKER.md) para instrucciones detalladas.

#### Inicio Rápido con Docker

1. Crear archivo `.env`:
```bash
cp .env.example .env
# Editar .env y agregar GROQ_API_KEY
```

2. Construir imagen:
```bash
./run_docker.sh build
```

3. Ejecutar pipeline:
```bash
./run_docker.sh run-eval
```

## Uso

> **⚠️ Recordatorio**: Asegúrate de tener configurada la variable `GROQ_API_KEY` antes de ejecutar el Modelo A (LLM). Ver sección "Pre-requisitos Importantes".

### Pipeline Completo

```bash
# 1. Verificar API key (opcional)
echo $GROQ_API_KEY

# 2. Ejecutar pipeline completo
python main.py
```

### Solo Evaluación (sin entrenar modelos)

```bash
python main.py --skip-preprocessing --skip-models
```

### Argumentos Disponibles

```bash
python main.py --help
```

Opciones:
- `--config PATH`: Ruta a archivo de configuración personalizado
- `--skip-preprocessing`: Omitir preprocesamiento (usar datos procesados existentes)
- `--skip-models`: Omitir entrenamiento (usar predicciones existentes)
- `--n-productos N`: Número de productos para visualizar (default: 3)

### Ejemplos

```bash
# Pipeline completo
python main.py

# Solo análisis con predicciones existentes
python main.py --skip-preprocessing --skip-models

# Generar más visualizaciones
python main.py --skip-preprocessing --skip-models --n-productos 5

# Usar configuración personalizada
python main.py --config config/config_custom.yaml
```

## Configuración

El archivo `config/config.yaml` contiene todos los parámetros configurables:

### Datos
- `test_size`: Proporción del conjunto de prueba (default: 0.2)
- `random_seed`: Semilla aleatoria para reproducibilidad (default: 42)

### Preprocesamiento
- `min_registros_producto`: Mínimo de registros por producto (default: 10)
- `umbral_zscore_ground_truth`: Umbral de Z-score para anomalías (default: 3.0)

### Modelo A (LLM)
- `model_name`: Modelo de Groq a usar (default: llama-3.3-70b-versatile)
- `sample_pct`: Porcentaje de muestra (default: 0.05 = 5%)
- `temperature`: Temperatura del modelo (default: 0.0)
- `max_requests_per_minute`: Rate limit (default: 30)

### Modelo B (Isolation Forest)
- `contamination`: Proporción esperada de anomalías (default: 0.02 = 2%)
- `n_estimators`: Número de árboles (default: 100)
- `max_samples`: Muestras por árbol (default: 256)

## Modelos

### Modelo A: LLM (Groq API)

- **Proveedor**: Groq
- **Modelo**: llama-3.3-70b-versatile
- **Enfoque**: Zero-shot classification usando prompts estructurados
- **Output**: JSON con label, confidence, y reason
- **Ventajas**: Capacidad de razonamiento, explicaciones humanas
- **Limitaciones**: Rate limits, costo por token, latencia

### Modelo B: Isolation Forest

- **Biblioteca**: scikit-learn
- **Enfoque**: Detección de anomalías no supervisada
- **Features**: 10 features engineered (precio, lags, rolling stats, z-score)
- **Ventajas**: Rápido, sin costo, escalable
- **Limitaciones**: Sin explicaciones, requiere tuning de contamination

## Métricas de Evaluación

El proyecto calcula las siguientes métricas para ambos modelos:

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Media armónica de Precision y Recall
- **PR-AUC**: Área bajo la curva Precision-Recall
- **Confusion Matrix**: TP, TN, FP, FN

### Evaluación Independiente

**IMPORTANTE**: Los modelos se evalúan de forma independiente:

- **Modelo A (LLM)**: Evaluado sobre muestra de 5% del test set (~100 registros)
- **Modelo B (IF)**: Evaluado sobre 100% del test set (~35,000 registros)

Esto permite comparar el rendimiento de cada modelo en su contexto de uso.

## A/B Testing

El proyecto implementa A/B testing estadístico usando:

- **Método**: Bootstrap estratificado
- **Iteraciones**: 1,000 muestras bootstrap
- **Métricas comparadas**: Precision, Recall, F1-Score
- **Significancia**: Alpha = 0.05
- **Output**: p-values, intervalos de confianza (95%)

## Resultados

Los resultados se generan en:

### Archivos de Métricas
- `outputs/results/comparacion_modelos.csv`: Tabla comparativa
- `outputs/results/evaluacion_completa.json`: Métricas completas
- `outputs/results/ab_test_results.json`: Resultados de A/B test

### Gráficos
- `outputs/plots/confusion_matrices.png`: Matrices de confusión
- `outputs/plots/precision_recall_curves.png`: Curvas PR
- `outputs/plots/series_temporales_comparacion_modelos.png`: Series temporales

### Predicciones
- `outputs/results/predicciones_llm.csv`: Predicciones del LLM
- `outputs/results/predicciones_isolation_forest.csv`: Predicciones de IF

## Features Engineered

El preprocesamiento genera 10 features:

1. `PRICE`: Precio original
2. `PRICE_LAG_1`: Precio anterior (lag 1)
3. `PRICE_MEAN_GLOBAL`: Media global del producto
4. `PRICE_STD_GLOBAL`: Desviación estándar global
5. `PRICE_MEAN_ROLLING`: Media móvil (ventana 7)
6. `PRICE_STD_ROLLING`: Desviación estándar móvil
7. `PRICE_DIFF_VS_MEAN`: Diferencia vs media global
8. `PRICE_DIFF_PCT`: Diferencia porcentual vs precio anterior
9. `PRICE_DIFF`: Diferencia absoluta vs precio anterior
10. `ZSCORE`: Z-score del precio

## Troubleshooting

### Error: API key no configurada

**Síntoma**: `ValueError: API key no encontrada` o `Error code: 401 - Unauthorized`

**Solución**:
1. Verificar que la API key esté configurada:
   ```bash
   echo $GROQ_API_KEY
   ```
2. Si está vacía, configurarla:
   ```bash
   export GROQ_API_KEY="gsk_tu_api_key_aqui"
   ```
3. O crear archivo `.env` con:
   ```
   GROQ_API_KEY=gsk_tu_api_key_aqui
   ```
4. Verificar que la key sea válida en: https://console.groq.com/

> **Nota**: La API key debe empezar con `gsk_` y tener al menos 50 caracteres.

### Error: Rate limit de Groq

**Síntoma**: `Error code: 429 - Rate limit exceeded`

**Solución**:
- Esperar hasta que se resetee el límite diario (24 horas)
- Reducir `sample_pct` en config.yaml (ej: 0.01 = 1%)
- Ejecutar solo evaluación: `python main.py --skip-models`

### Error: No module named 'pandas'

**Síntoma**: `ModuleNotFoundError: No module named 'pandas'`

**Solución**:
- Activar entorno virtual: `source venv/bin/activate`
- Reinstalar dependencias: `pip install -r requirements.txt`

### Gráficos en blanco

**Síntoma**: PR curves o visualizaciones aparecen vacías

**Solución**:
- Verificar que existan archivos de predicciones en `outputs/results/`
- Reejecutar modelos si los archivos están corruptos
- Verificar logs para errores en el merge de datos

## Autor

Danilo Melo

## Fecha

2026-01-11

## Licencia

Este proyecto es parte de un ejercicio técnico.
