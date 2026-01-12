# ✅ Docker Setup Completado

## Archivos Creados

### 🐳 Docker Core
- ✅ **Dockerfile** - Imagen Docker con Python 3.11 + dependencias
- ✅ **docker-compose.yml** - Orquestación de servicios (análisis + web viewer)
- ✅ **.dockerignore** - Excluye archivos innecesarios del build
- ✅ **.env.example** - Template de variables de entorno

### 🛠️ Utilidades
- ✅ **Makefile** - Comandos simplificados (make build, make run, etc.)
- ✅ **docker-validate.sh** - Script de validación pre-Docker
- ✅ **quick-start.sh** - Script de inicio rápido interactivo

### 📚 Documentación
- ✅ **README.md** - Documentación principal del proyecto
- ✅ **README_DOCKER.md** - Guía completa de Docker
- ✅ **DOCKER_SETUP_COMPLETE.md** - Este archivo

### 🐍 Python Core (ya existentes)
- ✅ **main.py** - Pipeline completo de 7 fases
- ✅ **requirements.txt** - Dependencias Python
- ✅ **config/config.yaml** - Configuración centralizada
- ✅ **src/fase*.py** - Scripts de cada fase (1-7)

---

## 🚀 Inicio Rápido

### Opción 1: Script Interactivo (Más Fácil)

```bash
./quick-start.sh
```

Selecciona opción **1** para Docker o **2** para Python local.

### Opción 2: Docker Directo

```bash
# 1. Validar entorno (opcional)
./docker-validate.sh

# 2. Construir imagen
make build

# 3. Ejecutar pipeline completo
make run

# 4. Ver visualizaciones
make web
# Abre http://localhost:8000
```

### Opción 3: Python Local

```bash
# 1. Activar entorno virtual
source venv/bin/activate

# 2. Ejecutar pipeline
python main.py
```

---

## 📋 Comandos Make Disponibles

```bash
make help           # Ver todos los comandos
make build          # Construir imagen Docker
make run            # Ejecutar pipeline completo (7 fases)
make run-fase N=X   # Ejecutar solo fase X (1-7)
make shell          # Shell interactivo en contenedor
make logs           # Ver logs en tiempo real
make web            # Servidor web para visualizaciones
make stop           # Detener contenedores
make clean          # Limpiar outputs generados
make clean-all      # Limpiar todo (outputs + imágenes)
```

### Shortcuts para Fases Individuales

```bash
make fase1          # Selección de subgrafo
make fase2          # Análisis estadístico
make fase3          # Métricas de centralidad
make fase4          # Ranking Top-20
make fase5          # Visualización
make fase6          # Simulación de propagación
make fase7          # Recomendaciones
```

---

## 🏗️ Arquitectura Docker

### Imagen Base
- **Python**: 3.11-slim
- **Dependencias**: NetworkX, Pandas, NumPy, Matplotlib, Pyvis
- **Tamaño**: ~500MB (comprimido)

### Volúmenes Montados
```
./data     → /app/data          (Dataset de entrada)
./outputs  → /app/outputs       (Resultados generados)
./config   → /app/config        (Configuración)
```

### Servicios

#### 1. grafos-analisis (Principal)
- Ejecuta el pipeline de análisis
- Límites: 4 CPUs, 8GB RAM (ajustables)
- Comando: `python main.py`

#### 2. web-viewer (Opcional)
- Servidor HTTP simple para visualizaciones
- Puerto: 8000
- Activar con: `make web` o `docker-compose --profile web up`

---

## 📊 Pipeline de 7 Fases

### FASE 1: Selección de Subgrafo (2-5 min)
- Reduce dataset de 281k nodos → 2k nodos
- Genera: `data/processed/subgrafo.gpickle`

### FASE 2: Análisis Estadístico (10-30 seg)
- Calcula métricas básicas del grafo
- Genera: `outputs/results/fase2_estadisticas.json`

### FASE 3: Métricas de Centralidad (5-15 min)
- PageRank, HITS, Betweenness, Closeness
- Genera: `outputs/results/fase3_metricas_centralidad.csv`

### FASE 4: Ranking Top-20 (10-30 seg)
- Identifica nodos más importantes y roles
- Genera: `outputs/results/fase4_analisis_top20.json`

### FASE 5: Visualización (1-2 min)
- Grafos estáticos (PNG) e interactivos (HTML)
- Genera: `outputs/visualizaciones/fase5_*.{png,html}`

### FASE 6: Simulación de Propagación (10-30 min)
- Modelo Independent Cascade, 5 estrategias
- Genera: `outputs/propagacion/fase6_*.json`

### FASE 7: Recomendaciones (10-30 seg)
- 3 accionables para e-commerce
- Genera: `outputs/recomendaciones/fase7_*.*` (JSON, MD, CSV, TXT)

**Tiempo Total**: 20-60 minutos (según CPU/RAM)

---

## 🎯 Resultados Clave

### Accionables Generados (FASE 7)

#### ACT-001: Link Building Interno
- **Problema**: 14% de páginas huérfanas (sin enlaces)
- **ROI**: Alto
- **KPI**: Reducir huérfanos 50, PageRank +20%

#### ACT-002: Campañas Virales Optimizadas
- **Estrategia**: Top-5 por Betweenness
- **ROI**: Muy Alto
- **KPI**: Cobertura +2.7%, Viralización +30%

#### ACT-003: Potenciar Páginas Autoridad
- **Oportunidad**: 3 páginas autoridad identificadas
- **ROI**: Alto
- **KPI**: Conversión +15%, Engagement +25%

### Visualizaciones

#### Estática (PNG)
- Top-10 nodos + vecinos
- Tamaño proporcional a PageRank
- Colores por rol

#### Interactiva (HTML)
- Zoom, pan, selección
- Tooltips con métricas
- Filtros por rol
- **URL**: http://localhost:8000/fase5_grafo_interactivo.html

---

## 🔧 Configuración

### Variables de Entorno (.env)

```bash
# Copiar template
cp .env.example .env

# Editar variables
vim .env
```

Variables principales:
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR
- `TARGET_EDGES`: Tamaño del subgrafo (30000)
- `PAGERANK_ALPHA`: Damping factor (0.85)
- `NUM_SIMULATIONS`: Simulaciones propagación (100)
- `WEB_PORT`: Puerto servidor web (8000)

### Ajustar Recursos (docker-compose.yml)

```yaml
deploy:
  resources:
    limits:
      cpus: '8.0'      # Aumentar CPUs
      memory: 16G      # Aumentar memoria
```

---

## 🐛 Troubleshooting

### ❌ Error: "Docker daemon not running"

```bash
# macOS: Iniciar Docker Desktop
open -a Docker

# Linux: Iniciar servicio
sudo systemctl start docker
```

### ❌ Error: "No such file: web-Stanford.txt"

```bash
# Descarga el dataset
# URL: https://snap.stanford.edu/data/web-Stanford.html
# Coloca en: data/raw/web-Stanford.txt
```

### ❌ Error: "Out of memory"

**Solución 1**: Reducir tamaño del subgrafo en `config/config.yaml`:

```yaml
grafo:
  target_edges: 10000  # Reducir a 10k
```

**Solución 2**: Aumentar memoria en `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 16G  # Aumentar a 16GB
```

### ❌ Error: "Permission denied" en outputs/

```bash
chmod -R 777 outputs/
```

### ❓ Visualización no se ve

```bash
# Usar servidor web
make web

# O abrir con navegador
firefox outputs/visualizaciones/fase5_grafo_interactivo.html
```

---

## 📦 Estructura de Outputs

```
outputs/
├── results/
│   ├── fase2_estadisticas.json
│   ├── fase3_metricas_centralidad.csv
│   └── fase4_analisis_top20.json
│
├── visualizaciones/
│   ├── fase5_grafo_top10.png
│   └── fase5_grafo_interactivo.html
│
├── propagacion/
│   ├── fase6_comparacion_estrategias.json
│   ├── fase6_sensibilidad_cobertura.png
│   └── fase6_sensibilidad_pasos.png
│
└── recomendaciones/
    ├── fase7_recomendaciones.json    # Para developers
    ├── fase7_recomendaciones.md      # Para PMs
    ├── fase7_accionables.csv         # Para tracking
    └── fase7_resumen_ejecutivo.txt   # Para directivos
```

---

## ✅ Checklist de Validación

```bash
# 1. Validar Docker
./docker-validate.sh

# 2. Construir imagen
make build

# 3. Verificar dataset
ls data/raw/web-Stanford.txt

# 4. Ejecutar FASE 1 (rápida)
make fase1

# 5. Verificar outputs
ls outputs/results/
ls data/processed/

# 6. Ejecutar pipeline completo (si FASE 1 funcionó)
make run

# 7. Ver visualizaciones
make web
# Abre: http://localhost:8000
```

---

## 📚 Documentación Adicional

- **Uso general**: [README.md](README.md)
- **Docker detallado**: [README_DOCKER.md](README_DOCKER.md)
- **Configuración**: [config/config.yaml](config/config.yaml)

---

## 🎉 Próximos Pasos

1. ✅ **Validar**: `./docker-validate.sh`
2. ✅ **Construir**: `make build`
3. ✅ **Ejecutar**: `make run`
4. ✅ **Ver resultados**: `make web` → http://localhost:8000
5. ✅ **Leer recomendaciones**: `cat outputs/recomendaciones/fase7_recomendaciones.md`

---

## 📞 Soporte

¿Problemas o dudas?

1. Revisa **Troubleshooting** arriba
2. Consulta [README_DOCKER.md](README_DOCKER.md)
3. Verifica logs: `make logs`
4. Contacta al autor: Danilo Melo

---

**¡El setup de Docker está completo y listo para usar! 🐳🚀**

*Generado: 2026-01-12*
