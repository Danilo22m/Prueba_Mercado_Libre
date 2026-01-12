# Ejercicio 2: Análisis de Grafos y Autoridad de URLs

Análisis completo del Stanford Web Graph Dataset usando NetworkX para identificar páginas autoridad y simular propagación de información.

## Descripción del Proyecto

Este proyecto implementa un pipeline completo de análisis de grafos de URLs web, dividido en 7 fases:

1. **FASE 1**: Selección de Subgrafo - Reduce el dataset original a un subgrafo manejable
2. **FASE 2**: Análisis Estadístico - Calcula métricas básicas del grafo
3. **FASE 3**: Métricas de Centralidad - PageRank, HITS, Betweenness, Closeness
4. **FASE 4**: Ranking Top-20 - Identifica nodos más importantes y sus roles
5. **FASE 5**: Visualización - Genera grafos estáticos e interactivos
6. **FASE 6**: Simulación de Propagación - Modelo Independent Cascade
7. **FASE 7**: Recomendaciones Accionables - Genera estrategias para e-commerce

## Requisitos del Sistema

- **Python**: 3.11+
- **Memoria RAM**: Mínimo 8GB (recomendado 16GB)
- **Espacio en disco**: 10GB libres
- **Sistema Operativo**: Linux, macOS, Windows (con WSL2)

## Instalación

### Opción 1: Ejecución Local (Desarrollo)

```bash
# 1. Clonar o descargar el proyecto
cd Ejercicio_2

# 2. Crear entorno virtual
python3.11 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar dataset (si no lo tienes)
# Descarga web-Stanford.txt de: https://snap.stanford.edu/data/web-Stanford.html
# Colócalo en: data/raw/web-Stanford.txt

# 5. Ejecutar pipeline completo
python main.py
```

### Opción 2: Ejecución con Docker (Producción) ⭐ Recomendado

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

Ver [README_DOCKER.md](README_DOCKER.md) para instrucciones detalladas de Docker.

## Uso Rápido

### Pipeline Completo

```bash
# Local
python main.py

# Docker
make run
```

### Ejecutar Fases Individuales

```bash
# Local
python src/fase1_seleccion_subgrafo.py
python src/fase2_analisis_grafo.py
python src/fase3_metricas_centralidad.py
python src/fase4_ranking_top20.py
python src/fase5_visualizacion.py
python src/fase6_propagacion.py
python src/fase7_recomendaciones_ecommerce.py

# Docker
make fase1
make fase2
# ... etc
```

## Estructura del Proyecto

```
Ejercicio_2/
├── README.md                    # Este archivo
├── README_DOCKER.md             # Documentación de Docker
├── main.py                      # Pipeline completo
├── Dockerfile                   # Imagen Docker
├── docker-compose.yml           # Orquestación Docker
├── docker-validate.sh           # Script de validación
├── Makefile                     # Comandos simplificados
├── requirements.txt             # Dependencias Python
│
├── config/
│   └── config.yaml             # Configuración centralizada
│
├── src/
│   ├── fase1_seleccion_subgrafo.py
│   ├── fase2_analisis_grafo.py
│   ├── fase3_metricas_centralidad.py
│   ├── fase4_ranking_top20.py
│   ├── fase5_visualizacion.py
│   ├── fase6_propagacion.py
│   └── fase7_recomendaciones_ecommerce.py
│
├── data/
│   ├── raw/                    # Dataset original
│   │   └── web-Stanford.txt
│   └── processed/              # Subgrafo procesado
│       └── subgrafo.gpickle
│
└── outputs/
    ├── results/                # CSVs y JSONs de resultados
    ├── visualizaciones/        # Gráficos (PNG, HTML)
    ├── propagacion/            # Resultados de simulación
    └── recomendaciones/        # Accionables para e-commerce
```

## Configuración

Toda la configuración está centralizada en `config/config.yaml`:

```yaml
# Ejemplo de configuración
grafo:
  target_edges: 30000           # Tamaño del subgrafo

metricas:
  pagerank:
    alpha: 0.85                 # Damping factor

propagacion:
  probabilidad: 0.1             # Probabilidad de activación
  num_simulaciones: 100         # Número de simulaciones

# Ver config/config.yaml para todas las opciones
```

## Resultados Generados

### Archivos Principales

#### FASE 1 - Selección
- `data/processed/subgrafo.gpickle` - Subgrafo seleccionado

#### FASE 2 - Análisis
- `outputs/results/fase2_estadisticas.json` - Estadísticas del grafo

#### FASE 3 - Métricas
- `outputs/results/fase3_metricas_centralidad.csv` - Todas las métricas

#### FASE 4 - Ranking
- `outputs/results/fase4_analisis_top20.json` - Análisis de roles

#### FASE 5 - Visualización
- `outputs/visualizaciones/fase5_grafo_top10.png` - Visualización estática
- `outputs/visualizaciones/fase5_grafo_interactivo.html` - Visualización interactiva

#### FASE 6 - Propagación
- `outputs/propagacion/fase6_comparacion_estrategias.json` - Comparación de estrategias
- `outputs/propagacion/fase6_sensibilidad_*.png` - Gráficos de sensibilidad

#### FASE 7 - Recomendaciones
- `outputs/recomendaciones/fase7_recomendaciones.json` - Formato técnico
- `outputs/recomendaciones/fase7_recomendaciones.md` - Formato legible
- `outputs/recomendaciones/fase7_accionables.csv` - Para tracking
- `outputs/recomendaciones/fase7_resumen_ejecutivo.txt` - Para directivos

## Comandos Útiles (Makefile)

```bash
make help           # Ver todos los comandos
make build          # Construir imagen Docker
make run            # Ejecutar pipeline completo
make fase1          # Ejecutar solo FASE 1
make web            # Servidor web (puerto 8000)
make shell          # Shell interactivo
make clean          # Limpiar outputs
make clean-all      # Limpiar todo
```

## Visualizaciones

### Visualización Estática (PNG)
- Grafo de los Top-10 nodos con sus vecinos
- Tamaño de nodo proporcional a PageRank
- Colores según roles (Autoridad, Hub, Puente, etc.)

### Visualización Interactiva (HTML)
- Grafo interactivo con física de layout
- Zoom, pan, selección de nodos
- Tooltips con métricas detalladas
- Filtros por tipo de rol

Para ver las visualizaciones:

```bash
# Opción 1: Servidor web con Docker
make web
# Abre http://localhost:8000/fase5_grafo_interactivo.html

# Opción 2: Abrir directamente
open outputs/visualizaciones/fase5_grafo_interactivo.html
```

## Recomendaciones Generadas

La FASE 7 genera 3 accionables principales:

1. **ACT-001: Link Building Interno**
   - Problema: 14% de páginas sin enlaces internos
   - ROI: Alto
   - KPI: Reducir huérfanos, aumentar PageRank +20%

2. **ACT-002: Campañas Virales Optimizadas**
   - Estrategia: Usar nodos top por Betweenness
   - ROI: Muy Alto
   - KPI: Cobertura viral +2.7%, viralización +30%

3. **ACT-003: Potenciar Páginas Autoridad**
   - Oportunidad: 3 páginas de alta autoridad identificadas
   - ROI: Alto
   - KPI: Conversión +15%, engagement +25%

Ver detalles en: `outputs/recomendaciones/fase7_recomendaciones.md`

## Troubleshooting

### Error: "No module named 'yaml'"

```bash
# Local
pip install -r requirements.txt

# Docker
make build
```

### Error: "KeyError: 'betweenness_centrality'"

Ya está corregido. Actualiza a la última versión del código.

### Error: "Out of memory"

**Solución 1**: Reduce el tamaño del subgrafo en `config/config.yaml`:

```yaml
grafo:
  target_edges: 10000  # Reducir a 10k aristas
```

**Solución 2**: Aumenta memoria en Docker:

```yaml
# docker-compose.yml
deploy:
  resources:
    limits:
      memory: 16G
```

### Visualización interactiva no se ve

```bash
# Usa el servidor web
make web

# O abre con navegador
firefox outputs/visualizaciones/fase5_grafo_interactivo.html
```

## Tecnologías Utilizadas

- **NetworkX 3.2.1**: Análisis de grafos
- **Pandas 2.1.4**: Manipulación de datos
- **NumPy 1.26.3**: Operaciones numéricas
- **Matplotlib 3.8.2**: Visualizaciones estáticas
- **Pyvis 0.3.2**: Visualizaciones interactivas
- **PyYAML 6.0.1**: Configuración
- **Docker**: Contenedorización

## Dataset

**Nombre**: Stanford Web Graph
**Fuente**: https://snap.stanford.edu/data/web-Stanford.html
**Descripción**: Grafo de enlaces web del dominio stanford.edu (2002)
**Tamaño**: 281,903 nodos, 2,312,497 aristas
**Formato**: Lista de aristas (from_node to_node)

## Métricas Implementadas

### Centralidad
- **PageRank**: Importancia global (influencia)
- **In-Degree**: Popularidad (enlaces entrantes)
- **Out-Degree**: Actividad (enlaces salientes)
- **HITS Authority**: Páginas de referencia (destinos)
- **HITS Hub**: Páginas índice (directorios)
- **Betweenness**: Nodos puente (conectores)
- **Closeness**: Distancia promedio (accesibilidad)

### Roles Identificados
- **Autoridad**: Alta authority, alto in-degree
- **Hub**: Alto hub, alto out-degree
- **Puente**: Alto betweenness
- **Propagador**: Alta closeness
- **Multi-Rol**: Combina varios roles

## Rendimiento

### Tiempos de Ejecución (Aproximados)

| Fase | Tiempo | Memoria |
|------|--------|---------|
| FASE 1 | 2-5 min | 2-4 GB |
| FASE 2 | 10-30 seg | 1-2 GB |
| FASE 3 | 5-15 min | 2-4 GB |
| FASE 4 | 10-30 seg | 1-2 GB |
| FASE 5 | 1-2 min | 1-2 GB |
| FASE 6 | 10-30 min | 2-4 GB |
| FASE 7 | 10-30 seg | 1 GB |
| **Total** | **20-60 min** | **8-16 GB** |

*Nota: Tiempos varían según CPU, RAM y tamaño del subgrafo*

## Autor

**Danilo Melo**
Fecha: 2026-01-12

## Licencia

Este proyecto es parte de una prueba técnica para Mercado Libre.

## Siguientes Pasos

1. **Ejecutar pipeline**: `make run` o `python main.py`
2. **Revisar resultados**: `ls -R outputs/`
3. **Ver visualizaciones**: `make web` → http://localhost:8000
4. **Leer recomendaciones**: `cat outputs/recomendaciones/fase7_recomendaciones.md`
5. **Implementar accionables**: Usar KPIs de FASE 7

## Soporte

Para dudas o problemas:
1. Revisa [README_DOCKER.md](README_DOCKER.md) para Docker
2. Revisa la sección Troubleshooting arriba
3. Verifica logs: `make logs` (Docker) o revisa consola (local)
4. Contacta al autor

---

