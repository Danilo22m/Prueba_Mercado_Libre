# Docker - Ejercicio 2: Análisis de Grafos de URLs

Este documento explica cómo ejecutar el proyecto usando Docker.

## Requisitos Previos

- Docker instalado (versión 20.10+)
- Docker Compose instalado (versión 2.0+)
- Al menos 8GB de RAM disponible
- 10GB de espacio en disco

## Estructura del Proyecto

```
Ejercicio_2/
├── Dockerfile              # Definición de la imagen Docker
├── docker-compose.yml      # Orquestación de servicios
├── .dockerignore          # Archivos excluidos del build
├── Makefile               # Comandos simplificados
├── main.py                # Pipeline completo
├── config/
│   └── config.yaml        # Configuración del proyecto
├── src/                   # Scripts de cada fase
├── data/
│   ├── raw/              # Dataset original
│   └── processed/        # Datos procesados
└── outputs/              # Resultados generados
    ├── results/
    ├── visualizaciones/
    ├── propagacion/
    └── recomendaciones/
```

## Inicio Rápido

### 1. Preparar el Dataset

Asegúrate de tener el dataset en `data/raw/web-Stanford.txt`:

```bash
ls -lh data/raw/web-Stanford.txt
```

### 2. Construir la Imagen Docker

```bash
# Opción 1: Usando Makefile (recomendado)
make build

# Opción 2: Usando docker-compose directamente
docker-compose build
```

### 3. Ejecutar el Pipeline Completo

```bash
# Opción 1: Usando Makefile (recomendado)
make run

# Opción 2: Usando docker-compose
docker-compose run --rm grafos-analisis python main.py
```

Esto ejecutará todas las fases en secuencia:
1. Selección de subgrafo
2. Análisis estadístico
3. Métricas de centralidad
4. Ranking Top-20
5. Visualización
6. Simulación de propagación
7. Recomendaciones

## Comandos Disponibles (Makefile)

### Ejecución

```bash
# Ver ayuda
make help

# Ejecutar pipeline completo
make run

# Ejecutar solo una fase específica
make run-fase N=1  # FASE 1: Selección de subgrafo
make run-fase N=2  # FASE 2: Análisis estadístico
make run-fase N=3  # FASE 3: Métricas de centralidad
make run-fase N=4  # FASE 4: Ranking Top-20
make run-fase N=5  # FASE 5: Visualización
make run-fase N=6  # FASE 6: Propagación
make run-fase N=7  # FASE 7: Recomendaciones

# O usando alias
make fase1
make fase2
# ... etc
```

### Servidor Web para Visualizaciones

```bash
# Iniciar servidor web en http://localhost:8000
make web

# Acceder a:
# http://localhost:8000/fase5_grafo_interactivo.html
# http://localhost:8000/fase5_grafo_top10.png
```

### Utilidades

```bash
# Abrir shell interactivo en el contenedor
make shell

# Ver logs en tiempo real
make logs

# Detener contenedores
make stop

# Limpiar outputs generados
make clean

# Limpiar todo (outputs + imágenes Docker)
make clean-all
```

## Comandos Docker Avanzados

### Sin Makefile

Si prefieres usar Docker directamente:

```bash
# Construir imagen
docker-compose build grafos-analisis

# Ejecutar pipeline completo
docker-compose run --rm grafos-analisis python main.py

# Ejecutar fase específica
docker-compose run --rm grafos-analisis python src/fase1_seleccion_subgrafo.py

# Shell interactivo
docker-compose run --rm grafos-analisis /bin/bash

# Servidor web
docker-compose --profile web up web-viewer

# Detener todo
docker-compose down
```

### Personalización Avanzada

#### Modificar Recursos (CPU/Memoria)

Edita `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '8.0'      # Aumentar CPUs
      memory: 16G      # Aumentar memoria
```

#### Ejecutar con Variables de Entorno

```bash
docker-compose run --rm \
  -e LOG_LEVEL=DEBUG \
  grafos-analisis python main.py
```

#### Montar Configuración Personalizada

```bash
docker-compose run --rm \
  -v ./mi_config.yaml:/app/config/config.yaml \
  grafos-analisis python main.py
```

## Volúmenes y Persistencia

Los siguientes directorios están montados como volúmenes:

- `./data` → `/app/data` (Dataset de entrada)
- `./outputs` → `/app/outputs` (Resultados generados)
- `./config` → `/app/config` (Configuración)

Esto significa que:
- Los resultados se guardan en tu máquina local
- Puedes modificar la configuración sin reconstruir la imagen
- El dataset no se duplica dentro del contenedor

## Troubleshooting

### Error: "No such file or directory: data/raw/web-Stanford.txt"

**Solución**: Asegúrate de tener el dataset en el lugar correcto:

```bash
ls data/raw/web-Stanford.txt
```

Si no existe, descárgalo desde: https://snap.stanford.edu/data/web-Stanford.html

### Error: "Out of memory"

**Solución 1**: Aumenta la memoria en `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 16G  # Aumentar a 16GB
```

**Solución 2**: Ejecuta solo fases específicas en lugar del pipeline completo.

### Error: "Permission denied" en outputs/

**Solución**: Cambia permisos del directorio:

```bash
chmod -R 777 outputs/
```

### La visualización interactiva no se ve

**Solución**: Usa el servidor web:

```bash
make web
# Luego abre: http://localhost:8000/fase5_grafo_interactivo.html
```

## Limpieza

```bash
# Limpiar solo outputs
make clean

# Limpiar todo (outputs + imágenes Docker)
make clean-all

# O manualmente:
docker-compose down --rmi all --volumes
rm -rf outputs/results/* outputs/visualizaciones/* outputs/propagacion/* outputs/recomendaciones/*
```

## Verificación de la Instalación

Para verificar que Docker está correctamente configurado:

```bash
# Verificar versiones
docker --version
docker-compose --version

# Construir imagen de prueba
make build

# Ejecutar solo FASE 1 (rápida)
make fase1

# Verificar outputs
ls -lh outputs/results/
ls -lh data/processed/
```

## Comparación: Docker vs Local

### Docker (Recomendado para Producción)
✅ Entorno reproducible
✅ Sin conflictos de dependencias
✅ Fácil de compartir
✅ Aislamiento completo
❌ Overhead de virtualización

### Local (Recomendado para Desarrollo)
✅ Ejecución más rápida
✅ Debugging más fácil
❌ Requiere configurar entorno
❌ Posibles conflictos de versiones

## Siguientes Pasos

1. Ejecuta el pipeline completo: `make run`
2. Revisa los resultados en `outputs/`
3. Abre las visualizaciones con: `make web`
4. Lee las recomendaciones en: `outputs/recomendaciones/fase7_recomendaciones.md`

## Soporte

Para más información, consulta:
- Documentación de Docker: https://docs.docker.com/
- Documentación de Docker Compose: https://docs.docker.com/compose/
- Issues del proyecto: [URL del repositorio]
