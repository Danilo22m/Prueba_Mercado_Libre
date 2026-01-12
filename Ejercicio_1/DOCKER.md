# Docker - Guía de Uso

Esta guía explica cómo ejecutar el proyecto usando Docker.

## Pre-requisitos

- Docker instalado (versión 20.10+)
- Docker Compose instalado (versión 1.29+)
- API Key de Groq (para usar el Modelo A - LLM)

## Configuración Inicial

### 1. Crear archivo .env

```bash
cp .env.example .env
```

Edita `.env` y agrega tu API key de Groq:

```bash
GROQ_API_KEY=tu_api_key_aqui
```

### 2. Verificar que existe el archivo de datos

Asegúrate de que existe `data/raw/precios_historicos.csv`

## Uso Básico

### Construir la imagen

```bash
./run_docker.sh build
```

O manualmente:

```bash
docker-compose build
```

### Ejecutar solo evaluación (recomendado si ya tienes predicciones)

```bash
./run_docker.sh run-eval
```

O manualmente:

```bash
docker-compose run --rm ejercicio1 --skip-preprocessing --skip-models
```

### Ejecutar pipeline completo

```bash
./run_docker.sh run-full
```

O manualmente:

```bash
docker-compose run --rm ejercicio1
```

## Comandos Avanzados

### Ejecutar con argumentos personalizados

```bash
./run_docker.sh run-custom --n-productos 5 --skip-preprocessing --skip-models
```

O manualmente:

```bash
docker-compose run --rm ejercicio1 --n-productos 5 --skip-preprocessing --skip-models
```

### Abrir shell interactivo

```bash
./run_docker.sh shell
```

O manualmente:

```bash
docker-compose run --rm --entrypoint /bin/bash ejercicio1
```

### Limpiar outputs

```bash
./run_docker.sh clean
```

## Argumentos Disponibles

El contenedor acepta los mismos argumentos que `main.py`:

- `--config PATH`: Ruta a archivo de configuración personalizado
- `--skip-preprocessing`: Omitir preprocesamiento
- `--skip-models`: Omitir entrenamiento de modelos
- `--n-productos N`: Número de productos para visualizar (default: 3)

## Volúmenes

Los siguientes directorios están montados como volúmenes:

- `./data/raw` → `/app/data/raw` (read-only)
- `./data/processed` → `/app/data/processed` (read-write)
- `./outputs` → `/app/outputs` (read-write)
- `./logs` → `/app/logs` (read-write)
- `./config` → `/app/config` (read-only)

Esto permite:
- Leer datos de entrada desde tu máquina local
- Persistir resultados en tu máquina local
- Modificar configuración sin reconstruir la imagen

## Estructura de la Imagen

La imagen Docker incluye:

- Python 3.11-slim como base
- Todas las dependencias de `requirements.txt`
- Código fuente del proyecto
- Configuración por defecto

## Troubleshooting

### Error: GROQ_API_KEY no encontrada

Asegúrate de haber creado el archivo `.env` con tu API key.

### Error: data/raw/precios_historicos.csv no encontrado

Coloca el archivo de datos en el directorio `data/raw/`.

### Error: Permission denied en outputs/

Dale permisos de escritura al directorio:

```bash
chmod -R 777 outputs logs data/processed
```

### Reconstruir imagen después de cambios

```bash
docker-compose build --no-cache
```

## Limpieza

### Eliminar contenedor

```bash
docker-compose down
```

### Eliminar imagen

```bash
docker rmi ejercicio1:latest
```

### Limpiar todo (contenedores, imágenes, volúmenes)

```bash
docker-compose down -v
docker system prune -a
```
