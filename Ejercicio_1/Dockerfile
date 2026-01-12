# Dockerfile para Ejercicio 1 - Comparación LLM vs Modelo Clásico
FROM python:3.11-slim

# Metadata
LABEL maintainer="Danilo Melo"
LABEL description="Ejercicio 1: Comparación LLM vs Modelo Clásico para Detección de Anomalías"
LABEL version="1.0"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Directorio de trabajo
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias del sistema necesarias para matplotlib y scipy
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    apt-get remove -y gcc g++ && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copiar código fuente
COPY config/ ./config/
COPY src/ ./src/
COPY main.py .

# Crear directorios necesarios
RUN mkdir -p data/raw data/processed outputs/results outputs/plots outputs/models logs

# Punto de entrada
ENTRYPOINT ["python", "main.py"]

# Comando por defecto (puede ser sobrescrito)
CMD ["--skip-preprocessing", "--skip-models"]
