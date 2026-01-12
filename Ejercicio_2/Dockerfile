# Dockerfile para Ejercicio 2: Análisis de Grafos de URLs
# Imagen base: Python 3.11 slim
FROM python:3.11-slim

# Metadata
LABEL maintainer="Danilo Melo"
LABEL description="Análisis de Grafos de URLs - Stanford Web Graph Dataset"
LABEL version="1.0"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para NetworkX y visualización
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt primero (para aprovechar cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar archivos del proyecto
COPY config/ config/
COPY src/ src/
COPY main.py .

# Crear directorios de salida
RUN mkdir -p data/raw data/processed outputs/results outputs/visualizaciones outputs/propagacion outputs/recomendaciones

# Volúmenes para datos y resultados
VOLUME ["/app/data", "/app/outputs"]

# Puerto expuesto (por si se necesita servir visualizaciones)
EXPOSE 8000

# Comando por defecto: ejecutar el pipeline completo
CMD ["python", "main.py"]
