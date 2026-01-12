# Dockerfile para Sistema RAG + Agente Critico
# Ejercicio 3 - Prueba Tecnica Mercado Libre

FROM python:3.11-slim

# Metadata
LABEL maintainer="Danilo Melo"
LABEL description="Sistema RAG + Agente Critico para Q&A de Laptops"

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (para cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Crear directorios necesarios
RUN mkdir -p data/raw data/processed outputs/logs outputs/results outputs/models

# Comando por defecto
CMD ["python", "main.py"]
