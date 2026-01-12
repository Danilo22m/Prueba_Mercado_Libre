# Makefile para Ejercicio 2: Análisis de Grafos
# Simplifica comandos de Docker y ejecución

.PHONY: help build run run-pipeline run-fase clean logs shell web stop

# Variables
IMAGE_NAME = ejercicio2-grafos
CONTAINER_NAME = ejercicio2-grafos

help:
	@echo "Comandos disponibles:"
	@echo "  make build          - Construir imagen Docker"
	@echo "  make run            - Ejecutar pipeline completo (todas las fases)"
	@echo "  make run-fase N=X   - Ejecutar solo la fase X (1-7)"
	@echo "  make shell          - Abrir shell interactivo en el contenedor"
	@echo "  make logs           - Ver logs del contenedor"
	@echo "  make web            - Iniciar servidor web para visualizaciones (puerto 8000)"
	@echo "  make stop           - Detener contenedores"
	@echo "  make clean          - Limpiar outputs generados"
	@echo "  make clean-all      - Limpiar todo (outputs + imágenes Docker)"

build:
	@echo "Construyendo imagen Docker..."
	docker-compose build grafos-analisis

run: build
	@echo "Ejecutando pipeline completo..."
	docker-compose run --rm grafos-analisis python main.py

run-pipeline: run

run-fase:
	@if [ -z "$(N)" ]; then \
		echo "Error: Especifica el número de fase con N=X"; \
		echo "Ejemplo: make run-fase N=1"; \
		exit 1; \
	fi
	@echo "Ejecutando FASE $(N)..."
	docker-compose run --rm grafos-analisis python src/fase$(N)_*.py

shell:
	@echo "Abriendo shell en el contenedor..."
	docker-compose run --rm grafos-analisis /bin/bash

logs:
	docker-compose logs -f grafos-analisis

web:
	@echo "Iniciando servidor web en http://localhost:8000"
	@echo "Visualizaciones disponibles en:"
	@echo "  - http://localhost:8000/fase5_grafo_interactivo.html"
	@echo "  - http://localhost:8000/fase5_grafo_top10.png"
	docker-compose --profile web up web-viewer

stop:
	@echo "Deteniendo contenedores..."
	docker-compose down

clean:
	@echo "Limpiando outputs generados..."
	rm -rf outputs/results/*
	rm -rf outputs/visualizaciones/*
	rm -rf outputs/propagacion/*
	rm -rf outputs/recomendaciones/*
	rm -rf data/processed/*
	@echo "Outputs limpiados."

clean-all: clean
	@echo "Limpiando imágenes Docker..."
	docker-compose down --rmi all --volumes
	@echo "Todo limpiado."

# Alias
fase1:
	@make run-fase N=1

fase2:
	@make run-fase N=2

fase3:
	@make run-fase N=3

fase4:
	@make run-fase N=4

fase5:
	@make run-fase N=5

fase6:
	@make run-fase N=6

fase7:
	@make run-fase N=7
