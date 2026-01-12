# Makefile - Sistema RAG + Agente Critico
# Ejercicio 3 - Prueba Tecnica Mercado Libre

.PHONY: help build run shell logs stop clean clean-all fase1 fase2 fase3 fase4 fase5 fase6 test-groq

# Variables
DOCKER_COMPOSE = docker-compose
SERVICE = rag-pipeline

# Colores para output
GREEN = \033[0;32m
YELLOW = \033[0;33m
NC = \033[0m # No Color

help: ## Mostrar esta ayuda
	@echo "$(GREEN)Sistema RAG + Agente Critico - Comandos Disponibles$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Ejemplos:$(NC)"
	@echo "  make build      # Construir imagen Docker"
	@echo "  make run        # Ejecutar pipeline completo"
	@echo "  make fase1      # Ejecutar solo FASE 1"

# ==============================================================================
# CONSTRUCCION
# ==============================================================================

build: ## Construir imagen Docker
	@echo "$(GREEN)Construyendo imagen Docker...$(NC)"
	$(DOCKER_COMPOSE) build

build-no-cache: ## Construir imagen sin cache
	@echo "$(GREEN)Construyendo imagen Docker (sin cache)...$(NC)"
	$(DOCKER_COMPOSE) build --no-cache

# ==============================================================================
# EJECUCION
# ==============================================================================

run: ## Ejecutar pipeline completo (FASE 1-6)
	@echo "$(GREEN)Ejecutando pipeline completo...$(NC)"
	$(DOCKER_COMPOSE) run --rm $(SERVICE) python main.py

fase1: ## Ejecutar FASE 1: Ingesta y Normalizacion
	@echo "$(GREEN)Ejecutando FASE 1: Ingesta y Normalizacion...$(NC)"
	$(DOCKER_COMPOSE) run --rm $(SERVICE) python src/fase1_ingesta.py

fase2: ## Ejecutar FASE 2: Chunking e Indexacion
	@echo "$(GREEN)Ejecutando FASE 2: Chunking e Indexacion...$(NC)"
	$(DOCKER_COMPOSE) run --rm $(SERVICE) python src/fase2_chunking.py

fase3: ## Ejecutar FASE 3: Retrieval (test)
	@echo "$(GREEN)Ejecutando FASE 3: Retrieval...$(NC)"
	$(DOCKER_COMPOSE) run --rm $(SERVICE) python src/fase3_retrieval.py

fase4: ## Ejecutar FASE 4: Generation (requiere API key)
	@echo "$(GREEN)Ejecutando FASE 4: Generation...$(NC)"
	$(DOCKER_COMPOSE) run --rm $(SERVICE) python src/fase4_generation.py

fase5: ## Ejecutar FASE 5: Agente Critico (test simulado)
	@echo "$(GREEN)Ejecutando FASE 5: Agente Critico...$(NC)"
	$(DOCKER_COMPOSE) run --rm $(SERVICE) python src/fase5_agente_critico.py

fase6: ## Ejecutar FASE 6: Evaluacion completa (requiere API key)
	@echo "$(GREEN)Ejecutando FASE 6: Evaluacion...$(NC)"
	$(DOCKER_COMPOSE) run --rm $(SERVICE) python src/fase6_evaluacion.py

test-groq: ## Probar conexion con API de Groq
	@echo "$(GREEN)Probando conexion con Groq API...$(NC)"
	$(DOCKER_COMPOSE) run --rm $(SERVICE) python test_groq.py

# ==============================================================================
# UTILIDADES
# ==============================================================================

shell: ## Abrir shell interactivo en el contenedor
	@echo "$(GREEN)Abriendo shell...$(NC)"
	$(DOCKER_COMPOSE) run --rm $(SERVICE) /bin/bash

logs: ## Ver logs del contenedor
	$(DOCKER_COMPOSE) logs -f

stop: ## Detener contenedores
	@echo "$(GREEN)Deteniendo contenedores...$(NC)"
	$(DOCKER_COMPOSE) down

# ==============================================================================
# LIMPIEZA
# ==============================================================================

clean: ## Limpiar outputs generados
	@echo "$(GREEN)Limpiando outputs...$(NC)"
	rm -rf outputs/results/*
	rm -rf outputs/logs/*
	rm -rf outputs/models/*
	rm -rf data/processed/*
	@echo "$(GREEN)Outputs limpiados$(NC)"

clean-all: clean ## Limpiar todo (outputs + imagenes Docker)
	@echo "$(GREEN)Limpiando imagenes Docker...$(NC)"
	$(DOCKER_COMPOSE) down --rmi all --volumes
	@echo "$(GREEN)Todo limpio$(NC)"

# ==============================================================================
# VERIFICACION
# ==============================================================================

check-env: ## Verificar que .env existe
	@if [ -f .env ]; then \
		echo "$(GREEN).env encontrado$(NC)"; \
	else \
		echo "$(YELLOW)ADVERTENCIA: .env no encontrado. Copia .env.example a .env$(NC)"; \
	fi

check-data: ## Verificar que el dataset existe
	@if [ -f data/raw/Laptops_with_technical_specifications.csv ]; then \
		echo "$(GREEN)Dataset encontrado$(NC)"; \
	else \
		echo "$(YELLOW)ADVERTENCIA: Dataset no encontrado en data/raw/$(NC)"; \
	fi

verify: check-env check-data ## Verificar requisitos
	@echo "$(GREEN)Verificacion completada$(NC)"
