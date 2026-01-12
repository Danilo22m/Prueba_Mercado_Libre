#!/bin/bash
# Script de ayuda para ejecutar el pipeline dockerizado

set -e

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Ejercicio 1 - Docker Runner${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Verificar que existe .env
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  Archivo .env no encontrado${NC}"
    echo -e "${YELLOW}   Creando .env desde .env.example...${NC}"
    cp .env.example .env
    echo -e "${RED}   Por favor edita .env y agrega tu GROQ_API_KEY${NC}"
    exit 1
fi

# Verificar que existe el archivo de datos
if [ ! -f "data/raw/precios_historicos.csv" ]; then
    echo -e "${RED}❌ Error: data/raw/precios_historicos.csv no encontrado${NC}"
    echo -e "${YELLOW}   Por favor coloca el archivo de datos en data/raw/${NC}"
    exit 1
fi

# Función para mostrar ayuda
show_help() {
    echo "Uso: ./run_docker.sh [COMANDO]"
    echo ""
    echo "Comandos disponibles:"
    echo "  build              - Construir la imagen Docker"
    echo "  run-full           - Ejecutar pipeline completo"
    echo "  run-eval           - Ejecutar solo evaluación (sin modelos)"
    echo "  run-custom ARGS    - Ejecutar con argumentos personalizados"
    echo "  shell              - Abrir shell interactivo en el contenedor"
    echo "  clean              - Limpiar outputs"
    echo "  help               - Mostrar esta ayuda"
    echo ""
    echo "Ejemplos:"
    echo "  ./run_docker.sh build"
    echo "  ./run_docker.sh run-eval"
    echo "  ./run_docker.sh run-custom --n-productos 5 --skip-preprocessing --skip-models"
}

# Procesar comandos
case "$1" in
    build)
        echo -e "${GREEN}🔨 Construyendo imagen Docker...${NC}"
        docker-compose build
        echo -e "${GREEN}✓ Imagen construida exitosamente${NC}"
        ;;

    run-full)
        echo -e "${GREEN}🚀 Ejecutando pipeline completo...${NC}"
        docker-compose run --rm ejercicio1
        echo -e "${GREEN}✓ Pipeline completado${NC}"
        ;;

    run-eval)
        echo -e "${GREEN}📊 Ejecutando solo evaluación...${NC}"
        docker-compose run --rm ejercicio1 --skip-preprocessing --skip-models
        echo -e "${GREEN}✓ Evaluación completada${NC}"
        ;;

    run-custom)
        shift
        echo -e "${GREEN}🔧 Ejecutando con argumentos personalizados: $@${NC}"
        docker-compose run --rm ejercicio1 "$@"
        echo -e "${GREEN}✓ Ejecución completada${NC}"
        ;;

    shell)
        echo -e "${GREEN}🐚 Abriendo shell interactivo...${NC}"
        docker-compose run --rm --entrypoint /bin/bash ejercicio1
        ;;

    clean)
        echo -e "${YELLOW}🧹 Limpiando outputs...${NC}"
        rm -rf outputs/* logs/*
        echo -e "${GREEN}✓ Outputs limpiados${NC}"
        ;;

    help|--help|-h)
        show_help
        ;;

    *)
        echo -e "${RED}❌ Comando no reconocido: $1${NC}\n"
        show_help
        exit 1
        ;;
esac
