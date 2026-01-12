#!/bin/bash
# Script de validación pre-Docker
# Verifica que todo esté listo para ejecutar con Docker

set -e

echo "=========================================="
echo "VALIDACIÓN PRE-DOCKER"
echo "=========================================="
echo ""

# Colores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para checks
check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1"
        return 1
    fi
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# 1. Verificar Docker
echo "1. Verificando Docker..."
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    check "Docker instalado: $DOCKER_VERSION"
else
    check "Docker instalado" && false
    echo "   Por favor instala Docker desde: https://docs.docker.com/get-docker/"
    exit 1
fi

# 2. Verificar Docker Compose
echo ""
echo "2. Verificando Docker Compose..."
if command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version)
    check "Docker Compose instalado: $COMPOSE_VERSION"
else
    check "Docker Compose instalado" && false
    echo "   Por favor instala Docker Compose"
    exit 1
fi

# 3. Verificar que Docker está corriendo
echo ""
echo "3. Verificando estado de Docker..."
if docker info &> /dev/null; then
    check "Docker daemon está corriendo"
else
    check "Docker daemon está corriendo" && false
    echo "   Por favor inicia Docker Desktop o el servicio de Docker"
    exit 1
fi

# 4. Verificar estructura del proyecto
echo ""
echo "4. Verificando estructura del proyecto..."

FILES_REQUIRED=(
    "Dockerfile"
    "docker-compose.yml"
    ".dockerignore"
    "requirements.txt"
    "config/config.yaml"
    "main.py"
)

ALL_FILES_OK=true
for file in "${FILES_REQUIRED[@]}"; do
    if [ -f "$file" ]; then
        check "Archivo: $file"
    else
        check "Archivo: $file" && false
        ALL_FILES_OK=false
    fi
done

# 5. Verificar dataset
echo ""
echo "5. Verificando dataset..."
if [ -f "data/raw/web-Stanford.txt" ]; then
    SIZE=$(du -h data/raw/web-Stanford.txt | cut -f1)
    check "Dataset encontrado: data/raw/web-Stanford.txt ($SIZE)"
else
    warn "Dataset NO encontrado: data/raw/web-Stanford.txt"
    echo "   Descarga el dataset desde: https://snap.stanford.edu/data/web-Stanford.html"
    echo "   O ejecuta FASE 1 con dataset reducido"
fi

# 6. Verificar directorios de output
echo ""
echo "6. Verificando directorios de output..."
DIRS=(
    "data/raw"
    "data/processed"
    "outputs/results"
    "outputs/visualizaciones"
    "outputs/propagacion"
    "outputs/recomendaciones"
)

for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        check "Creado directorio: $dir"
    else
        check "Directorio existe: $dir"
    fi
done

# 7. Verificar scripts de fases
echo ""
echo "7. Verificando scripts de fases..."
PHASE_SCRIPTS=(
    "src/fase1_seleccion_subgrafo.py"
    "src/fase2_analisis_grafo.py"
    "src/fase3_metricas_centralidad.py"
    "src/fase4_ranking_top20.py"
    "src/fase5_visualizacion.py"
    "src/fase6_propagacion.py"
    "src/fase7_recomendaciones_ecommerce.py"
)

ALL_PHASES_OK=true
for script in "${PHASE_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        check "Script: $(basename $script)"
    else
        check "Script: $(basename $script)" && false
        ALL_PHASES_OK=false
    fi
done

# 8. Verificar espacio en disco
echo ""
echo "8. Verificando espacio en disco..."
AVAILABLE_SPACE=$(df -h . | awk 'NR==2 {print $4}')
echo "   Espacio disponible: $AVAILABLE_SPACE"
check "Verificación de espacio en disco"

# 9. Verificar memoria disponible
echo ""
echo "9. Verificando memoria del sistema..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    TOTAL_MEM=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)"GB"}')
    echo "   Memoria total: $TOTAL_MEM"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    TOTAL_MEM=$(free -h | awk 'NR==2 {print $2}')
    echo "   Memoria total: $TOTAL_MEM"
fi
check "Verificación de memoria"

# Resumen final
echo ""
echo "=========================================="
echo "RESUMEN"
echo "=========================================="

if [ "$ALL_FILES_OK" = true ] && [ "$ALL_PHASES_OK" = true ]; then
    echo -e "${GREEN}✓ Todos los checks pasaron correctamente${NC}"
    echo ""
    echo "Próximos pasos:"
    echo "  1. Construir imagen: make build"
    echo "  2. Ejecutar pipeline: make run"
    echo "  3. Ver visualizaciones: make web"
    echo ""
    echo "O ejecuta directamente:"
    echo "  docker-compose run --rm grafos-analisis python main.py"
    exit 0
else
    echo -e "${RED}✗ Algunos checks fallaron${NC}"
    echo ""
    echo "Por favor corrige los problemas antes de continuar."
    exit 1
fi
