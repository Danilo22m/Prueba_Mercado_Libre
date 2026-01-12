#!/bin/bash
# Quick Start Script - Inicio rápido del proyecto
# Este script detecta el entorno y ejecuta el método más apropiado

set -e

echo "========================================"
echo "EJERCICIO 2: ANÁLISIS DE GRAFOS"
echo "Quick Start"
echo "========================================"
echo ""

# Detectar si Docker está disponible
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo "✓ Docker detectado"
    USE_DOCKER=true
else
    echo "✗ Docker no detectado"
    USE_DOCKER=false
fi

# Detectar si hay entorno virtual Python
if [ -d "venv" ]; then
    echo "✓ Entorno virtual Python detectado"
    USE_VENV=true
else
    echo "✗ Entorno virtual Python no detectado"
    USE_VENV=false
fi

echo ""
echo "Opciones de ejecución:"
echo "  1. Docker (recomendado para producción)"
echo "  2. Python local (recomendado para desarrollo)"
echo "  3. Validar entorno Docker"
echo "  4. Crear entorno virtual Python"
echo "  5. Salir"
echo ""

read -p "Selecciona una opción [1-5]: " OPTION

case $OPTION in
    1)
        if [ "$USE_DOCKER" = true ]; then
            echo ""
            echo "Ejecutando con Docker..."
            echo "Esto puede tardar 20-60 minutos en la primera ejecución."
            echo ""
            read -p "¿Continuar? [y/n]: " CONFIRM
            if [ "$CONFIRM" = "y" ] || [ "$CONFIRM" = "Y" ]; then
                echo ""
                echo "1/3 Construyendo imagen Docker..."
                make build

                echo ""
                echo "2/3 Validando entorno..."
                ./docker-validate.sh || true

                echo ""
                echo "3/3 Ejecutando pipeline completo..."
                make run

                echo ""
                echo "=========================================="
                echo "¡COMPLETADO!"
                echo "=========================================="
                echo ""
                echo "Resultados disponibles en:"
                echo "  - outputs/results/"
                echo "  - outputs/visualizaciones/"
                echo "  - outputs/propagacion/"
                echo "  - outputs/recomendaciones/"
                echo ""
                echo "Para ver visualizaciones interactivas:"
                echo "  make web"
                echo "  Luego abre: http://localhost:8000"
            else
                echo "Cancelado."
            fi
        else
            echo ""
            echo "Error: Docker no está instalado."
            echo "Por favor instala Docker desde: https://docs.docker.com/get-docker/"
        fi
        ;;

    2)
        if [ "$USE_VENV" = true ]; then
            echo ""
            echo "Ejecutando con Python local..."
            echo "Esto puede tardar 20-60 minutos."
            echo ""
            read -p "¿Continuar? [y/n]: " CONFIRM
            if [ "$CONFIRM" = "y" ] || [ "$CONFIRM" = "Y" ]; then
                echo ""
                echo "Activando entorno virtual..."
                source venv/bin/activate

                echo ""
                echo "Ejecutando pipeline completo..."
                python main.py

                echo ""
                echo "=========================================="
                echo "¡COMPLETADO!"
                echo "=========================================="
                echo ""
                echo "Resultados disponibles en:"
                echo "  - outputs/results/"
                echo "  - outputs/visualizaciones/"
                echo "  - outputs/propagacion/"
                echo "  - outputs/recomendaciones/"
                echo ""
                echo "Para ver visualizaciones:"
                echo "  open outputs/visualizaciones/fase5_grafo_interactivo.html"
            else
                echo "Cancelado."
            fi
        else
            echo ""
            echo "Error: No se encontró entorno virtual."
            echo "Primero crea uno con la opción 4."
        fi
        ;;

    3)
        if [ "$USE_DOCKER" = true ]; then
            echo ""
            echo "Validando entorno Docker..."
            ./docker-validate.sh
        else
            echo ""
            echo "Error: Docker no está instalado."
            echo "Por favor instala Docker desde: https://docs.docker.com/get-docker/"
        fi
        ;;

    4)
        echo ""
        echo "Creando entorno virtual Python..."

        # Verificar Python 3.11+
        if command -v python3.11 &> /dev/null; then
            PYTHON_CMD=python3.11
        elif command -v python3 &> /dev/null; then
            PYTHON_CMD=python3
        else
            echo "Error: Python 3 no encontrado."
            echo "Por favor instala Python 3.11+"
            exit 1
        fi

        echo "Usando: $($PYTHON_CMD --version)"

        # Crear venv
        $PYTHON_CMD -m venv venv

        # Activar y actualizar pip
        source venv/bin/activate
        pip install --upgrade pip

        # Instalar dependencias
        echo ""
        echo "Instalando dependencias..."
        pip install -r requirements.txt

        echo ""
        echo "✓ Entorno virtual creado exitosamente"
        echo ""
        echo "Para activarlo manualmente:"
        echo "  source venv/bin/activate"
        echo ""
        echo "Ahora puedes ejecutar el pipeline con la opción 2."
        ;;

    5)
        echo "Saliendo..."
        exit 0
        ;;

    *)
        echo "Opción inválida."
        exit 1
        ;;
esac

echo ""
echo "========================================"
