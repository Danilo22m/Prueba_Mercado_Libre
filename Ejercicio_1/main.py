#!/usr/bin/env python3
"""
EJERCICIO 1: COMPARACIÓN LLM vs MODELO CLÁSICO PARA DETECCIÓN DE ANOMALÍAS

Pipeline completo que ejecuta:
1. Preprocesamiento de datos
2. Entrenamiento y predicción con Modelo A (LLM)
3. Entrenamiento y predicción con Modelo B (Isolation Forest)
4. Evaluación comparativa
5. A/B Testing estadístico
6. Visualizaciones de series temporales

Autor: Danilo Melo
Fecha: 2026-01-11
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Agregar src al path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / 'src'))

from preprocessing import ejecutar_preprocesamiento
from model_a_llm import ejecutar_modelo_a
from model_b_classic import ejecutar_modelo_b
from evaluation import ejecutar_evaluacion
from ab_test import ejecutar_ab_test
from visualizaciones_series import crear_visualizaciones_series_temporales

# Directorios
LOGS_DIR = BASE_DIR / 'outputs' / 'logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging() -> logging.Logger:
    """Configura logging dual: consola + archivo"""
    # Crear nombre de archivo con timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOGS_DIR / f'pipeline_completo_{timestamp}.log'

    # Crear logger
    logger = logging.getLogger('ejercicio1')
    logger.setLevel(logging.INFO)

    # Limpiar handlers existentes
    logger.handlers.clear()

    # Formato
    log_format = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # Handler para archivo
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    logger.info(f"Log guardado en: {log_file}")

    return logger, log_file


# Configurar logging
logger, log_file = setup_logging()


def log_print(mensaje: str):
    """Escribe mensaje a consola y archivo de log"""
    print(mensaje)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(mensaje + '\n')


def print_banner():
    """Imprime banner inicial"""
    log_print("")
    log_print("=" * 80)
    log_print("  EJERCICIO 1: COMPARACION LLM vs MODELO CLASICO")
    log_print("  Deteccion de Anomalias en Series Temporales de Precios")
    log_print("=" * 80)
    log_print("")


def print_step(step_num, total_steps, description):
    """Imprime informacion del paso actual"""
    log_print("")
    log_print("-" * 80)
    log_print(f"  PASO {step_num}/{total_steps}: {description}")
    log_print("-" * 80)
    log_print("")


def ejecutar_pipeline_completo(config_path=None, skip_preprocessing=False,
                                skip_models=False, n_productos=3):
    """
    Ejecuta el pipeline completo de comparación de modelos.

    Args:
        config_path: Ruta al archivo de configuración (None usa default)
        skip_preprocessing: Si True, omite el preprocesamiento
        skip_models: Si True, omite el entrenamiento de modelos (usa predicciones existentes)
        n_productos: Número de productos para visualizaciones
    """
    print_banner()

    total_steps = 6
    current_step = 0

    try:
        # PASO 1: Preprocesamiento
        if not skip_preprocessing:
            current_step += 1
            print_step(current_step, total_steps, "PREPROCESAMIENTO DE DATOS")
            logger.info("Ejecutando preprocesamiento...")
            ejecutar_preprocesamiento(config_path)
            logger.info("✓ Preprocesamiento completado")
        else:
            logger.info("⊘ Preprocesamiento omitido (skip_preprocessing=True)")

        # PASO 2: Modelo A (LLM)
        if not skip_models:
            current_step += 1
            print_step(current_step, total_steps, "MODELO A - LLM (GROQ)")
            logger.info("Ejecutando Modelo A (LLM)...")
            ejecutar_modelo_a(config_path)
            logger.info("✓ Modelo A completado")
        else:
            logger.info("⊘ Modelo A omitido (skip_models=True)")

        # PASO 3: Modelo B (Isolation Forest)
        if not skip_models:
            current_step += 1
            print_step(current_step, total_steps, "MODELO B - ISOLATION FOREST")
            logger.info("Ejecutando Modelo B (Isolation Forest)...")
            ejecutar_modelo_b(config_path)
            logger.info("✓ Modelo B completado")
        else:
            logger.info("⊘ Modelo B omitido (skip_models=True)")

        # PASO 4: Evaluación
        current_step += 1
        print_step(current_step, total_steps, "EVALUACIÓN COMPARATIVA")
        logger.info("Ejecutando evaluación...")
        ejecutar_evaluacion(config_path)
        logger.info("✓ Evaluación completada")

        # PASO 5: A/B Testing
        current_step += 1
        print_step(current_step, total_steps, "A/B TESTING ESTADÍSTICO")
        logger.info("Ejecutando A/B test...")
        ejecutar_ab_test(config_path)
        logger.info("✓ A/B Testing completado")

        # PASO 6: Visualizaciones
        current_step += 1
        print_step(current_step, total_steps, "VISUALIZACIONES DE SERIES TEMPORALES")
        logger.info("Generando visualizaciones...")
        crear_visualizaciones_series_temporales(config_path, n_productos=n_productos)
        logger.info("✓ Visualizaciones completadas")

        # Resumen final
        log_print("")
        log_print("=" * 80)
        log_print("  PIPELINE COMPLETADO EXITOSAMENTE")
        log_print("=" * 80)
        log_print("")
        log_print("RESULTADOS GENERADOS:")
        log_print("  - Datos procesados:         data/processed/")
        log_print("  - Predicciones:             outputs/results/")
        log_print("  - Metricas:                 outputs/results/")
        log_print("  - Graficos:                 outputs/plots/")
        log_print("  - Modelos:                  outputs/models/")
        log_print("")
        log_print("ARCHIVOS CLAVE:")
        log_print("  - comparacion_modelos.csv")
        log_print("  - ab_test_results.json")
        log_print("  - confusion_matrices.png")
        log_print("  - precision_recall_curves.png")
        log_print("  - series_temporales_comparacion_modelos.png")
        log_print("")
        log_print(f"LOG GUARDADO EN: {log_file}")
        log_print("=" * 80)

        return True

    except Exception as e:
        logger.error(f"Error en el pipeline: {e}", exc_info=True)
        log_print("")
        log_print("=" * 80)
        log_print("  ERROR: EL PIPELINE FALLO")
        log_print("=" * 80)
        log_print(f"{str(e)}")
        return False


def main():
    """Función principal con argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Ejercicio 1: Comparación LLM vs Modelo Clásico para Detección de Anomalías",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Ejecutar pipeline completo
  python main.py

  # Ejecutar solo evaluación (omitir preprocesamiento y modelos)
  python main.py --skip-preprocessing --skip-models

  # Ejecutar con configuración personalizada
  python main.py --config config/config_custom.yaml

  # Generar más visualizaciones de productos
  python main.py --n-productos 5 --skip-preprocessing --skip-models
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Ruta al archivo de configuración YAML (default: config/config.yaml)'
    )

    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Omitir paso de preprocesamiento (usar datos procesados existentes)'
    )

    parser.add_argument(
        '--skip-models',
        action='store_true',
        help='Omitir entrenamiento de modelos (usar predicciones existentes)'
    )

    parser.add_argument(
        '--n-productos',
        type=int,
        default=3,
        help='Número de productos para visualizaciones (default: 3)'
    )

    args = parser.parse_args()

    # Ejecutar pipeline
    success = ejecutar_pipeline_completo(
        config_path=args.config,
        skip_preprocessing=args.skip_preprocessing,
        skip_models=args.skip_models,
        n_productos=args.n_productos
    )

    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
