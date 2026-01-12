#!/usr/bin/env python3
"""
MAIN - Pipeline Completo de Análisis de Grafos de URLs
=======================================================

Ejecuta todas las fases del proyecto en secuencia:
- FASE 1: Selección de subgrafo
- FASE 2: Análisis estadístico del grafo
- FASE 3: Cálculo de métricas de centralidad
- FASE 4: Ranking y análisis de roles Top-20
- FASE 5: Visualización del grafo
- FASE 6: Simulación de propagación
- FASE 7: Generación de recomendaciones accionables

Autor: Danilo Melo
Fecha: 2026-01-12
"""

import sys
import logging
import yaml
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime
import subprocess


# Directorios
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / 'outputs' / 'logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURACION
# =============================================================================
def cargar_configuracion() -> Dict:
    """Carga la configuracion desde config.yaml"""
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict) -> Tuple[logging.Logger, Path]:
    """Configura logging dual: consola + archivo"""
    # Crear nombre de archivo con timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOGS_DIR / f'pipeline_completo_{timestamp}.log'

    # Crear logger
    logger = logging.getLogger('ejercicio2')
    log_config = config.get('logging', {})
    logger.setLevel(getattr(logging, log_config.get('level', 'INFO')))

    # Limpiar handlers existentes
    logger.handlers.clear()

    # Formato
    log_format = logging.Formatter(
        log_config.get('format', '[%(asctime)s] %(levelname)s - %(message)s'),
        datefmt=log_config.get('date_format', '%Y-%m-%d %H:%M:%S')
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


# =============================================================================
# EJECUCIÓN DE FASES
# =============================================================================
def ejecutar_fase(fase_num: int, script_path: str, logger: logging.Logger) -> bool:
    """
    Ejecuta un script de fase usando subprocess.

    Args:
        fase_num: Número de fase
        script_path: Ruta al script de la fase
        logger: Logger para mensajes

    Returns:
        bool: True si se ejecutó correctamente, False en caso contrario
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"EJECUTANDO FASE {fase_num}: {Path(script_path).stem}")
    logger.info("=" * 80)

    try:
        # Ejecutar el script usando el intérprete de Python actual
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,  # Mostrar output en tiempo real
            text=True
        )

        logger.info(f"✓ FASE {fase_num} completada exitosamente")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Error en FASE {fase_num}: {e}")
        logger.error(f"  Código de salida: {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"✗ Error inesperado en FASE {fase_num}: {e}")
        return False


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Funcion principal que ejecuta todo el pipeline"""

    # Cargar configuracion y setup logging
    config = cargar_configuracion()
    logger, log_file = setup_logging(config)

    logger.info("")
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETO - ANÁLISIS DE GRAFOS DE URLs")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Este script ejecutará las siguientes fases:")
    logger.info("  1. Selección de subgrafo")
    logger.info("  2. Análisis estadístico del grafo")
    logger.info("  3. Cálculo de métricas de centralidad")
    logger.info("  4. Ranking y análisis de roles Top-20")
    logger.info("  5. Visualización del grafo")
    logger.info("  6. Simulación de propagación")
    logger.info("  7. Generación de recomendaciones accionables")
    logger.info("")

    # Directorio de scripts
    src_dir = Path(__file__).parent / "src"

    # Definir fases a ejecutar
    fases = [
        (1, src_dir / "fase1_seleccion_subgrafo.py"),
        (2, src_dir / "fase2_analisis_grafo.py"),
        (3, src_dir / "fase3_metricas_centralidad.py"),
        (4, src_dir / "fase4_ranking_top20.py"),
        (5, src_dir / "fase5_visualizacion.py"),
        (6, src_dir / "fase6_propagacion.py"),
        (7, src_dir / "fase7_recomendaciones_ecommerce.py")
    ]

    # Ejecutar cada fase secuencialmente
    fases_exitosas = 0
    for fase_num, script_path in fases:
        if not script_path.exists():
            logger.error(f"✗ No se encontró el script: {script_path}")
            logger.error(f"  Saltando FASE {fase_num}")
            continue

        # Ejecutar fase
        exito = ejecutar_fase(fase_num, str(script_path), logger)

        if exito:
            fases_exitosas += 1
        else:
            logger.error(f"✗ La FASE {fase_num} falló. Abortando pipeline.")
            logger.error("")
            logger.error("RECOMENDACIÓN:")
            logger.error(f"  Revisa el error de FASE {fase_num} y corrige el problema.")
            logger.error(f"  Luego puedes ejecutar manualmente:")
            logger.error(f"    python {script_path}")
            logger.error("")
            sys.exit(1)

    # Resumen final
    logger.info("")
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETADO")
    logger.info("=" * 80)
    logger.info(f"✓ {fases_exitosas}/{len(fases)} fases ejecutadas exitosamente")
    logger.info("")
    logger.info("ARCHIVOS GENERADOS:")
    logger.info(f"  - Datos procesados: {config['outputs']['results_dir']}")
    logger.info(f"  - Visualizaciones: {config['outputs']['visualizaciones_dir']}")
    logger.info(f"  - Propagación: {config['outputs']['propagacion_dir']}")
    logger.info(f"  - Recomendaciones: {config['outputs']['recomendaciones_dir']}")
    logger.info("")
    logger.info("PROXIMOS PASOS:")
    logger.info("  1. Revisar visualizaciones en outputs/visualizaciones/")
    logger.info("  2. Abrir grafo interactivo: outputs/visualizaciones/fase5_grafo_interactivo.html")
    logger.info("  3. Leer recomendaciones: outputs/recomendaciones/fase7_recomendaciones.md")
    logger.info("")
    logger.info(f"LOG GUARDADO EN: {log_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
