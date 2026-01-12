"""
Main - Pipeline Completo RAG + Agente Critico
Autor: Danilo Melo
Fecha: 2026-01-12

Ejecuta el flujo completo:
FASE 1 -> FASE 2 -> FASE 3 -> FASE 4 -> FASE 5 -> FASE 6

Los logs se guardan automaticamente en outputs/logs/
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar paths
BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))

# Crear directorio de logs si no existe
LOGS_DIR = BASE_DIR / 'outputs' / 'logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configurar logging dual: consola + archivo (sobrescribe el log anterior)
log_file = LOGS_DIR / 'pipeline_completo.log'

# Crear handlers
console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(log_file, encoding='utf-8')

# Formato de logs
log_format = logging.Formatter(
    '[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(log_format)
file_handler.setFormatter(log_format)

# Configurar logger raiz
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)


def verificar_requisitos():
    """Verificar que todos los requisitos esten cumplidos"""
    logger.info("Verificando requisitos...")

    # Verificar API key
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        logger.error("GROQ_API_KEY no configurada en archivo .env")
        return False

    api_key = api_key.strip()
    if not api_key.startswith('gsk_') or len(api_key) < 20:
        logger.error("GROQ_API_KEY parece invalida")
        return False

    logger.info("API Key configurada correctamente")

    # Verificar archivos necesarios
    archivos_requeridos = [
        BASE_DIR / 'config' / 'config.yaml',
        BASE_DIR / 'data' / 'raw' / 'Laptops_with_technical_specifications.csv',
    ]

    for archivo in archivos_requeridos:
        if not archivo.exists():
            logger.error(f"Archivo no encontrado: {archivo}")
            return False

    logger.info("Archivos requeridos encontrados")
    return True


def ejecutar_fase1():
    """Ejecutar FASE 1: Ingesta y Normalizacion"""
    logger.info("="*60)
    logger.info("FASE 1: Ingesta y Normalizacion")
    logger.info("="*60)

    from fase1_ingesta import Fase1Ingesta
    import yaml

    config_path = BASE_DIR / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    fase1 = Fase1Ingesta(config)
    df = fase1.ejecutar()

    logger.info(f"FASE 1 completada: {len(df)} laptops normalizados")
    return True


def ejecutar_fase2():
    """Ejecutar FASE 2: Chunking e Indexacion"""
    logger.info("="*60)
    logger.info("FASE 2: Chunking e Indexacion con Embeddings")
    logger.info("="*60)

    from fase2_chunking import Fase2Chunking
    import yaml

    config_path = BASE_DIR / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    fase2 = Fase2Chunking(config)
    chunks, embeddings = fase2.ejecutar()

    logger.info(f"FASE 2 completada: {len(chunks)} chunks, embeddings shape {embeddings.shape}")
    return True


def ejecutar_fase3_4_5_6(num_queries: int = 10):
    """Ejecutar FASES 3-6: Retrieval, Generation, Agente Critico, Evaluacion"""
    logger.info("="*60)
    logger.info("FASES 3-6: Pipeline RAG + Agente Critico + Evaluacion")
    logger.info("="*60)

    from fase6_evaluacion import Fase6Evaluacion
    import yaml

    config_path = BASE_DIR / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    evaluador = Fase6Evaluacion(config)
    resultado = evaluador.ejecutar(num_queries=num_queries)

    return resultado


def main():
    """Funcion principal - ejecuta todo el pipeline"""
    logger.info("="*70)
    logger.info("PIPELINE COMPLETO RAG + AGENTE CRITICO")
    logger.info(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log guardado en: {log_file}")
    logger.info("="*70)

    # Verificar requisitos
    if not verificar_requisitos():
        logger.error("Requisitos no cumplidos. Abortando.")
        return

    try:
        # FASE 1: Ingesta y Normalizacion
        ejecutar_fase1()

        # FASE 2: Chunking e Indexacion
        ejecutar_fase2()

        # FASES 3-6: Retrieval + Generation + Agente Critico + Evaluacion
        resultado = ejecutar_fase3_4_5_6(num_queries=10)

        # Resumen final
        logger.info("\n" + "="*70)
        logger.info("RESUMEN FINAL DEL PIPELINE")
        logger.info("="*70)

        metricas = resultado['metricas']
        resumen = metricas['resumen']

        logger.info(f"Total queries evaluadas: {resumen['total_queries']}")
        logger.info(f"Respuestas aprobadas: {resumen['aprobadas']}")
        logger.info(f"Respuestas rechazadas: {resumen['rechazadas']}")
        logger.info(f"Tasa de aprobacion: {resumen['tasa_aprobacion']}%")
        logger.info(f"Tiempo promedio: {metricas['tiempos']['total']['promedio_ms']} ms")

        # Metricas avanzadas de RAG
        logger.info("\n" + "-"*70)
        logger.info("METRICAS AVANZADAS DE RAG")
        logger.info("-"*70)

        avanzadas = metricas.get('metricas_avanzadas', {})
        precision = avanzadas.get('precision', {}).get('promedio', 0)
        recall = avanzadas.get('recall', {}).get('promedio', 0)
        f1 = avanzadas.get('f1_score', 0)
        faithfulness = avanzadas.get('faithfulness', {}).get('promedio', 0)
        coverage = avanzadas.get('answer_coverage', {}).get('promedio', 0)

        logger.info(f"Precision (Retrieval):     {precision:.4f}")
        logger.info(f"Recall (Retrieval):        {recall:.4f}")
        logger.info(f"F1-Score:                  {f1:.4f}")
        logger.info(f"Faithfulness:              {faithfulness:.4f}")
        logger.info(f"Answer Coverage:           {coverage:.4f}")

        logger.info(f"\nResultados guardados en: outputs/results/")
        logger.info(f"Logs guardados en: {log_file}")

        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"Error en pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
