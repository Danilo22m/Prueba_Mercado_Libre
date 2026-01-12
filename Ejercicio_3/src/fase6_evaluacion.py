"""
FASE 6: Evaluacion - Pipeline Completo RAG + Agente Critico
Autor: Danilo Melo
Fecha: 2026-01-12

Funcionalidades:
- Ejecutar pipeline completo con queries de prueba
- Generar ejemplos completos (Query -> Chunks -> Respuesta -> Veredicto)
- Calcular metricas consolidadas
- Guardar logs JSON del agente critico
- Generar reporte final de evaluacion
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Fase6Evaluacion:
    """Clase para evaluacion del pipeline RAG completo"""

    # Queries de prueba para evaluacion
    TEST_QUERIES = [
        "Que procesador tiene el HP 15?",
        "Que laptops tienen tarjeta grafica NVIDIA?",
        "Cual es el tamano de pantalla del Lenovo ThinkPad?",
        "Que laptop tiene mejor rendimiento de CPU?",
        "Cuales laptops tienen WiFi 6?",
        "Que GPU tiene el ASUS ROG?",
        "Laptops con pantalla de 15.6 pulgadas",
        "Que laptops tienen puerto Ethernet?",
        "Cual es el peso del Dell XPS?",
        "Que resolucion de pantalla tiene el Lenovo IdeaPad?",
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializar evaluador

        Args:
            config: Diccionario con configuracion del proyecto
        """
        self.config = config
        self.eval_config = config.get('evaluation', {})

        # Paths de salida
        self.results_dir = Path('outputs/results')
        self.logs_dir = Path('outputs/logs')

        # Crear directorios
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Resultados
        self.resultados = []
        self.metricas = {}

        # Componentes (se inicializan en ejecutar())
        self.retrieval = None
        self.generation = None
        self.agente_critico = None

    def ejecutar(self, num_queries: int = 10) -> Dict[str, Any]:
        """
        Ejecutar evaluacion completa del pipeline

        Args:
            num_queries: Numero de queries a evaluar (max 10)

        Returns:
            Diccionario con resultados y metricas
        """
        logger.info("="*60)
        logger.info("FASE 6: Evaluacion del Pipeline RAG")
        logger.info("="*60)

        # Inicializar componentes
        self._inicializar_componentes()

        # Seleccionar queries
        queries = self.TEST_QUERIES[:min(num_queries, len(self.TEST_QUERIES))]
        logger.info(f"Evaluando {len(queries)} queries")

        # Ejecutar pipeline para cada query
        tiempos_totales = []
        tiempos_retrieval = []
        tiempos_generation = []
        tiempos_verificacion = []

        for i, query in enumerate(queries, 1):
            logger.info(f"\n--- Query {i}/{len(queries)} ---")
            logger.info(f"Query: {query}")

            inicio_total = time.time()

            # FASE 3: Retrieval
            inicio_retrieval = time.time()
            resultado_retrieval = self.retrieval.buscar(query)
            tiempo_retrieval = (time.time() - inicio_retrieval) * 1000
            tiempos_retrieval.append(tiempo_retrieval)

            chunks = resultado_retrieval['retrieved_chunks']
            logger.info(f"Chunks recuperados: {len(chunks)}")

            # FASE 4 + FASE 5: Generation + Agente Critico
            inicio_gen = time.time()
            resultado_pipeline = self.agente_critico.ejecutar_pipeline_con_critica(
                query=query,
                chunks=chunks,
                generador=self.generation
            )
            tiempo_gen_critica = (time.time() - inicio_gen) * 1000

            tiempo_total = (time.time() - inicio_total) * 1000
            tiempos_totales.append(tiempo_total)

            # Extraer tiempos individuales
            if resultado_pipeline.get('historial'):
                ultimo = resultado_pipeline['historial'][-1]
                t_verif = ultimo.get('verificacion', {}).get('verification_time_ms', 0)
                tiempos_verificacion.append(t_verif)
                tiempos_generation.append(tiempo_gen_critica - t_verif)
            else:
                tiempos_generation.append(tiempo_gen_critica)
                tiempos_verificacion.append(0)

            # Guardar resultado completo
            resultado_completo = {
                'query_id': i,
                'query': query,
                'chunks': chunks,
                'respuesta': resultado_pipeline.get('response', ''),
                'citations_used': resultado_pipeline.get('citations_used', []),
                'aprobada': resultado_pipeline.get('aprobada', False),
                'intentos': resultado_pipeline.get('intentos', 0),
                'verificacion': resultado_pipeline.get('verificacion', {}),
                'advertencia': resultado_pipeline.get('advertencia', ''),
                'tiempos': {
                    'total_ms': round(tiempo_total, 2),
                    'retrieval_ms': round(tiempo_retrieval, 2),
                    'generation_ms': round(tiempos_generation[-1], 2),
                    'verificacion_ms': round(tiempos_verificacion[-1], 2)
                }
            }

            self.resultados.append(resultado_completo)

            # Log del resultado
            estado = "APROBADA" if resultado_completo['aprobada'] else "RECHAZADA"
            logger.info(f"Respuesta: {resultado_completo['respuesta'][:80]}...")
            logger.info(f"Estado: {estado} | Intentos: {resultado_completo['intentos']} | Tiempo: {tiempo_total:.0f}ms")

        # Calcular metricas
        self._calcular_metricas(tiempos_totales, tiempos_retrieval, tiempos_generation, tiempos_verificacion)

        # Guardar resultados
        self._guardar_resultados()

        logger.info("\n" + "="*60)
        logger.info("Evaluacion completada")
        logger.info("="*60)

        return {
            'resultados': self.resultados,
            'metricas': self.metricas
        }

    def _inicializar_componentes(self):
        """Inicializar componentes del pipeline"""
        from fase3_retrieval import Fase3Retrieval
        from fase4_generation import Fase4Generation
        from fase5_agente_critico import Fase5AgenteCritico

        logger.info("Inicializando componentes...")

        # FASE 3: Retrieval
        self.retrieval = Fase3Retrieval(self.config)
        self.retrieval.cargar_index()

        # FASE 4: Generation
        self.generation = Fase4Generation(self.config)

        # FASE 5: Agente Critico
        self.agente_critico = Fase5AgenteCritico(self.config)

        logger.info("Componentes inicializados")

    def _calcular_metricas(
        self,
        tiempos_totales: List[float],
        tiempos_retrieval: List[float],
        tiempos_generation: List[float],
        tiempos_verificacion: List[float]
    ):
        """Calcular metricas consolidadas"""
        import numpy as np

        total_queries = len(self.resultados)
        aprobadas = sum(1 for r in self.resultados if r['aprobada'])
        rechazadas = total_queries - aprobadas
        total_intentos = sum(r['intentos'] for r in self.resultados)

        self.metricas = {
            'resumen': {
                'total_queries': total_queries,
                'aprobadas': aprobadas,
                'rechazadas': rechazadas,
                'tasa_aprobacion': round(aprobadas / total_queries * 100, 2) if total_queries > 0 else 0,
                'total_intentos': total_intentos,
                'promedio_intentos': round(total_intentos / total_queries, 2) if total_queries > 0 else 0
            },
            'tiempos': {
                'total': {
                    'promedio_ms': round(np.mean(tiempos_totales), 2),
                    'min_ms': round(np.min(tiempos_totales), 2),
                    'max_ms': round(np.max(tiempos_totales), 2),
                    'std_ms': round(np.std(tiempos_totales), 2)
                },
                'retrieval': {
                    'promedio_ms': round(np.mean(tiempos_retrieval), 2),
                    'min_ms': round(np.min(tiempos_retrieval), 2),
                    'max_ms': round(np.max(tiempos_retrieval), 2)
                },
                'generation': {
                    'promedio_ms': round(np.mean(tiempos_generation), 2),
                    'min_ms': round(np.min(tiempos_generation), 2),
                    'max_ms': round(np.max(tiempos_generation), 2)
                },
                'verificacion': {
                    'promedio_ms': round(np.mean(tiempos_verificacion), 2),
                    'min_ms': round(np.min(tiempos_verificacion), 2),
                    'max_ms': round(np.max(tiempos_verificacion), 2)
                }
            },
            'agente_critico': self.agente_critico.obtener_estadisticas(),
            'retrieval': self.retrieval.obtener_estadisticas(),
            'generation': self.generation.obtener_estadisticas()
        }

    def _guardar_resultados(self):
        """Guardar todos los resultados y reportes"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 1. Ejemplos completos (JSON)
        ejemplos_path = self.results_dir / f'ejemplos_completos_{timestamp}.json'
        with open(ejemplos_path, 'w', encoding='utf-8') as f:
            json.dump(self.resultados, f, indent=2, ensure_ascii=False)
        logger.info(f"Ejemplos guardados: {ejemplos_path}")

        # 2. Metricas consolidadas (JSON)
        metricas_path = self.results_dir / f'metricas_{timestamp}.json'
        with open(metricas_path, 'w', encoding='utf-8') as f:
            json.dump(self.metricas, f, indent=2, ensure_ascii=False)
        logger.info(f"Metricas guardadas: {metricas_path}")

        # 3. Logs del agente critico (JSON)
        logs_critico = []
        for r in self.resultados:
            logs_critico.append({
                'query_id': r['query_id'],
                'query': r['query'],
                'aprobada': r['aprobada'],
                'intentos': r['intentos'],
                'verificacion': r['verificacion'],
                'advertencia': r.get('advertencia', '')
            })

        logs_path = self.logs_dir / f'logs_agente_critico_{timestamp}.json'
        with open(logs_path, 'w', encoding='utf-8') as f:
            json.dump(logs_critico, f, indent=2, ensure_ascii=False)
        logger.info(f"Logs guardados: {logs_path}")

        # 4. Reporte legible (TXT)
        reporte_path = self.results_dir / f'reporte_evaluacion_{timestamp}.txt'
        self._generar_reporte_txt(reporte_path)
        logger.info(f"Reporte guardado: {reporte_path}")

    def _generar_reporte_txt(self, path: Path):
        """Generar reporte en formato legible"""
        lines = []
        lines.append("="*70)
        lines.append("REPORTE DE EVALUACION - SISTEMA RAG + AGENTE CRITICO")
        lines.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("="*70)

        # Metricas resumen
        lines.append("\n## METRICAS CONSOLIDADAS\n")
        resumen = self.metricas['resumen']
        lines.append(f"Total queries evaluadas: {resumen['total_queries']}")
        lines.append(f"Respuestas aprobadas: {resumen['aprobadas']}")
        lines.append(f"Respuestas rechazadas: {resumen['rechazadas']}")
        lines.append(f"Tasa de aprobacion: {resumen['tasa_aprobacion']}%")
        lines.append(f"Total intentos: {resumen['total_intentos']}")
        lines.append(f"Promedio intentos por query: {resumen['promedio_intentos']}")

        # Tiempos
        lines.append("\n## TIEMPOS DE RESPUESTA\n")
        tiempos = self.metricas['tiempos']
        lines.append(f"Tiempo total promedio: {tiempos['total']['promedio_ms']} ms")
        lines.append(f"  - Retrieval: {tiempos['retrieval']['promedio_ms']} ms")
        lines.append(f"  - Generation: {tiempos['generation']['promedio_ms']} ms")
        lines.append(f"  - Verificacion: {tiempos['verificacion']['promedio_ms']} ms")

        # Ejemplos
        lines.append("\n" + "="*70)
        lines.append("EJEMPLOS COMPLETOS (Query -> Chunks -> Respuesta)")
        lines.append("="*70)

        for r in self.resultados:
            lines.append(f"\n--- Query {r['query_id']} ---")
            lines.append(f"Q: {r['query']}")
            lines.append(f"\nChunks recuperados ({len(r['chunks'])}):")
            for i, chunk in enumerate(r['chunks'][:3], 1):  # Solo primeros 3
                lines.append(f"  {i}. {chunk['text'][:70]}... {chunk['citation']}")
            if len(r['chunks']) > 3:
                lines.append(f"  ... y {len(r['chunks'])-3} chunks mas")

            lines.append(f"\nRespuesta: {r['respuesta']}")
            lines.append(f"Citas usadas: {', '.join(r['citations_used'])}")

            estado = "APROBADA" if r['aprobada'] else "RECHAZADA"
            lines.append(f"\nEstado: {estado}")
            lines.append(f"Intentos: {r['intentos']}")

            if r.get('verificacion', {}).get('problemas'):
                lines.append("Problemas detectados:")
                for p in r['verificacion']['problemas']:
                    lines.append(f"  - [{p.get('tipo')}] {p.get('descripcion', '')[:50]}")

            if r.get('advertencia'):
                lines.append(f"ADVERTENCIA: {r['advertencia']}")

            lines.append(f"\nTiempos: Total={r['tiempos']['total_ms']}ms | Retrieval={r['tiempos']['retrieval_ms']}ms | Gen={r['tiempos']['generation_ms']}ms")

        # Analisis
        lines.append("\n" + "="*70)
        lines.append("ANALISIS DE CASOS")
        lines.append("="*70)

        aprobadas = [r for r in self.resultados if r['aprobada']]
        rechazadas = [r for r in self.resultados if not r['aprobada']]

        lines.append(f"\nCasos APROBADOS ({len(aprobadas)}):")
        for r in aprobadas:
            lines.append(f"  - Query {r['query_id']}: {r['query'][:40]}...")

        lines.append(f"\nCasos RECHAZADOS ({len(rechazadas)}):")
        for r in rechazadas:
            lines.append(f"  - Query {r['query_id']}: {r['query'][:40]}...")
            if r.get('advertencia'):
                lines.append(f"    Razon: {r['advertencia'][:60]}")

        # Guardar
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


def main():
    """Funcion principal para ejecutar FASE 6"""
    import yaml

    # Cargar configuracion
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Verificar API key
    if not os.environ.get('GROQ_API_KEY'):
        print("\nError: GROQ_API_KEY no configurada")
        print("Configura en archivo .env")
        return

    # Ejecutar evaluacion
    evaluador = Fase6Evaluacion(config)

    # Por defecto evalua 10 queries (configurable)
    num_queries = config.get('evaluation', {}).get('num_test_queries', 10)
    resultado = evaluador.ejecutar(num_queries=num_queries)

    # Mostrar resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)

    metricas = resultado['metricas']
    resumen = metricas['resumen']

    print(f"\nQueries evaluadas: {resumen['total_queries']}")
    print(f"Tasa de aprobacion: {resumen['tasa_aprobacion']}%")
    print(f"Tiempo promedio: {metricas['tiempos']['total']['promedio_ms']} ms")

    print(f"\nArchivos generados en outputs/results/ y outputs/logs/")


if __name__ == "__main__":
    main()
