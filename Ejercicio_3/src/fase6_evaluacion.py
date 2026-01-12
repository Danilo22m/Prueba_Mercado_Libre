"""
FASE 6: Evaluacion - Pipeline Completo RAG + Agente Critico
Autor: Danilo Melo
Fecha: 2026-01-12

Funcionalidades:
- Ejecutar pipeline completo con queries de prueba
- Generar ejemplos completos (Query -> Chunks -> Respuesta -> Veredicto)
- Calcular metricas consolidadas
- Calcular metricas avanzadas: Precision, Recall, Faithfulness, Answer Coverage
- Guardar logs JSON del agente critico
- Generar reporte final de evaluacion
"""

import os
import re
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

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

    # Queries de prueba con ground truth para calcular Precision y Recall
    # Cada query tiene: texto, campos relevantes esperados, expected_info para validar
    # NOTA: Los campos deben coincidir EXACTAMENTE con los valores en chunks.json:
    # cpu, gpu, display_size, display_resolution, weight, ethernet, wifi_version,
    # bluetooth_version, hdmi_version, ram, disc
    # expected_info debe contener terminos que aparecen en el TEXTO de los chunks
    TEST_QUERIES_WITH_GROUND_TRUTH = [
        {
            "query": "Que procesador tiene el HP 15?",
            "relevant_fields": ["cpu"],
            "expected_info": ["HP 15", "procesador", "Intel"]
        },
        {
            "query": "Que laptops tienen tarjeta grafica NVIDIA?",
            "relevant_fields": ["gpu"],
            "expected_info": ["NVIDIA", "GeForce", "RTX"]
        },
        {
            "query": "Cual es el tamano de pantalla del Lenovo ThinkPad?",
            "relevant_fields": ["display_size"],
            "expected_info": ["ThinkPad", "pantalla", "inches"]
        },
        {
            "query": "Que laptop tiene mejor rendimiento de CPU?",
            "relevant_fields": ["cpu"],
            "expected_info": ["CPU Mark", "rendimiento", "procesador"]
        },
        {
            "query": "Cuales laptops tienen WiFi 6?",
            "relevant_fields": ["wifi_version"],
            "expected_info": ["WiFi", "802.11ax", "802.11ac"]
        },
        {
            "query": "Que GPU tiene el ASUS ROG?",
            "relevant_fields": ["gpu"],
            "expected_info": ["ASUS", "ROG", "grafica"]
        },
        {
            "query": "Laptops con pantalla de 15.6 pulgadas",
            "relevant_fields": ["display_size"],
            "expected_info": ["15.6", "pantalla", "inches"]
        },
        {
            "query": "Que laptops tienen puerto Ethernet?",
            "relevant_fields": ["ethernet"],
            "expected_info": ["Ethernet", "Mbit"]
        },
        {
            "query": "Cual es el peso del Dell XPS?",
            "relevant_fields": ["weight"],
            "expected_info": ["Dell XPS", "peso", "mm"]
        },
        {
            "query": "Que resolucion de pantalla tiene el Lenovo IdeaPad?",
            "relevant_fields": ["display_resolution"],
            "expected_info": ["IdeaPad", "resolucion", "1920"]
        },
    ]

    # Lista simple de queries (compatibilidad)
    TEST_QUERIES = [q["query"] for q in TEST_QUERIES_WITH_GROUND_TRUTH]

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

        # Cliente LLM para calcular Faithfulness
        self.groq_client = None
        self._init_groq_client()

    def _init_groq_client(self):
        """Inicializar cliente Groq para metricas avanzadas"""
        api_key = os.environ.get('GROQ_API_KEY')
        if api_key:
            self.groq_client = Groq(api_key=api_key.strip())

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

        # Seleccionar queries con ground truth
        queries_gt = self.TEST_QUERIES_WITH_GROUND_TRUTH[:min(num_queries, len(self.TEST_QUERIES_WITH_GROUND_TRUTH))]
        logger.info(f"Evaluando {len(queries_gt)} queries")

        # Ejecutar pipeline para cada query
        tiempos_totales = []
        tiempos_retrieval = []
        tiempos_generation = []
        tiempos_verificacion = []

        # Metricas avanzadas
        precision_scores = []
        recall_scores = []
        faithfulness_scores = []
        answer_coverage_scores = []

        for i, query_info in enumerate(queries_gt, 1):
            query = query_info["query"]
            ground_truth = query_info
            logger.info(f"\n--- Query {i}/{len(queries_gt)} ---")
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

            # Calcular metricas avanzadas para esta query
            respuesta = resultado_pipeline.get('response', '')

            # Precision y Recall del Retrieval
            precision, recall = self._calcular_precision_recall(
                chunks, ground_truth
            )
            precision_scores.append(precision)
            recall_scores.append(recall)

            # Faithfulness (fidelidad al contexto)
            faithfulness = self._calcular_faithfulness(respuesta, chunks)
            faithfulness_scores.append(faithfulness)

            # Answer Coverage (cobertura de la respuesta)
            coverage = self._calcular_answer_coverage(respuesta, chunks, ground_truth)
            answer_coverage_scores.append(coverage)

            # Guardar resultado completo
            resultado_completo = {
                'query_id': i,
                'query': query,
                'chunks': chunks,
                'respuesta': respuesta,
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
                },
                'metricas_avanzadas': {
                    'precision': round(precision, 4),
                    'recall': round(recall, 4),
                    'faithfulness': round(faithfulness, 4),
                    'answer_coverage': round(coverage, 4)
                }
            }

            self.resultados.append(resultado_completo)

            # Log del resultado
            estado = "APROBADA" if resultado_completo['aprobada'] else "RECHAZADA"
            logger.info(f"Respuesta: {resultado_completo['respuesta'][:80]}...")
            logger.info(f"Estado: {estado} | Intentos: {resultado_completo['intentos']} | Tiempo: {tiempo_total:.0f}ms")
            logger.info(f"Metricas: Precision={precision:.2f} | Recall={recall:.2f} | Faithfulness={faithfulness:.2f} | Coverage={coverage:.2f}")

        # Calcular metricas
        self._calcular_metricas(
            tiempos_totales, tiempos_retrieval, tiempos_generation, tiempos_verificacion,
            precision_scores, recall_scores, faithfulness_scores, answer_coverage_scores
        )

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
        tiempos_verificacion: List[float],
        precision_scores: List[float],
        recall_scores: List[float],
        faithfulness_scores: List[float],
        answer_coverage_scores: List[float]
    ):
        """Calcular metricas consolidadas incluyendo metricas avanzadas de RAG"""
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
            'metricas_avanzadas': {
                'precision': {
                    'promedio': round(np.mean(precision_scores), 4) if precision_scores else 0,
                    'min': round(np.min(precision_scores), 4) if precision_scores else 0,
                    'max': round(np.max(precision_scores), 4) if precision_scores else 0,
                    'std': round(np.std(precision_scores), 4) if precision_scores else 0
                },
                'recall': {
                    'promedio': round(np.mean(recall_scores), 4) if recall_scores else 0,
                    'min': round(np.min(recall_scores), 4) if recall_scores else 0,
                    'max': round(np.max(recall_scores), 4) if recall_scores else 0,
                    'std': round(np.std(recall_scores), 4) if recall_scores else 0
                },
                'faithfulness': {
                    'promedio': round(np.mean(faithfulness_scores), 4) if faithfulness_scores else 0,
                    'min': round(np.min(faithfulness_scores), 4) if faithfulness_scores else 0,
                    'max': round(np.max(faithfulness_scores), 4) if faithfulness_scores else 0,
                    'std': round(np.std(faithfulness_scores), 4) if faithfulness_scores else 0
                },
                'answer_coverage': {
                    'promedio': round(np.mean(answer_coverage_scores), 4) if answer_coverage_scores else 0,
                    'min': round(np.min(answer_coverage_scores), 4) if answer_coverage_scores else 0,
                    'max': round(np.max(answer_coverage_scores), 4) if answer_coverage_scores else 0,
                    'std': round(np.std(answer_coverage_scores), 4) if answer_coverage_scores else 0
                },
                'f1_score': round(
                    2 * (np.mean(precision_scores) * np.mean(recall_scores)) /
                    (np.mean(precision_scores) + np.mean(recall_scores)), 4
                ) if precision_scores and recall_scores and (np.mean(precision_scores) + np.mean(recall_scores)) > 0 else 0
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

    def _calcular_precision_recall(
        self,
        chunks_recuperados: List[Dict],
        ground_truth: Dict
    ) -> Tuple[float, float]:
        """
        Calcular Precision y Recall del Retrieval

        Precision = chunks relevantes recuperados / total chunks recuperados
        Recall = al menos un chunk relevante fue recuperado (binario)

        Un chunk es relevante si:
        1. El campo (field) coincide con los campos esperados, Y
        2. El texto contiene informacion relacionada con la query

        Args:
            chunks_recuperados: Lista de chunks devueltos por el retrieval
            ground_truth: Diccionario con campos relevantes y expected_info

        Returns:
            Tupla (precision, recall)
        """
        if not chunks_recuperados:
            return 0.0, 0.0

        relevant_fields = set(ground_truth.get('relevant_fields', []))
        expected_info = ground_truth.get('expected_info', [])

        # Contar chunks relevantes recuperados
        chunks_relevantes = 0
        for chunk in chunks_recuperados:
            field = chunk.get('field', '')
            text = chunk.get('text', '').lower()

            # Un chunk es relevante si:
            # 1. Es del campo relevante
            if field not in relevant_fields:
                continue

            # 2. El texto contiene alguna info esperada (si se especifica)
            if expected_info:
                tiene_info_esperada = any(
                    info.lower() in text for info in expected_info
                )
                if tiene_info_esperada:
                    chunks_relevantes += 1
            else:
                # Si no hay expected_info, solo verificar el campo
                chunks_relevantes += 1

        # Precision: de los recuperados, cuantos son relevantes
        precision = chunks_relevantes / len(chunks_recuperados) if chunks_recuperados else 0.0

        # Recall: proporcion de chunks relevantes recuperados vs top_k
        # Como no sabemos el total de chunks relevantes en todo el dataset,
        # usamos un recall simplificado: si recuperamos al menos 1 relevante = 1.0
        # Escalamos por la proporcion de chunks relevantes encontrados
        recall = min(1.0, chunks_relevantes / max(1, len(chunks_recuperados) // 2))

        return precision, recall

    def _calcular_faithfulness(
        self,
        respuesta: str,
        chunks: List[Dict]
    ) -> float:
        """
        Calcular Faithfulness: que tan fiel es la respuesta al contexto

        Usa el LLM para evaluar si las afirmaciones en la respuesta
        estan soportadas por los chunks.

        Args:
            respuesta: Respuesta generada
            chunks: Chunks usados como contexto

        Returns:
            Score de faithfulness (0.0 a 1.0)
        """
        if not respuesta or not chunks:
            return 0.0

        # Si no hay cliente LLM, usar heuristica basada en citas
        if not self.groq_client:
            return self._faithfulness_heuristica(respuesta, chunks)

        try:
            # Construir contexto
            context = "\n".join([
                f"- {chunk.get('text', '')} {chunk.get('citation', '')}"
                for chunk in chunks[:5]  # Limitar para no exceder tokens
            ])

            prompt = f"""Evalua la fidelidad de la siguiente respuesta respecto al contexto dado.

CONTEXTO:
{context}

RESPUESTA:
{respuesta}

Analiza cada afirmacion en la respuesta y determina si esta soportada por el contexto.
Responde SOLO con un numero decimal entre 0.0 y 1.0 donde:
- 1.0 = Todas las afirmaciones estan soportadas por el contexto
- 0.5 = Aproximadamente la mitad de las afirmaciones estan soportadas
- 0.0 = Ninguna afirmacion esta soportada (todo es alucinacion)

SCORE:"""

            response = self.groq_client.chat.completions.create(
                model=self.config.get('generation', {}).get('model_name', 'llama-3.1-8b-instant'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )

            # Extraer score
            score_text = response.choices[0].message.content.strip()
            # Buscar numero decimal en la respuesta
            match = re.search(r'(\d+\.?\d*)', score_text)
            if match:
                score = float(match.group(1))
                return min(1.0, max(0.0, score))  # Clamp entre 0 y 1

            return 0.5  # Default si no se puede parsear

        except Exception as e:
            logger.warning(f"Error calculando faithfulness con LLM: {e}")
            return self._faithfulness_heuristica(respuesta, chunks)

    def _faithfulness_heuristica(self, respuesta: str, chunks: List[Dict]) -> float:
        """Heuristica simple para faithfulness basada en citas"""
        # Contar citas en la respuesta
        citas_en_respuesta = re.findall(r'\[\d+:\w+\]', respuesta)
        citas_validas = set(chunk.get('citation', '') for chunk in chunks)

        if not citas_en_respuesta:
            return 0.3  # Penalizar si no hay citas

        # Proporcion de citas validas
        citas_correctas = sum(1 for c in citas_en_respuesta if c in citas_validas)
        return citas_correctas / len(citas_en_respuesta) if citas_en_respuesta else 0.0

    def _calcular_answer_coverage(
        self,
        respuesta: str,
        chunks: List[Dict],
        ground_truth: Dict
    ) -> float:
        """
        Calcular Answer Coverage: que proporcion de la informacion esperada
        esta presente en la respuesta.

        Args:
            respuesta: Respuesta generada
            chunks: Chunks usados como contexto
            ground_truth: Informacion esperada

        Returns:
            Score de cobertura (0.0 a 1.0)
        """
        if not respuesta:
            return 0.0

        expected_info = ground_truth.get('expected_info', [])
        if not expected_info:
            return 1.0  # Si no hay expectativas, asumimos cobertura completa

        respuesta_lower = respuesta.lower()

        # Contar cuantos elementos esperados aparecen en la respuesta
        encontrados = 0
        for info in expected_info:
            if info.lower() in respuesta_lower:
                encontrados += 1

        return encontrados / len(expected_info) if expected_info else 0.0

    def _guardar_resultados(self):
        """Guardar todos los resultados y reportes (sobrescribe archivos existentes)"""
        # 1. Ejemplos completos (JSON)
        ejemplos_path = self.results_dir / 'ejemplos_completos.json'
        with open(ejemplos_path, 'w', encoding='utf-8') as f:
            json.dump(self.resultados, f, indent=2, ensure_ascii=False)
        logger.info(f"Ejemplos guardados: {ejemplos_path}")

        # 2. Metricas consolidadas (JSON)
        metricas_path = self.results_dir / 'metricas.json'
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

        logs_path = self.logs_dir / 'logs_agente_critico.json'
        with open(logs_path, 'w', encoding='utf-8') as f:
            json.dump(logs_critico, f, indent=2, ensure_ascii=False)
        logger.info(f"Logs guardados: {logs_path}")

        # 4. Reporte legible (TXT)
        reporte_path = self.results_dir / 'reporte_evaluacion.txt'
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

        # Metricas avanzadas
        lines.append("\n## METRICAS AVANZADAS DE RAG\n")
        avanzadas = self.metricas.get('metricas_avanzadas', {})
        lines.append(f"Precision (Retrieval):     {avanzadas.get('precision', {}).get('promedio', 0):.4f} (min: {avanzadas.get('precision', {}).get('min', 0):.4f}, max: {avanzadas.get('precision', {}).get('max', 0):.4f})")
        lines.append(f"Recall (Retrieval):        {avanzadas.get('recall', {}).get('promedio', 0):.4f} (min: {avanzadas.get('recall', {}).get('min', 0):.4f}, max: {avanzadas.get('recall', {}).get('max', 0):.4f})")
        lines.append(f"F1-Score:                  {avanzadas.get('f1_score', 0):.4f}")
        lines.append(f"Faithfulness:              {avanzadas.get('faithfulness', {}).get('promedio', 0):.4f} (min: {avanzadas.get('faithfulness', {}).get('min', 0):.4f}, max: {avanzadas.get('faithfulness', {}).get('max', 0):.4f})")
        lines.append(f"Answer Coverage:           {avanzadas.get('answer_coverage', {}).get('promedio', 0):.4f} (min: {avanzadas.get('answer_coverage', {}).get('min', 0):.4f}, max: {avanzadas.get('answer_coverage', {}).get('max', 0):.4f})")

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
