"""
FASE 5: Agente Critico - Verificacion de Respuestas
Autor: Danilo Melo
Fecha: 2026-01-12

Funcionalidades:
- Verificar que cada afirmacion tenga soporte en los chunks
- Validar que las citas sean correctas
- Detectar alucinaciones del LLM
- Decidir APROBAR o REHACER
- Proporcionar feedback para regeneracion
"""

import os
import re
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dotenv import load_dotenv
from groq import Groq

# Cargar variables de entorno desde .env
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Fase5AgenteCritico:
    """Agente Critico para verificacion de respuestas del LLM"""

    SYSTEM_PROMPT = """Eres un agente critico experto en verificacion de respuestas. Tu trabajo es verificar si una respuesta generada esta correctamente fundamentada en los chunks de contexto proporcionados.

REGLAS DE VERIFICACION:
1. Cada afirmacion en la respuesta DEBE tener soporte en algun chunk del contexto
2. Las citas [laptop_id:campo] DEBEN corresponder al chunk correcto
3. Los datos citados DEBEN coincidir EXACTAMENTE con lo que dice el chunk
4. NO debe haber informacion inventada o alucinada

DEBES RESPONDER EN FORMATO JSON EXACTO:
{{
    "veredicto": "APROBAR" o "REHACER",
    "problemas": [
        {{
            "tipo": "alucinacion" | "cita_incorrecta" | "dato_erroneo" | "sin_soporte",
            "descripcion": "explicacion del problema",
            "cita_afectada": "[laptop_id:campo]"
        }}
    ],
    "feedback": "instrucciones especificas para corregir si es REHACER, vacio si es APROBAR"
}}

Si la respuesta es correcta y todas las afirmaciones tienen soporte, responde con veredicto "APROBAR" y problemas vacio."""

    USER_PROMPT_TEMPLATE = """CONTEXTO (chunks disponibles):
{context}

RESPUESTA A VERIFICAR:
{response}

Verifica si la respuesta esta correctamente fundamentada en los chunks. Analiza cada afirmacion y cada cita."""

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializar Agente Critico

        Args:
            config: Diccionario con configuracion del proyecto
        """
        self.config = config
        self.critical_config = config['critical_agent']
        self.generation_config = config['generation']

        # Configuracion
        self.max_retries = self.critical_config['max_retries']
        self.model_name = self.generation_config['model_name']
        self.temperature = 0.0  # Deterministico para verificacion

        # Inicializar cliente Groq
        self.client = None
        self._init_client()

        # Estadisticas
        self.stats = {
            'total_verificaciones': 0,
            'aprobadas': 0,
            'rechazadas': 0,
            'total_reintentos': 0,
            'problemas_detectados': {
                'alucinacion': 0,
                'cita_incorrecta': 0,
                'dato_erroneo': 0,
                'sin_soporte': 0
            }
        }

    def _init_client(self):
        """Inicializar cliente de Groq"""
        api_key = os.environ.get('GROQ_API_KEY')
        if not api_key:
            logger.warning("GROQ_API_KEY no encontrada en variables de entorno")
        else:
            api_key = api_key.strip()
            self.client = Groq(api_key=api_key)
            logger.info("Agente Critico inicializado")

    def verificar_respuesta(
        self,
        respuesta: str,
        chunks: List[Dict[str, Any]],
        query: str = ""
    ) -> Dict[str, Any]:
        """
        Verificar si una respuesta esta correctamente fundamentada

        Args:
            respuesta: Respuesta generada por el LLM
            chunks: Lista de chunks usados como contexto
            query: Pregunta original (opcional, para contexto)

        Returns:
            Diccionario con veredicto, problemas y feedback
        """
        if not self.client:
            return {
                'veredicto': 'ERROR',
                'problemas': [],
                'feedback': 'Cliente LLM no inicializado',
                'error': True
            }

        self.stats['total_verificaciones'] += 1
        start_time = time.time()

        try:
            # Construir contexto desde los chunks
            context = self._build_context(chunks)

            # Construir prompt
            user_prompt = self.USER_PROMPT_TEMPLATE.format(
                context=context,
                response=respuesta
            )

            # Llamar al LLM para verificacion
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=500
            )

            # Parsear respuesta
            resultado = self._parsear_veredicto(response.choices[0].message.content)

            # Actualizar estadisticas
            if resultado['veredicto'] == 'APROBAR':
                self.stats['aprobadas'] += 1
            else:
                self.stats['rechazadas'] += 1
                for problema in resultado.get('problemas', []):
                    tipo = problema.get('tipo', 'sin_soporte')
                    if tipo in self.stats['problemas_detectados']:
                        self.stats['problemas_detectados'][tipo] += 1

            resultado['verification_time_ms'] = round((time.time() - start_time) * 1000, 2)
            resultado['error'] = False

            return resultado

        except Exception as e:
            logger.error(f"Error en verificacion: {str(e)}")
            return {
                'veredicto': 'ERROR',
                'problemas': [],
                'feedback': str(e),
                'error': True
            }

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Construir contexto formateado desde los chunks"""
        if not chunks:
            return "No hay chunks disponibles."

        context_lines = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get('text', '')
            citation = chunk.get('citation', '')
            context_lines.append(f"{i}. {text} {citation}")

        return "\n".join(context_lines)

    def _parsear_veredicto(self, respuesta_llm: str) -> Dict[str, Any]:
        """Parsear respuesta JSON del agente critico"""
        try:
            respuesta_llm = respuesta_llm.strip()

            # Limpiar markdown si existe
            if '```json' in respuesta_llm:
                respuesta_llm = respuesta_llm.split('```json')[1].split('```')[0].strip()
            elif '```' in respuesta_llm:
                respuesta_llm = respuesta_llm.split('```')[1].split('```')[0].strip()

            resultado = json.loads(respuesta_llm)

            # Validar estructura
            veredicto = resultado.get('veredicto', 'REHACER').upper()
            if veredicto not in ['APROBAR', 'REHACER']:
                veredicto = 'REHACER'

            return {
                'veredicto': veredicto,
                'problemas': resultado.get('problemas', []),
                'feedback': resultado.get('feedback', '')
            }

        except Exception as e:
            logger.warning(f"Error parseando veredicto: {e}. Asumiendo REHACER")
            return {
                'veredicto': 'REHACER',
                'problemas': [{'tipo': 'error_parseo', 'descripcion': str(e)}],
                'feedback': 'No se pudo parsear la verificacion. Regenerar respuesta.'
            }

    def ejecutar_pipeline_con_critica(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        generador,  # Fase4Generation instance
    ) -> Dict[str, Any]:
        """
        Ejecutar pipeline completo: generacion + verificacion con reintentos

        Args:
            query: Pregunta del usuario
            chunks: Chunks recuperados
            generador: Instancia de Fase4Generation

        Returns:
            Resultado final con respuesta verificada
        """
        logger.info(f"Iniciando pipeline con critica para: {query[:50]}...")

        intentos = 0
        mejor_respuesta = None
        historial = []
        feedback_acumulado = ""

        while intentos < self.max_retries + 1:
            intentos += 1
            logger.info(f"Intento {intentos}/{self.max_retries + 1}")

            # Generar respuesta (con feedback si hay)
            if feedback_acumulado:
                # Agregar feedback al query para correccion
                query_con_feedback = f"{query}\n\nNOTA IMPORTANTE: {feedback_acumulado}"
                resultado_gen = generador.generar_respuesta(query_con_feedback, chunks)
            else:
                resultado_gen = generador.generar_respuesta(query, chunks)

            if resultado_gen.get('error'):
                logger.error(f"Error en generacion: {resultado_gen.get('error_message')}")
                break

            respuesta = resultado_gen['response']
            logger.info(f"Respuesta generada: {respuesta[:100]}...")

            # Verificar respuesta
            verificacion = self.verificar_respuesta(respuesta, chunks, query)

            historial.append({
                'intento': intentos,
                'respuesta': respuesta,
                'verificacion': verificacion
            })

            if verificacion.get('error'):
                logger.error("Error en verificacion")
                mejor_respuesta = resultado_gen
                break

            if verificacion['veredicto'] == 'APROBAR':
                logger.info("Respuesta APROBADA")
                return {
                    'query': query,
                    'response': respuesta,
                    'citations_used': resultado_gen.get('citations_used', []),
                    'verificacion': verificacion,
                    'intentos': intentos,
                    'aprobada': True,
                    'historial': historial
                }

            # Respuesta rechazada
            logger.info(f"Respuesta RECHAZADA: {verificacion.get('feedback', '')[:100]}")
            self.stats['total_reintentos'] += 1
            feedback_acumulado = verificacion.get('feedback', '')
            mejor_respuesta = resultado_gen

        # Se agotaron los intentos
        logger.warning(f"Se agotaron los intentos ({intentos}). Retornando mejor respuesta.")

        return {
            'query': query,
            'response': mejor_respuesta.get('response', '') if mejor_respuesta else '',
            'citations_used': mejor_respuesta.get('citations_used', []) if mejor_respuesta else [],
            'verificacion': historial[-1]['verificacion'] if historial else {},
            'intentos': intentos,
            'aprobada': False,
            'advertencia': 'Respuesta no verificada completamente. Usar con precaucion.',
            'historial': historial
        }

    def obtener_estadisticas(self) -> Dict[str, Any]:
        """Retornar estadisticas de uso"""
        return self.stats.copy()


def main():
    """Funcion principal para testing de FASE 5 (solo verificacion, sin re-ejecutar fases anteriores)"""
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

    # Inicializar solo el Agente Critico
    agente_critico = Fase5AgenteCritico(config)

    # Datos de prueba SIMULADOS (no consume API de fases anteriores)
    # Estos son ejemplos para probar la logica del agente critico
    test_cases = [
        {
            "query": "Que procesador tiene el HP 15?",
            "respuesta": "El HP 15 tiene un procesador Intel Core i5-1135G7 con un rendimiento de 9901 puntos en CPU Mark [4:cpu].",
            "chunks": [
                {
                    "chunk_id": 0,
                    "laptop_id": 4,
                    "field": "cpu",
                    "text": "HP 15 de HP tiene procesador Intel Core i5-1135G7, con rendimiento de 9901 puntos en CPU Mark",
                    "citation": "[4:cpu]"
                }
            ]
        },
        {
            "query": "Que GPU tiene el HP 15?",
            "respuesta": "El HP 15 tiene una tarjeta grafica NVIDIA RTX 3080 [4:gpu].",  # ALUCINACION - el chunk dice Intel Iris
            "chunks": [
                {
                    "chunk_id": 1,
                    "laptop_id": 4,
                    "field": "gpu",
                    "text": "HP 15 de HP tiene tarjeta grafica Intel Iris Xe Graphics G7 (80EU)",
                    "citation": "[4:gpu]"
                }
            ]
        }
    ]

    print("\n" + "="*70)
    print("FASE 5: Testing de Agente Critico")
    print("(Usando datos simulados - NO consume API de fases anteriores)")
    print("="*70)

    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Query: {test['query']}")
        print(f"Respuesta a verificar: {test['respuesta']}")
        print(f"Chunk disponible: {test['chunks'][0]['text'][:60]}...")
        print("-" * 50)

        # Verificar respuesta
        verificacion = agente_critico.verificar_respuesta(
            respuesta=test['respuesta'],
            chunks=test['chunks'],
            query=test['query']
        )

        print(f"\nVeredicto: {verificacion['veredicto']}")
        print(f"Tiempo: {verificacion.get('verification_time_ms', 0)} ms")

        if verificacion.get('problemas'):
            print("Problemas detectados:")
            for p in verificacion['problemas']:
                print(f"  - [{p.get('tipo')}] {p.get('descripcion', '')}")

        if verificacion.get('feedback'):
            print(f"Feedback: {verificacion['feedback']}")

    # Estadisticas finales
    print("\n" + "="*70)
    print("Estadisticas del Agente Critico")
    print("="*70)
    stats = agente_critico.obtener_estadisticas()
    print(f"Total verificaciones: {stats['total_verificaciones']}")
    print(f"Aprobadas: {stats['aprobadas']}")
    print(f"Rechazadas: {stats['rechazadas']}")
    print(f"Problemas detectados: {stats['problemas_detectados']}")


if __name__ == "__main__":
    main()
