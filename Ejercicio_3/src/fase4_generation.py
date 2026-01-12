"""
FASE 4: Generation - Generacion de Respuestas con LLM
Autor: Danilo Melo
Fecha: 2026-01-12

Funcionalidades:
- Recibir query y chunks de FASE 3
- Construir prompt estructurado con contexto
- Llamar a Groq API (Llama 3.3 70B)
- Generar respuesta con citas
- Manejar errores y reintentos
"""

import os
import re
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
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


class Fase4Generation:
    """Clase para generacion de respuestas usando LLM"""

    SYSTEM_PROMPT = """Eres un asistente experto en laptops. Tu trabajo es responder preguntas sobre especificaciones tecnicas de laptops usando UNICAMENTE la informacion proporcionada en el contexto.

REGLAS ESTRICTAS:
1. Responde SOLO con informacion que este en el CONTEXTO proporcionado
2. SIEMPRE incluye las citas en formato [laptop_id:campo] despues de cada afirmacion
3. Si no hay informacion suficiente en el contexto, responde: "No tengo informacion suficiente para responder esta pregunta."
4. NO inventes ni agregues informacion que no este en el contexto
5. Responde de forma concisa y directa
6. Maximo {max_words} palabras en tu respuesta
7. Responde en espanol"""

    USER_PROMPT_TEMPLATE = """CONTEXTO:
{context}

PREGUNTA: {query}

Responde la pregunta usando SOLO la informacion del contexto. Incluye las citas [laptop_id:campo] para cada afirmacion."""

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializar generador de respuestas

        Args:
            config: Diccionario con configuracion del proyecto
        """
        self.config = config
        self.generation_config = config['generation']

        # Configuracion del LLM
        self.provider = self.generation_config['provider']
        self.model_name = self.generation_config['model_name']
        self.temperature = self.generation_config['temperature']
        self.max_tokens = self.generation_config['max_tokens']
        self.max_words = self.generation_config['max_words_response']

        # Inicializar cliente Groq
        self.client = None
        self._init_client()

        # Estadisticas
        self.stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'avg_generation_time_ms': 0,
            'total_tokens_used': 0
        }
        self._generation_times = []

    def _init_client(self):
        """Inicializar cliente de Groq"""
        api_key = os.environ.get('GROQ_API_KEY')
        if not api_key:
            logger.warning("GROQ_API_KEY no encontrada en variables de entorno")
            logger.warning("Configura la variable en archivo .env")
        else:
            # Limpiar espacios y saltos de linea de la API key
            api_key = api_key.strip()
            self.client = Groq(api_key=api_key)
            logger.info(f"Cliente Groq inicializado - Modelo: {self.model_name}")

    def generar_respuesta(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Generar respuesta usando el LLM

        Args:
            query: Pregunta del usuario
            chunks: Lista de chunks relevantes de FASE 3
            include_metadata: Incluir metadata en el resultado

        Returns:
            Diccionario con respuesta y metadata
        """
        if not self.client:
            return {
                'query': query,
                'response': "Error: Cliente LLM no inicializado. Configura GROQ_API_KEY.",
                'error': True,
                'error_message': "GROQ_API_KEY no configurada"
            }

        start_time = time.time()
        self.stats['total_generations'] += 1

        try:
            # 1. Construir contexto desde los chunks
            context = self._build_context(chunks)

            # 2. Construir prompts
            system_prompt = self.SYSTEM_PROMPT.format(max_words=self.max_words)
            user_prompt = self.USER_PROMPT_TEMPLATE.format(
                context=context,
                query=query
            )

            # 3. Llamar al LLM
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # 4. Extraer respuesta
            generated_text = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens if response.usage else 0

            # 5. Extraer citas usadas
            citations_used = self._extract_citations(generated_text)

            # 6. Calcular tiempo
            generation_time_ms = (time.time() - start_time) * 1000

            # 7. Actualizar estadisticas
            self._update_stats(generation_time_ms, tokens_used, success=True)

            result = {
                'query': query,
                'response': generated_text,
                'citations_used': citations_used,
                'error': False
            }

            if include_metadata:
                result['metadata'] = {
                    'chunks_provided': len(chunks),
                    'generation_time_ms': round(generation_time_ms, 2),
                    'tokens_used': tokens_used,
                    'model': self.model_name,
                    'temperature': self.temperature
                }

            return result

        except Exception as e:
            generation_time_ms = (time.time() - start_time) * 1000
            self._update_stats(generation_time_ms, 0, success=False)

            logger.error(f"Error en generacion: {str(e)}")

            return {
                'query': query,
                'response': f"Error al generar respuesta: {str(e)}",
                'error': True,
                'error_message': str(e)
            }

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Construir contexto formateado desde los chunks

        Args:
            chunks: Lista de chunks

        Returns:
            String con contexto formateado
        """
        if not chunks:
            return "No se encontro informacion relevante."

        context_lines = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get('text', '')
            citation = chunk.get('citation', '')
            score = chunk.get('score', 0)

            # Formato: numero. texto [cita]
            context_lines.append(f"{i}. {text} {citation}")

        return "\n".join(context_lines)

    def _extract_citations(self, text: str) -> List[str]:
        """
        Extraer citas del texto generado

        Args:
            text: Texto generado por el LLM

        Returns:
            Lista de citas encontradas
        """
        # Patron para citas: [numero:campo] o [LAP###:campo]
        pattern = r'\[[\w\d]+:[^\]]+\]'
        citations = re.findall(pattern, text)
        return list(set(citations))  # Eliminar duplicados

    def _update_stats(self, generation_time_ms: float, tokens_used: int, success: bool):
        """Actualizar estadisticas de generacion"""
        self._generation_times.append(generation_time_ms)

        if success:
            self.stats['successful_generations'] += 1
            self.stats['total_tokens_used'] += tokens_used
        else:
            self.stats['failed_generations'] += 1

        self.stats['avg_generation_time_ms'] = round(
            sum(self._generation_times) / len(self._generation_times), 2
        )

    def obtener_estadisticas(self) -> Dict[str, Any]:
        """Retornar estadisticas de uso"""
        return self.stats.copy()


def main():
    """Funcion principal para testing de FASE 4"""
    import yaml
    from fase3_retrieval import Fase3Retrieval

    # Cargar configuracion
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Verificar API key
    if not os.environ.get('GROQ_API_KEY'):
        print("\nError: GROQ_API_KEY no configurada")
        print("Ejecuta: export GROQ_API_KEY='tu-api-key'")
        print("\nPuedes obtener una API key en: https://console.groq.com/")
        return

    # Inicializar componentes
    retrieval = Fase3Retrieval(config)
    retrieval.cargar_index()

    generation = Fase4Generation(config)

    # Queries de prueba
    test_queries = [
        "Que procesador tiene el HP 15?",
        "Que laptops tienen tarjeta grafica NVIDIA?",
        "Cual es el tamano de pantalla del Lenovo ThinkPad?",
    ]

    print("\n" + "="*70)
    print("FASE 4: Testing de Generation")
    print("="*70)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)

        # 1. Recuperar chunks relevantes (FASE 3)
        resultado_retrieval = retrieval.buscar(query)
        chunks = resultado_retrieval['retrieved_chunks']

        print(f"Chunks recuperados: {len(chunks)}")

        # 2. Generar respuesta (FASE 4)
        resultado_generation = generation.generar_respuesta(query, chunks)

        if resultado_generation['error']:
            print(f"Error: {resultado_generation.get('error_message', 'Unknown')}")
        else:
            print(f"\nRespuesta:")
            print(resultado_generation['response'])
            print(f"\nCitas usadas: {resultado_generation['citations_used']}")
            if 'metadata' in resultado_generation:
                meta = resultado_generation['metadata']
                print(f"Tiempo: {meta['generation_time_ms']} ms")
                print(f"Tokens: {meta['tokens_used']}")

    # Mostrar estadisticas finales
    print("\n" + "="*70)
    print("Estadisticas de Generation")
    print("="*70)
    stats = generation.obtener_estadisticas()
    print(f"Total generaciones: {stats['total_generations']}")
    print(f"Exitosas: {stats['successful_generations']}")
    print(f"Fallidas: {stats['failed_generations']}")
    print(f"Tiempo promedio: {stats['avg_generation_time_ms']} ms")
    print(f"Tokens totales: {stats['total_tokens_used']}")


if __name__ == "__main__":
    main()
