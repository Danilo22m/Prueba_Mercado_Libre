"""
FASE 3: Retrieval - Recuperacion de Chunks Relevantes
Autor: Danilo Melo
Fecha: 2026-01-12

Funcionalidades:
- Cargar chunks e embeddings de FASE 2
- Convertir query a embedding
- Busqueda por similitud coseno
- Retornar top-k chunks relevantes
- Filtrar por score minimo
"""

import json
import pickle
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Fase3Retrieval:
    """Clase para recuperacion de chunks relevantes usando embeddings"""

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializar sistema de retrieval

        Args:
            config: Diccionario con configuracion del proyecto
        """
        self.config = config
        self.retrieval_config = config['retrieval']
        self.indexing_config = config['indexing']

        # Paths
        self.chunks_path = Path('data/processed/chunks.json')
        self.embeddings_path = Path('outputs/models/embeddings.pkl')

        # Configuracion de retrieval
        self.top_k = self.retrieval_config['top_k']
        self.min_score = self.retrieval_config['min_score']

        # Modelo de embeddings
        self.embedding_model_name = self.indexing_config['embedding_model']
        self.embedding_model = None

        # Datos cargados
        self.chunks = None
        self.embeddings = None
        self.embedding_index = None

        # Estadisticas
        self.stats = {
            'total_queries': 0,
            'avg_retrieval_time_ms': 0,
            'avg_chunks_retrieved': 0,
            'queries_with_no_results': 0
        }
        self._retrieval_times = []
        self._chunks_retrieved = []

    def cargar_index(self):
        """Cargar chunks e embeddings desde archivos"""
        logger.info("Cargando index de chunks y embeddings")

        # Cargar chunks
        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        logger.info(f"Chunks cargados: {len(self.chunks)}")

        # Cargar embeddings
        with open(self.embeddings_path, 'rb') as f:
            self.embedding_index = pickle.load(f)
        self.embeddings = self.embedding_index['embeddings']
        logger.info(f"Embeddings cargados: shape {self.embeddings.shape}")

        # Cargar modelo de embeddings
        logger.info(f"Cargando modelo: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        logger.info("Modelo cargado exitosamente")

    def buscar(self, query: str, top_k: Optional[int] = None, min_score: Optional[float] = None) -> Dict[str, Any]:
        """
        Buscar chunks relevantes para una query

        Args:
            query: Pregunta del usuario
            top_k: Numero de chunks a retornar (opcional, usa config si no se especifica)
            min_score: Score minimo para considerar relevante (opcional)

        Returns:
            Diccionario con query, chunks recuperados y metadata
        """
        if self.chunks is None or self.embeddings is None:
            self.cargar_index()

        # Usar valores de config si no se especifican
        top_k = top_k or self.top_k
        min_score = min_score or self.min_score

        start_time = time.time()

        # 1. Generar embedding de la query
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # 2. Calcular similitud coseno con todos los chunks
        # Como los embeddings estan normalizados, el producto punto = similitud coseno
        similarities = np.dot(self.embeddings, query_embedding)

        # 3. Obtener indices de los top-k chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # 4. Construir resultado con chunks relevantes
        retrieved_chunks = []
        for idx in top_indices:
            score = float(similarities[idx])

            # Filtrar por score minimo
            if score < min_score:
                continue

            chunk = self.chunks[idx].copy()
            chunk['score'] = round(score, 4)
            retrieved_chunks.append(chunk)

        # Calcular tiempo de retrieval
        retrieval_time_ms = (time.time() - start_time) * 1000

        # Actualizar estadisticas
        self._actualizar_estadisticas(retrieval_time_ms, len(retrieved_chunks))

        result = {
            'query': query,
            'retrieved_chunks': retrieved_chunks,
            'num_chunks_retrieved': len(retrieved_chunks),
            'retrieval_time_ms': round(retrieval_time_ms, 2),
            'config': {
                'top_k': top_k,
                'min_score': min_score
            }
        }

        return result

    def _actualizar_estadisticas(self, retrieval_time_ms: float, num_chunks: int):
        """Actualizar estadisticas de retrieval"""
        self.stats['total_queries'] += 1
        self._retrieval_times.append(retrieval_time_ms)
        self._chunks_retrieved.append(num_chunks)

        if num_chunks == 0:
            self.stats['queries_with_no_results'] += 1

        self.stats['avg_retrieval_time_ms'] = round(
            sum(self._retrieval_times) / len(self._retrieval_times), 2
        )
        self.stats['avg_chunks_retrieved'] = round(
            sum(self._chunks_retrieved) / len(self._chunks_retrieved), 2
        )

    def obtener_contexto_para_llm(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Obtener contexto formateado para enviar al LLM

        Args:
            query: Pregunta del usuario
            top_k: Numero de chunks a recuperar

        Returns:
            String con contexto formateado para el prompt del LLM
        """
        resultado = self.buscar(query, top_k=top_k)

        if not resultado['retrieved_chunks']:
            return "No se encontro informacion relevante en la base de datos."

        contexto_lines = ["Informacion relevante encontrada:"]
        contexto_lines.append("")

        for i, chunk in enumerate(resultado['retrieved_chunks'], 1):
            contexto_lines.append(f"{i}. {chunk['text']} {chunk['citation']}")

        return "\n".join(contexto_lines)

    def obtener_estadisticas(self) -> Dict[str, Any]:
        """Retornar estadisticas de uso"""
        return self.stats.copy()


def main():
    """Funcion principal para testing de FASE 3"""
    import yaml

    # Cargar configuracion
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Inicializar retrieval
    retrieval = Fase3Retrieval(config)
    retrieval.cargar_index()

    # Queries de prueba
    test_queries = [
        "Que procesador tiene el HP 15?",
        "Cual es la laptop mas liviana?",
        "Que laptops tienen tarjeta grafica NVIDIA?",
        "Laptops con pantalla de 15.6 pulgadas",
        "Que laptop tiene mejor rendimiento de CPU?",
    ]

    print("\n" + "="*70)
    print("FASE 3: Testing de Retrieval")
    print("="*70)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)

        resultado = retrieval.buscar(query)

        print(f"Chunks recuperados: {resultado['num_chunks_retrieved']}")
        print(f"Tiempo: {resultado['retrieval_time_ms']} ms")
        print("\nTop chunks:")

        for chunk in resultado['retrieved_chunks'][:3]:
            print(f"  [{chunk['score']:.3f}] {chunk['citation']}")
            print(f"          {chunk['text'][:80]}...")

    # Mostrar estadisticas finales
    print("\n" + "="*70)
    print("Estadisticas de Retrieval")
    print("="*70)
    stats = retrieval.obtener_estadisticas()
    print(f"Total queries: {stats['total_queries']}")
    print(f"Tiempo promedio: {stats['avg_retrieval_time_ms']} ms")
    print(f"Chunks promedio recuperados: {stats['avg_chunks_retrieved']}")
    print(f"Queries sin resultados: {stats['queries_with_no_results']}")

    # Mostrar ejemplo de contexto para LLM
    print("\n" + "="*70)
    print("Ejemplo de contexto para LLM")
    print("="*70)
    contexto = retrieval.obtener_contexto_para_llm("Que GPU tiene el Lenovo ThinkPad?")
    print(contexto)


if __name__ == "__main__":
    main()
