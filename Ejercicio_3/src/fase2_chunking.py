"""
FASE 2: Chunking e Indexacion con Embeddings
Autor: Danilo Melo
Fecha: 2026-01-12

Funcionalidades:
- Cargar laptops normalizados de FASE 1
- Crear chunks field-based (un chunk por campo)
- Contar tokens para mantener rango 50-120
- Generar embeddings con sentence-transformers
- Guardar chunks e index de embeddings
- Generar reporte de chunking
"""

import json
import pickle
import logging
import tiktoken
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Fase2Chunking:
    """Clase para chunking e indexacion con embeddings"""

    # Mapeo de campos a nombres descriptivos en espanol
    FIELD_NAMES = {
        'cpu': 'procesador',
        'ram': 'memoria RAM',
        'gpu': 'tarjeta grafica',
        'disc': 'almacenamiento',
        'display_size': 'tamano de pantalla',
        'price_in_dollar': 'precio',
        'weight': 'peso',
        'wifi_version': 'version WiFi',
        'bluetooth_version': 'version Bluetooth',
        'hdmi_version': 'version HDMI',
        'ethernet': 'puerto Ethernet',
        'display_resolution': 'resolucion de pantalla',
        'display_hz': 'frecuencia de pantalla',
        'ram_tech': 'tecnologia RAM'
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializar procesador de chunking

        Args:
            config: Diccionario con configuracion del proyecto
        """
        self.config = config
        self.chunking_config = config['chunking']
        self.indexing_config = config['indexing']

        self.processed_json_path = Path(config['data']['processed_json'])
        self.chunks_path = Path('data/processed/chunks.json')
        self.embeddings_path = Path('outputs/models/embeddings.pkl')
        self.report_path = Path('data/processed/chunking_report.json')

        # Configuracion de chunking
        self.min_tokens = self.chunking_config['min_tokens']
        self.max_tokens = self.chunking_config['max_tokens']

        # Inicializar tokenizer
        self.encoder = tiktoken.get_encoding("cl100k_base")

        # Modelo de embeddings
        self.embedding_model_name = self.indexing_config['embedding_model']
        self.embedding_model = None

        # Estadisticas
        self.stats = {
            'total_laptops': 0,
            'total_chunks': 0,
            'chunks_by_field': {},
            'token_stats': {
                'min': 0,
                'max': 0,
                'mean': 0,
                'std': 0
            },
            'discarded_chunks': 0,
            'embedding_dimension': 0
        }

    def ejecutar(self) -> Tuple[List[Dict], np.ndarray]:
        """
        Ejecutar pipeline completo de FASE 2

        Returns:
            Tuple con (chunks, embeddings)
        """
        logger.info("Iniciando FASE 2: Chunking e Indexacion con Embeddings")

        # 1. Cargar laptops normalizados
        laptops = self._cargar_laptops()

        # 2. Crear chunks field-based
        chunks = self._crear_chunks(laptops)

        # 3. Generar embeddings
        embeddings = self._generar_embeddings(chunks)

        # 4. Guardar resultados
        self._guardar_chunks(chunks)
        self._guardar_embeddings(chunks, embeddings)

        # 5. Generar reporte
        self._generar_reporte()

        logger.info("FASE 2 completada exitosamente")
        return chunks, embeddings

    def _cargar_laptops(self) -> List[Dict[str, Any]]:
        """Cargar laptops normalizados de FASE 1"""
        logger.info(f"Cargando laptops: {self.processed_json_path}")

        with open(self.processed_json_path, 'r', encoding='utf-8') as f:
            laptops = json.load(f)

        self.stats['total_laptops'] = len(laptops)
        logger.info(f"Laptops cargados: {len(laptops)}")

        return laptops

    def _crear_chunks(self, laptops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Crear chunks field-based para cada laptop

        Args:
            laptops: Lista de laptops normalizados

        Returns:
            Lista de chunks
        """
        logger.info("Creando chunks field-based")

        chunks = []
        chunk_id = 0

        for laptop in laptops:
            laptop_id = laptop['laptop_id']
            producer = laptop['basic_info']['producer']
            model = laptop['basic_info']['model']
            full_name = laptop['basic_info'].get('full_name', f"{producer} {model}")

            # Chunks de especificaciones tecnicas
            specs_chunks = self._crear_chunks_specs(
                laptop_id, full_name, producer, laptop['specs'], chunk_id
            )
            chunks.extend(specs_chunks)
            chunk_id += len(specs_chunks)

            # Chunk de precio
            price_chunk = self._crear_chunk_precio(
                laptop_id, full_name, producer, laptop['basic_info'], chunk_id
            )
            if price_chunk:
                chunks.append(price_chunk)
                chunk_id += 1

            # Chunks de conectividad
            connectivity_chunks = self._crear_chunks_connectivity(
                laptop_id, full_name, producer, laptop['connectivity'], chunk_id
            )
            chunks.extend(connectivity_chunks)
            chunk_id += len(connectivity_chunks)

        self.stats['total_chunks'] = len(chunks)
        logger.info(f"Chunks creados: {len(chunks)}")

        # Calcular estadisticas de tokens
        self._calcular_estadisticas_tokens(chunks)

        return chunks

    def _crear_chunks_specs(
        self,
        laptop_id: str,
        full_name: str,
        producer: str,
        specs: Dict[str, Any],
        start_chunk_id: int
    ) -> List[Dict[str, Any]]:
        """Crear chunks para especificaciones tecnicas"""
        chunks = []
        chunk_id = start_chunk_id

        # Campos principales de specs con informacion adicional
        fields_config = [
            ('cpu', 'cpu_mark'),
            ('ram', 'ram_tech'),
            ('gpu', None),
            ('disc', None),
            ('display_size', 'display_resolution'),
            ('display_resolution', 'display_hz'),
            ('weight', None)
        ]

        for field_tuple in fields_config:
            field = field_tuple[0]
            extra_field = field_tuple[1] if len(field_tuple) > 1 else None

            value = specs.get(field)

            # Validar que el valor no sea "No especificado" o similar o NaN
            if not value or value in ['No especificado', 'No', ''] or (isinstance(value, float) and np.isnan(value)):
                continue

            # Crear texto descriptivo con informacion adicional si existe
            field_name = self.FIELD_NAMES.get(field, field)
            text = f"{full_name} de {producer} tiene {field_name} {value}"

            # Agregar informacion extra si existe
            if extra_field:
                extra_value = specs.get(extra_field)
                if extra_value and extra_value not in ['No especificado', ''] and not (isinstance(extra_value, float) and np.isnan(extra_value)):
                    extra_field_name = self.FIELD_NAMES.get(extra_field, extra_field)
                    if field == 'cpu' and extra_field == 'cpu_mark':
                        text += f", con rendimiento de {extra_value} puntos en CPU Mark"
                    elif field == 'ram' and extra_field == 'ram_tech':
                        text += f" de tipo {extra_value}"
                    elif field == 'display_size' and extra_field == 'display_resolution':
                        text += f" con resolucion{extra_value}"
                    elif field == 'display_resolution' and extra_field == 'display_hz':
                        text += f" y frecuencia de{extra_value}"

            # Contar tokens
            tokens = len(self.encoder.encode(text))

            # Validar rango de tokens - mas flexible en el minimo
            if tokens < self.min_tokens or tokens > self.max_tokens:
                self.stats['discarded_chunks'] += 1
                continue

            # Crear chunk
            chunk = {
                'chunk_id': chunk_id,
                'laptop_id': laptop_id,
                'field': field,
                'text': text,
                'citation': f"[{laptop_id}:{field}]",
                'tokens': tokens
            }

            chunks.append(chunk)
            chunk_id += 1

            # Actualizar estadisticas por campo
            if field not in self.stats['chunks_by_field']:
                self.stats['chunks_by_field'][field] = 0
            self.stats['chunks_by_field'][field] += 1

        return chunks

    def _crear_chunk_precio(
        self,
        laptop_id: str,
        full_name: str,
        producer: str,
        basic_info: Dict[str, Any],
        chunk_id: int
    ) -> Dict[str, Any]:
        """Crear chunk para precio"""
        price = basic_info.get('price_in_dollar')

        # Validar que el precio sea valido (no null, no NaN, no 0)
        if price is None or (isinstance(price, float) and (np.isnan(price) or price == 0)):
            return None

        # Crear texto
        text = f"{full_name} de {producer} tiene precio ${price:.2f} dolares"

        # Contar tokens
        tokens = len(self.encoder.encode(text))

        # Validar rango
        if tokens < self.min_tokens or tokens > self.max_tokens:
            self.stats['discarded_chunks'] += 1
            return None

        # Actualizar estadisticas
        if 'price_in_dollar' not in self.stats['chunks_by_field']:
            self.stats['chunks_by_field']['price_in_dollar'] = 0
        self.stats['chunks_by_field']['price_in_dollar'] += 1

        return {
            'chunk_id': chunk_id,
            'laptop_id': laptop_id,
            'field': 'price_in_dollar',
            'text': text,
            'citation': f"[{laptop_id}:price_in_dollar]",
            'tokens': tokens
        }

    def _crear_chunks_connectivity(
        self,
        laptop_id: str,
        full_name: str,
        producer: str,
        connectivity: Dict[str, Any],
        start_chunk_id: int
    ) -> List[Dict[str, Any]]:
        """Crear chunks para conectividad"""
        chunks = []
        chunk_id = start_chunk_id

        # Campos de conectividad
        conn_fields = ['wifi_version', 'bluetooth_version', 'hdmi_version', 'ethernet']

        for field in conn_fields:
            value = connectivity.get(field)

            # Validar valor (incluyendo NaN)
            if not value or value in ['No especificado', 'No', ''] or (isinstance(value, float) and np.isnan(value)):
                continue

            # Crear texto
            field_name = self.FIELD_NAMES.get(field, field)
            text = f"{full_name} de {producer} tiene {field_name} {value}"

            # Contar tokens
            tokens = len(self.encoder.encode(text))

            # Validar rango
            if tokens < self.min_tokens or tokens > self.max_tokens:
                self.stats['discarded_chunks'] += 1
                continue

            chunk = {
                'chunk_id': chunk_id,
                'laptop_id': laptop_id,
                'field': field,
                'text': text,
                'citation': f"[{laptop_id}:{field}]",
                'tokens': tokens
            }

            chunks.append(chunk)
            chunk_id += 1

            # Actualizar estadisticas
            if field not in self.stats['chunks_by_field']:
                self.stats['chunks_by_field'][field] = 0
            self.stats['chunks_by_field'][field] += 1

        return chunks

    def _calcular_estadisticas_tokens(self, chunks: List[Dict[str, Any]]):
        """Calcular estadisticas de tokens"""
        if not chunks:
            return

        token_counts = [chunk['tokens'] for chunk in chunks]

        self.stats['token_stats'] = {
            'min': int(np.min(token_counts)),
            'max': int(np.max(token_counts)),
            'mean': float(np.mean(token_counts)),
            'std': float(np.std(token_counts))
        }

        logger.info(f"Estadisticas de tokens - min: {self.stats['token_stats']['min']}, "
                   f"max: {self.stats['token_stats']['max']}, "
                   f"mean: {self.stats['token_stats']['mean']:.1f}")

    def _generar_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generar embeddings para todos los chunks

        Args:
            chunks: Lista de chunks

        Returns:
            Array de embeddings
        """
        logger.info("Generando embeddings con sentence-transformers")
        logger.info(f"Modelo: {self.embedding_model_name}")

        # Cargar modelo
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Extraer textos
        chunk_texts = [chunk['text'] for chunk in chunks]

        # Generar embeddings
        logger.info(f"Generando embeddings para {len(chunk_texts)} chunks...")
        embeddings = self.embedding_model.encode(
            chunk_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalizar para cosine similarity
        )

        self.stats['embedding_dimension'] = embeddings.shape[1]
        logger.info(f"Embeddings generados: shape {embeddings.shape}")

        return embeddings

    def _guardar_chunks(self, chunks: List[Dict[str, Any]]):
        """Guardar chunks en JSON"""
        logger.info(f"Guardando chunks: {self.chunks_path}")

        # Crear directorio si no existe
        self.chunks_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        logger.info(f"Chunks guardados: {len(chunks)}")

    def _guardar_embeddings(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Guardar embeddings e index"""
        logger.info(f"Guardando embeddings: {self.embeddings_path}")

        # Crear directorio si no existe
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)

        # Guardar embeddings con metadata
        embedding_index = {
            'embeddings': embeddings,
            'model_name': self.embedding_model_name,
            'embedding_dimension': embeddings.shape[1],
            'num_chunks': len(chunks),
            'chunk_ids': [chunk['chunk_id'] for chunk in chunks],
            'laptop_ids': [chunk['laptop_id'] for chunk in chunks],
            'fecha_creacion': datetime.now().isoformat()
        }

        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(embedding_index, f)

        logger.info(f"Embeddings guardados: {embeddings.shape}")

    def _generar_reporte(self):
        """Generar reporte de chunking"""
        logger.info(f"Generando reporte: {self.report_path}")

        reporte = {
            'fecha_ejecucion': datetime.now().isoformat(),
            'estadisticas': self.stats,
            'configuracion': {
                'min_tokens': self.min_tokens,
                'max_tokens': self.max_tokens,
                'strategy': self.chunking_config['chunking_strategy'],
                'embedding_model': self.embedding_model_name
            },
            'archivos': {
                'input': str(self.processed_json_path),
                'chunks': str(self.chunks_path),
                'embeddings': str(self.embeddings_path)
            }
        }

        with open(self.report_path, 'w', encoding='utf-8') as f:
            json.dump(reporte, f, indent=2, ensure_ascii=False)

        logger.info("Reporte generado exitosamente")


def main():
    """Funcion principal para testing"""
    import yaml

    # Cargar configuracion
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Ejecutar FASE 2
    fase2 = Fase2Chunking(config)
    chunks, embeddings = fase2.ejecutar()

    print(f"\nFASE 2 completada:")
    print(f"- Total laptops: {fase2.stats['total_laptops']}")
    print(f"- Total chunks: {fase2.stats['total_chunks']}")
    print(f"- Chunks descartados: {fase2.stats['discarded_chunks']}")
    print(f"- Dimension embeddings: {fase2.stats['embedding_dimension']}")
    print(f"- Tokens (min/max/mean): {fase2.stats['token_stats']['min']}/{fase2.stats['token_stats']['max']}/{fase2.stats['token_stats']['mean']:.1f}")
    print(f"\nChunks por campo:")
    for field, count in sorted(fase2.stats['chunks_by_field'].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {field}: {count}")
    print(f"\nArchivos generados:")
    print(f"  - {fase2.chunks_path}")
    print(f"  - {fase2.embeddings_path}")
    print(f"  - {fase2.report_path}")


if __name__ == "__main__":
    main()
