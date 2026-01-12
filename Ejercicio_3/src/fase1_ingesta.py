"""
FASE 1: Ingesta y Normalizacion de Datos
Autor: Danilo Melo
Fecha: 2026-01-12

Funcionalidades:
- Cargar dataset CSV de laptops
- Seleccionar muestra de 200-400 laptops con mejor calidad
- Limpiar y normalizar datos
- Extraer campos tecnicos estructurados
- Generar reporte de normalizacion
"""

import pandas as pd
import numpy as np
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Fase1Ingesta:
    """Clase para ingesta y normalizacion de datos de laptops"""

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializar procesador de ingesta

        Args:
            config: Diccionario con configuracion del proyecto
        """
        self.config = config
        self.data_config = config['data']
        self.raw_csv_path = Path(self.data_config['raw_csv'])
        self.processed_json_path = Path(self.data_config['processed_json'])
        self.report_path = Path(self.data_config['normalization_report'])

        # Estadisticas para reporte
        self.stats = {
            'original_count': 0,
            'removed_duplicates': 0,
            'removed_incomplete': 0,
            'selected_count': 0,
            'fields_normalized': [],
            'null_handling': {}
        }

    def ejecutar(self) -> pd.DataFrame:
        """
        Ejecutar pipeline completo de FASE 1

        Returns:
            DataFrame con laptops normalizados
        """
        logger.info("Iniciando FASE 1: Ingesta y Normalizacion")

        # 1. Cargar dataset
        df = self._cargar_dataset()

        # 2. Eliminar duplicados
        df = self._eliminar_duplicados(df)

        # 3. Filtrar laptops incompletos
        df = self._filtrar_incompletos(df)

        # 4. Seleccionar muestra balanceada
        df = self._seleccionar_muestra(df)

        # 5. Normalizar campos
        df = self._normalizar_campos(df)

        # 6. Estructurar datos
        laptops_estructurados = self._estructurar_datos(df)

        # 7. Guardar resultados
        self._guardar_resultados(laptops_estructurados)

        # 8. Generar reporte
        self._generar_reporte()

        logger.info("FASE 1 completada exitosamente")
        return df

    def _cargar_dataset(self) -> pd.DataFrame:
        """Cargar dataset CSV"""
        logger.info(f"Cargando dataset: {self.raw_csv_path}")

        df = pd.read_csv(self.raw_csv_path)
        self.stats['original_count'] = len(df)

        logger.info(f"Dataset cargado: {len(df)} registros, {len(df.columns)} columnas")
        return df

    def _eliminar_duplicados(self, df: pd.DataFrame) -> pd.DataFrame:
        """Eliminar laptops duplicados"""
        logger.info("Eliminando duplicados")

        original_count = len(df)

        # Eliminar por laptop_id duplicado
        df = df.drop_duplicates(subset=['laptop_id'], keep='first')

        # Eliminar por especificaciones identicas
        tech_cols = ['producer', 'model', 'cpu', 'ram', 'disc', 'price_in_dollar']
        available_cols = [col for col in tech_cols if col in df.columns]
        df = df.drop_duplicates(subset=available_cols, keep='first')

        duplicates_removed = original_count - len(df)
        self.stats['removed_duplicates'] = duplicates_removed

        logger.info(f"Duplicados eliminados: {duplicates_removed}")
        return df

    def _filtrar_incompletos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtrar laptops con demasiados campos nulos"""
        logger.info("Filtrando laptops incompletos")

        original_count = len(df)
        min_fields = self.data_config['min_non_null_fields']

        # Contar campos no nulos por fila
        non_null_counts = df.notna().sum(axis=1)

        # Filtrar laptops con suficientes campos
        df = df[non_null_counts >= min_fields].copy()

        # Validar campos requeridos
        required_fields = self.data_config['required_fields']
        for field in required_fields:
            if field in df.columns:
                df = df[df[field].notna()].copy()

        incomplete_removed = original_count - len(df)
        self.stats['removed_incomplete'] = incomplete_removed

        logger.info(f"Laptops incompletos eliminados: {incomplete_removed}")
        logger.info(f"Laptops con specs completas: {len(df)}")

        return df

    def _seleccionar_muestra(self, df: pd.DataFrame) -> pd.DataFrame:
        """Seleccionar muestra balanceada de laptops"""
        target = self.data_config['target_laptops']
        seed = self.data_config['random_seed']

        logger.info(f"Seleccionando muestra balanceada de {target} laptops")

        if len(df) <= target:
            logger.info(f"Dataset tiene {len(df)} laptops, usando todos")
            self.stats['selected_count'] = len(df)
            return df

        # Seleccion estratificada por producer si es posible
        if 'producer' in df.columns:
            # Calcular proporciones por marca
            producer_counts = df['producer'].value_counts()

            # Seleccionar proporcionalmente
            sample_dfs = []
            for producer in producer_counts.index:
                producer_df = df[df['producer'] == producer]
                proportion = len(producer_df) / len(df)
                n_samples = max(1, int(target * proportion))
                n_samples = min(n_samples, len(producer_df))

                sample = producer_df.sample(n=n_samples, random_state=seed)
                sample_dfs.append(sample)

            df_sample = pd.concat(sample_dfs, ignore_index=True)

            # Ajustar si excede target
            if len(df_sample) > target:
                df_sample = df_sample.sample(n=target, random_state=seed)
        else:
            # Seleccion aleatoria simple
            df_sample = df.sample(n=target, random_state=seed)

        self.stats['selected_count'] = len(df_sample)
        logger.info(f"Muestra seleccionada: {len(df_sample)} laptops")

        return df_sample.reset_index(drop=True)

    def _normalizar_campos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizar campos tecnicos"""
        logger.info("Normalizando campos tecnicos")

        df = df.copy()

        # Normalizar RAM
        if 'ram' in df.columns:
            df['ram'] = df['ram'].apply(self._normalizar_ram)
            self.stats['fields_normalized'].append('ram')

        # Normalizar CPU
        if 'cpu' in df.columns:
            df['cpu'] = df['cpu'].apply(self._normalizar_cpu)
            self.stats['fields_normalized'].append('cpu')

        # Normalizar GPU
        if 'gpu' in df.columns:
            df['gpu'] = df['gpu'].apply(self._normalizar_gpu)
            self.stats['fields_normalized'].append('gpu')

        # Normalizar almacenamiento
        if 'disc' in df.columns:
            df['disc'] = df['disc'].apply(self._normalizar_disc)
            self.stats['fields_normalized'].append('disc')

        # Normalizar peso
        if 'weight' in df.columns:
            df['weight'] = df['weight'].apply(self._normalizar_weight)
            self.stats['fields_normalized'].append('weight')

        # Normalizar precio
        if 'price_in_dollar' in df.columns:
            df['price_in_dollar'] = pd.to_numeric(df['price_in_dollar'], errors='coerce')
            self.stats['fields_normalized'].append('price_in_dollar')

        # Normalizar display size
        if 'display_size' in df.columns:
            df['display_size'] = df['display_size'].apply(self._normalizar_display_size)
            self.stats['fields_normalized'].append('display_size')

        # Manejar nulos en campos opcionales
        self._manejar_nulos(df)

        logger.info(f"Campos normalizados: {len(self.stats['fields_normalized'])}")

        return df

    def _normalizar_ram(self, value: Any) -> str:
        """Normalizar valor de RAM"""
        if pd.isna(value):
            return "No especificado"

        value_str = str(value).upper().strip()

        # Extraer numero
        match = re.search(r'(\d+)\s*GB', value_str)
        if match:
            return f"{match.group(1)} GB"

        return value_str

    def _normalizar_cpu(self, value: Any) -> str:
        """Normalizar nombre de CPU"""
        if pd.isna(value):
            return "No especificado"

        value_str = str(value).strip()

        # Simplificar nombre (mantener info clave)
        # Ejemplo: "Intel Core i7-12700H" -> "Intel Core i7-12700H"
        return value_str

    def _normalizar_gpu(self, value: Any) -> str:
        """Normalizar nombre de GPU"""
        if pd.isna(value):
            return "Integrated graphics"

        value_str = str(value).strip()
        return value_str

    def _normalizar_disc(self, value: Any) -> str:
        """Normalizar almacenamiento"""
        if pd.isna(value):
            return "No especificado"

        value_str = str(value).upper().strip()

        # Estandarizar formato
        value_str = value_str.replace('TB', ' TB').replace('GB', ' GB')
        value_str = re.sub(r'\s+', ' ', value_str)

        return value_str

    def _normalizar_weight(self, value: Any) -> str:
        """Normalizar peso"""
        if pd.isna(value):
            return "No especificado"

        value_str = str(value).lower().strip()

        # Extraer numero y convertir a kg
        match = re.search(r'([\d.]+)\s*(kg|g|lbs?)', value_str)
        if match:
            num = float(match.group(1))
            unit = match.group(2)

            if unit in ['g']:
                num = num / 1000
            elif unit in ['lb', 'lbs']:
                num = num * 0.453592

            return f"{num:.2f} kg"

        return value_str

    def _normalizar_display_size(self, value: Any) -> str:
        """Normalizar tamano de pantalla"""
        if pd.isna(value):
            return "No especificado"

        value_str = str(value).strip()

        # Extraer numero
        match = re.search(r'([\d.]+)', value_str)
        if match:
            size = match.group(1)
            return f"{size} inches"

        return value_str

    def _manejar_nulos(self, df: pd.DataFrame):
        """Manejar valores nulos en campos opcionales"""
        optional_fields = {
            'hdmi_version': 'No especificado',
            'wifi_version': 'No especificado',
            'bluetooth_version': 'No especificado',
            'ethernet': 'No',
            'fingerprint_reader': 'No',
            'webcam_resolution': 'No especificado'
        }

        for field, default_value in optional_fields.items():
            if field in df.columns:
                df[field].fillna(default_value, inplace=True)
                self.stats['null_handling'][field] = default_value

    def _estructurar_datos(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Estructurar datos en formato JSON"""
        logger.info("Estructurando datos en formato JSON")

        laptops = []

        for _, row in df.iterrows():
            laptop = {
                'laptop_id': row.get('laptop_id', ''),
                'basic_info': {
                    'full_name': row.get('full_name', ''),
                    'producer': row.get('producer', ''),
                    'model': row.get('model', ''),
                    'price_in_dollar': float(row.get('price_in_dollar', 0)) if pd.notna(row.get('price_in_dollar')) else None
                },
                'specs': {
                    'cpu': row.get('cpu', 'No especificado'),
                    'cpu_mark': int(row.get('cpu_mark', 0)) if pd.notna(row.get('cpu_mark')) else None,
                    'gpu': row.get('gpu', 'Integrated graphics'),
                    'ram': row.get('ram', 'No especificado'),
                    'ram_tech': row.get('ram_tech', 'No especificado'),
                    'disc': row.get('disc', 'No especificado'),
                    'display_size': row.get('display_size', 'No especificado'),
                    'display_resolution': row.get('display_resolution', 'No especificado'),
                    'display_hz': row.get('display_hz', 'No especificado'),
                    'display_tech': row.get('display_tech', 'No especificado'),
                    'weight': row.get('weight', 'No especificado')
                },
                'connectivity': {
                    'wifi_version': row.get('wifi_version', 'No especificado'),
                    'bluetooth_version': row.get('bluetooth_version', 'No especificado'),
                    'hdmi_version': row.get('hdmi_version', 'No especificado'),
                    'ethernet': row.get('ethernet', 'No')
                },
                'metadata': {
                    'publishing_date': row.get('publishing_date', ''),
                    'review_link': row.get('review_link', ''),
                    'amazon_link': row.get('amazon_link', ''),
                    'dimensions': row.get('dimensions', 'No especificado')
                }
            }

            laptops.append(laptop)

        logger.info(f"Datos estructurados: {len(laptops)} laptops")
        return laptops

    def _guardar_resultados(self, laptops: List[Dict[str, Any]]):
        """Guardar laptops normalizados"""
        logger.info(f"Guardando resultados: {self.processed_json_path}")

        # Crear directorio si no existe
        self.processed_json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.processed_json_path, 'w', encoding='utf-8') as f:
            json.dump(laptops, f, indent=2, ensure_ascii=False)

        logger.info(f"Resultados guardados: {len(laptops)} laptops")

    def _generar_reporte(self):
        """Generar reporte de normalizacion"""
        logger.info(f"Generando reporte: {self.report_path}")

        reporte = {
            'fecha_ejecucion': datetime.now().isoformat(),
            'estadisticas': self.stats,
            'archivos': {
                'input': str(self.raw_csv_path),
                'output': str(self.processed_json_path)
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

    # Ejecutar FASE 1
    fase1 = Fase1Ingesta(config)
    df = fase1.ejecutar()

    print(f"\nFASE 1 completada:")
    print(f"- Laptops procesados: {len(df)}")
    print(f"- Archivo generado: {fase1.processed_json_path}")
    print(f"- Reporte: {fase1.report_path}")


if __name__ == "__main__":
    main()
