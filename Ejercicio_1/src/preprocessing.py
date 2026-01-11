import pandas as pd
import numpy as np
import logging
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Obtener directorio base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent


def cargar_configuracion(config_path=None):
    """Carga configuracion desde archivo YAML"""
    if config_path is None:
        config_path = BASE_DIR / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def cargar_datos(config):
    """Carga dataset raw"""
    logger.info("Cargando dataset...")
    data_path = config['paths']['data_raw']
    df = pd.read_csv(data_path)

    total_registros = len(df)
    logger.info(f"Dataset cargado: {total_registros:,} registros")
    logger.info(f"Productos unicos: {df['ITEM_ID'].nunique():,}")
    logger.info(f"Rango de fechas: {df['ORD_CLOSED_DT'].min()} a {df['ORD_CLOSED_DT'].max()}")

    return df


def limpiar_datos(df, config):
    """Limpieza de datos: nulos, duplicados, validaciones"""
    logger.info("Iniciando limpieza de datos...")

    total_inicial = len(df)

    # Nulos
    nulos_antes = df.isnull().sum().sum()
    df_clean = df.dropna()
    nulos_eliminados = total_inicial - len(df_clean)
    pct_nulos = (nulos_eliminados / total_inicial) * 100
    logger.info(f"Nulos eliminados: {nulos_eliminados:,} ({pct_nulos:.2f}%)")

    # Duplicados exactos (mismo producto, fecha Y precio)
    duplicados_antes = df_clean.duplicated(subset=['ITEM_ID', 'ORD_CLOSED_DT', 'PRICE']).sum()
    df_clean = df_clean.drop_duplicates(subset=['ITEM_ID', 'ORD_CLOSED_DT', 'PRICE'])
    logger.info(f"Duplicados exactos eliminados: {duplicados_antes:,}")

    # Precios invalidos
    df_clean = df_clean[df_clean['PRICE'] > 0]
    precios_invalidos = len(df_clean[df_clean['PRICE'] <= 0])
    logger.info(f"Precios invalidos eliminados: {precios_invalidos:,}")

    # Convertir fecha
    df_clean['ORD_CLOSED_DT'] = pd.to_datetime(df_clean['ORD_CLOSED_DT'])

    # Filtrar productos con pocos registros
    min_registros = config['preprocessing']['min_registros_producto']
    conteo_por_producto = df_clean.groupby('ITEM_ID').size()
    productos_validos = conteo_por_producto[conteo_por_producto >= min_registros].index
    df_clean = df_clean[df_clean['ITEM_ID'].isin(productos_validos)]

    productos_eliminados = df['ITEM_ID'].nunique() - df_clean['ITEM_ID'].nunique()
    logger.info(f"Productos con < {min_registros} registros eliminados: {productos_eliminados:,}")

    total_final = len(df_clean)
    pct_total_eliminado = ((total_inicial - total_final) / total_inicial) * 100
    logger.info(f"Total registros eliminados: {total_inicial - total_final:,} ({pct_total_eliminado:.2f}%)")
    logger.info(f"Dataset limpio: {total_final:,} registros, {df_clean['ITEM_ID'].nunique():,} productos")

    return df_clean


def reducir_dataset(df, config):
    """Reduce dataset para velocidad si esta configurado"""
    if not config['data']['reduce_dataset']:
        return df

    sample_size = config['data']['sample_size']
    if len(df) <= sample_size:
        logger.info(f"Dataset ya tiene {len(df)} registros, no se reduce")
        return df

    logger.info(f"Reduciendo dataset a ~{sample_size:,} registros para velocidad...")

    # Muestreo aleatorio simple
    df_reducido = df.sample(n=sample_size, random_state=config['data']['random_seed'])

    logger.info(f"Dataset reducido: {len(df_reducido):,} registros, {df_reducido['ITEM_ID'].nunique():,} productos")

    return df_reducido


def feature_engineering(df, config):
    """Creacion de features para modelos"""
    logger.info("Iniciando feature engineering...")

    ventana = config['preprocessing']['ventana_movil']
    df = df.sort_values(['ITEM_ID', 'ORD_CLOSED_DT']).reset_index(drop=True)

    # Agrupar por producto
    logger.info(f"Calculando estadisticas moviles (ventana={ventana} dias)...")

    df['PRICE_MEAN_GLOBAL'] = df.groupby('ITEM_ID')['PRICE'].transform('mean')
    df['PRICE_STD_GLOBAL'] = df.groupby('ITEM_ID')['PRICE'].transform('std')

    df['PRICE_MEAN_ROLLING'] = df.groupby('ITEM_ID')['PRICE'].transform(
        lambda x: x.rolling(window=ventana, min_periods=1).mean()
    )
    df['PRICE_STD_ROLLING'] = df.groupby('ITEM_ID')['PRICE'].transform(
        lambda x: x.rolling(window=ventana, min_periods=1).std()
    )

    df['PRICE_LAG_1'] = df.groupby('ITEM_ID')['PRICE'].shift(1)
    df['PRICE_DIFF'] = df['PRICE'] - df['PRICE_LAG_1']
    df['PRICE_DIFF_PCT'] = (df['PRICE_DIFF'] / df['PRICE_LAG_1']) * 100

    df['PRICE_DIFF_VS_MEAN'] = df['PRICE'] - df['PRICE_MEAN_ROLLING']

    df['PRICE_MIN_ROLLING'] = df.groupby('ITEM_ID')['PRICE'].transform(
        lambda x: x.rolling(window=ventana, min_periods=1).min()
    )
    df['PRICE_MAX_ROLLING'] = df.groupby('ITEM_ID')['PRICE'].transform(
        lambda x: x.rolling(window=ventana, min_periods=1).max()
    )

    # Rellenar NaN en lags
    df = df.fillna(0)

    features_creadas = [
        'PRICE_MEAN_GLOBAL',      # Media de precio de todo el historico del producto
        'PRICE_STD_GLOBAL',       # Desviacion estandar de todo el historico del producto
        'PRICE_MEAN_ROLLING',     # Media movil del precio en ventana de 7 dias
        'PRICE_STD_ROLLING',      # Desviacion estandar movil en ventana de 7 dias
        'PRICE_LAG_1',            # Precio del dia anterior
        'PRICE_DIFF',             # Diferencia absoluta entre precio actual y dia anterior
        'PRICE_DIFF_PCT',         # Diferencia porcentual entre precio actual y dia anterior
        'PRICE_DIFF_VS_MEAN',     # Diferencia entre precio actual y media movil
        'PRICE_MIN_ROLLING',      # Precio minimo en ventana de 7 dias
        'PRICE_MAX_ROLLING'       # Precio maximo en ventana de 7 dias
    ]

    logger.info(f"Features creadas: {len(features_creadas)}")
    for feat in features_creadas:
        logger.info(f"  - {feat}")

    return df


def generar_ground_truth(df, config):
    """Genera etiquetas ground truth usando Z-score"""
    logger.info("Generando ground truth (etiquetas)...")

    umbral = config['preprocessing']['umbral_zscore_ground_truth']
    metodo = config['preprocessing']['metodo_ground_truth']

    logger.info(f"Metodo: {metodo}, umbral Z-score: {umbral}")

    # Calcular Z-score por producto
    df['ZSCORE'] = (df['PRICE'] - df['PRICE_MEAN_GLOBAL']) / (df['PRICE_STD_GLOBAL'] + 1e-8)

    # Etiquetar
    df['TRUE_LABEL'] = np.where(np.abs(df['ZSCORE']) > umbral, 'ANOMALO', 'NORMAL')

    # Distribucion
    total = len(df)
    n_anomalos = (df['TRUE_LABEL'] == 'ANOMALO').sum()
    n_normales = (df['TRUE_LABEL'] == 'NORMAL').sum()
    pct_anomalos = (n_anomalos / total) * 100
    pct_normales = (n_normales / total) * 100

    logger.info(f"Distribucion de etiquetas:")
    logger.info(f"  - NORMAL: {n_normales:,} ({pct_normales:.2f}%)")
    logger.info(f"  - ANOMALO: {n_anomalos:,} ({pct_anomalos:.2f}%)")

    return df


def train_test_split_data(df, config):
    """Divide dataset en train y test"""
    logger.info("Dividiendo dataset en train/test...")

    test_size = config['data']['test_size']
    random_seed = config['data']['random_seed']

    # Split estratificado solo por label
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df['TRUE_LABEL']
    )

    logger.info(f"Train set: {len(train_df):,} registros ({(1-test_size)*100:.0f}%)")
    logger.info(f"  - NORMAL: {(train_df['TRUE_LABEL']=='NORMAL').sum():,}")
    logger.info(f"  - ANOMALO: {(train_df['TRUE_LABEL']=='ANOMALO').sum():,}")
    logger.info(f"  - Productos: {train_df['ITEM_ID'].nunique():,}")

    logger.info(f"Test set: {len(test_df):,} registros ({test_size*100:.0f}%)")
    logger.info(f"  - NORMAL: {(test_df['TRUE_LABEL']=='NORMAL').sum():,}")
    logger.info(f"  - ANOMALO: {(test_df['TRUE_LABEL']=='ANOMALO').sum():,}")
    logger.info(f"  - Productos: {test_df['ITEM_ID'].nunique():,}")

    return train_df, test_df


def guardar_datos_procesados(train_df, test_df, config):
    """Guarda datasets procesados"""
    logger.info("Guardando datos procesados...")

    output_dir = BASE_DIR / config['paths']['data_processed']
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / 'train.csv'
    test_path = output_dir / 'test.csv'

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Train guardado: {train_path}")
    logger.info(f"Test guardado: {test_path}")


def ejecutar_preprocesamiento(config_path=None):
    """Ejecuta pipeline completo de preprocesamiento"""
    logger.info("="*60)
    logger.info("INICIANDO PREPROCESAMIENTO")
    logger.info("="*60)

    config = cargar_configuracion(config_path)

    df = cargar_datos(config)
    df = limpiar_datos(df, config)
    df = reducir_dataset(df, config)
    df = feature_engineering(df, config)
    df = generar_ground_truth(df, config)
    train_df, test_df = train_test_split_data(df, config)
    guardar_datos_procesados(train_df, test_df, config)

    logger.info("="*60)
    logger.info("PREPROCESAMIENTO COMPLETADO")
    logger.info("="*60)

    return train_df, test_df


if __name__ == "__main__":
    ejecutar_preprocesamiento()
