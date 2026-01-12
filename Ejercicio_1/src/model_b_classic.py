import pandas as pd
import numpy as np
import logging
import yaml
import json
import time
import pickle
from pathlib import Path
from sklearn.ensemble import IsolationForest

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


def cargar_configuracion(config_path=None):
    """Carga configuracion desde archivo YAML"""
    if config_path is None:
        config_path = BASE_DIR / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def cargar_datos(config):
    """Carga datasets de train y test"""
    logger.info("Cargando datos de train y test...")

    train_path = BASE_DIR / config['paths']['data_processed'] / 'train.csv'
    test_path = BASE_DIR / config['paths']['data_processed'] / 'test.csv'

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    logger.info(f"Train cargado: {len(df_train):,} registros")
    logger.info(f"Test cargado: {len(df_test):,} registros")

    return df_train, df_test


def preparar_features(df):
    """Prepara features para Isolation Forest"""
    feature_cols = [
        'PRICE',
        'PRICE_LAG_1',
        'PRICE_MEAN_GLOBAL',
        'PRICE_STD_GLOBAL',
        'PRICE_MEAN_ROLLING',
        'PRICE_STD_ROLLING',
        'PRICE_DIFF_VS_MEAN',
        'PRICE_DIFF_PCT',
        'PRICE_DIFF',
        'ZSCORE'
    ]

    # Verificar que todas las features existen
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas en el dataset: {missing_cols}")

    X = df[feature_cols].copy()

    # Manejar NaN si existen
    if X.isnull().any().any():
        logger.warning("Encontrados valores NaN en features. Rellenando con media...")
        X = X.fillna(X.mean())

    return X, feature_cols


def entrenar_isolation_forest(X_train, config):
    """Entrena modelo Isolation Forest"""
    logger.info("Entrenando Isolation Forest...")

    start_time = time.time()

    model = IsolationForest(
        contamination=config['model_b']['contamination'],
        n_estimators=config['model_b']['n_estimators'],
        max_samples=config['model_b']['max_samples'],
        random_state=config['model_b']['random_state'],
        n_jobs=config['model_b']['n_jobs']
    )

    model.fit(X_train)

    tiempo_entrenamiento = time.time() - start_time

    logger.info(f"Modelo entrenado en {tiempo_entrenamiento:.2f} segundos")
    logger.info(f"Parametros: contamination={config['model_b']['contamination']}, "
                f"n_estimators={config['model_b']['n_estimators']}, "
                f"max_samples={config['model_b']['max_samples']}")

    return model, tiempo_entrenamiento


def predecir_con_isolation_forest(model, X_test):
    """Genera predicciones con Isolation Forest"""
    logger.info("Generando predicciones con Isolation Forest...")

    start_time = time.time()

    # Predicciones: -1 para anomalias, 1 para normal
    predictions = model.predict(X_test)

    # Scores de anomalia (mas negativo = mas anomalo)
    scores = model.score_samples(X_test)

    tiempo_prediccion = time.time() - start_time
    latencia_promedio = (tiempo_prediccion / len(X_test)) * 1000  # en ms

    logger.info(f"Predicciones completadas en {tiempo_prediccion:.2f} segundos")
    logger.info(f"Latencia promedio: {latencia_promedio:.2f}ms por prediccion")

    # Convertir predicciones a formato compatible
    # -1 (anomalia) -> ANOMALO, 1 (normal) -> NORMAL
    labels = ['ANOMALO' if p == -1 else 'NORMAL' for p in predictions]

    # Convertir scores a confidence [0,1]
    # Scores son negativos, normalizamos al rango [0,1]
    # Mas negativo = mas anomalo = mayor confidence de anomalia
    min_score = scores.min()
    max_score = scores.max()

    if max_score != min_score:
        normalized_scores = (scores - min_score) / (max_score - min_score)
    else:
        normalized_scores = np.ones_like(scores) * 0.5

    # Invertir para que valores altos = alta confidence
    confidences = 1 - normalized_scores

    return labels, confidences, scores, latencia_promedio


def calcular_metricas_isolation_forest(df, tiempo_entrenamiento, latencia_promedio):
    """Calcula metricas de performance del Isolation Forest"""
    logger.info("Calculando metricas de Isolation Forest...")

    total_predicciones = len(df)
    predicciones_anomalas = (df['PRED_IF'] == 'ANOMALO').sum()
    pct_anomalas = (predicciones_anomalas / total_predicciones) * 100

    confidence_promedio = df['CONFIDENCE_IF'].mean()
    score_promedio = df['SCORE_IF'].mean()

    logger.info(f"Tiempo entrenamiento: {tiempo_entrenamiento:.2f}s")
    logger.info(f"Latencia promedio: {latencia_promedio:.2f}ms")
    logger.info(f"Predicciones ANOMALO: {predicciones_anomalas:,} ({pct_anomalas:.2f}%)")
    logger.info(f"Confidence promedio: {confidence_promedio:.3f}")
    logger.info(f"Score promedio: {score_promedio:.4f}")

    metricas = {
        'tiempo_entrenamiento_s': tiempo_entrenamiento,
        'latencia_promedio_ms': latencia_promedio,
        'total_predicciones': total_predicciones,
        'predicciones_anomalas': int(predicciones_anomalas),
        'pct_anomalas': pct_anomalas,
        'confidence_promedio': confidence_promedio,
        'score_promedio': score_promedio,
        'costo_total_usd': 0.0,  # Isolation Forest es gratis
        'costo_promedio_usd': 0.0
    }

    return metricas


def guardar_modelo_y_predicciones(model, df_test, metricas, feature_cols, config):
    """Guarda modelo, predicciones y metricas"""
    logger.info("Guardando modelo y predicciones...")

    # Guardar modelo
    models_dir = BASE_DIR / config['paths']['models']
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / 'isolation_forest.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_cols': feature_cols,
            'config': config['model_b']
        }, f)
    logger.info(f"Modelo guardado: {model_path}")

    # Guardar predicciones
    results_dir = BASE_DIR / config['paths']['results']
    results_dir.mkdir(parents=True, exist_ok=True)

    pred_path = results_dir / 'predicciones_isolation_forest.csv'
    df_test.to_csv(pred_path, index=False)
    logger.info(f"Predicciones guardadas: {pred_path}")

    # Guardar metricas
    metricas_path = results_dir / 'metricas_isolation_forest.json'
    with open(metricas_path, 'w') as f:
        json.dump(metricas, f, indent=2)
    logger.info(f"Metricas guardadas: {metricas_path}")


def ejecutar_modelo_b(config_path=None):
    """Ejecuta pipeline completo del Modelo B - Isolation Forest"""
    logger.info("="*60)
    logger.info("MODELO B - ISOLATION FOREST")
    logger.info("="*60)

    config = cargar_configuracion(config_path)

    # Cargar datos
    df_train, df_test = cargar_datos(config)

    # Preparar features
    X_train, feature_cols = preparar_features(df_train)
    X_test, _ = preparar_features(df_test)

    # Entrenar modelo
    model, tiempo_entrenamiento = entrenar_isolation_forest(X_train, config)

    # Generar predicciones
    labels, confidences, scores, latencia_promedio = predecir_con_isolation_forest(model, X_test)

    # Agregar predicciones al dataframe de test
    df_test['PRED_IF'] = labels
    df_test['CONFIDENCE_IF'] = confidences
    df_test['SCORE_IF'] = scores

    # Calcular metricas
    metricas = calcular_metricas_isolation_forest(df_test, tiempo_entrenamiento, latencia_promedio)

    # Guardar modelo, predicciones y metricas
    guardar_modelo_y_predicciones(model, df_test, metricas, feature_cols, config)

    logger.info("="*60)
    logger.info("MODELO B COMPLETADO")
    logger.info("="*60)

    return model, df_test, metricas


if __name__ == "__main__":
    ejecutar_modelo_b()
