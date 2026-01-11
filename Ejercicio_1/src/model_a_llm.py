import pandas as pd
import numpy as np
import logging
import yaml
import json
import time
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

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


def cargar_datos_test(config):
    """Carga dataset de test"""
    logger.info("Cargando datos de test...")
    test_path = BASE_DIR / config['paths']['data_processed'] / 'test.csv'
    df = pd.read_csv(test_path)
    logger.info(f"Test cargado: {len(df):,} registros")

    if config['model_a'].get('use_sample', False):
        sample_pct = config['model_a'].get('sample_pct', 0.05)
        sample_size = int(len(df) * sample_pct)
        logger.info(f"Usando muestra del {sample_pct*100:.1f}% para LLM ({sample_size:,} registros)")
        df = df.sample(n=sample_size, random_state=config['data']['random_seed'])
        logger.info(f"Muestra estratificada: {(df['TRUE_LABEL']=='ANOMALO').sum():,} anomalos, {(df['TRUE_LABEL']=='NORMAL').sum():,} normales")

    return df


def crear_prompt(row):
    """Crea prompt para LLM con datos de una fila"""
    prompt = f"""Eres un experto detector de anomalias de precios.

Analiza el siguiente precio y determina si es una anomalia:

Precio actual: {row['PRICE']:.2f}
Precio dia anterior: {row['PRICE_LAG_1']:.2f}
Media historica: {row['PRICE_MEAN_GLOBAL']:.2f}
Desviacion estandar: {row['PRICE_STD_GLOBAL']:.2f}
Media movil 7 dias: {row['PRICE_MEAN_ROLLING']:.2f}
Diferencia vs media: {row['PRICE_DIFF_VS_MEAN']:.2f}
Cambio porcentual: {row['PRICE_DIFF_PCT']:.2f}%

Responde SOLO en formato JSON exacto:
{{"label": "ANOMALO" o "NORMAL", "confidence": 0.0-1.0, "reason": "explicacion breve"}}"""

    return prompt


def llamar_llm(prompt, config, client):
    """Llama al LLM usando OpenAI API"""
    response = client.chat.completions.create(
        model=config['model_a']['model_name'],
        messages=[{"role": "user", "content": prompt}],
        temperature=config['model_a']['temperature'],
        max_tokens=config['model_a']['max_tokens']
    )

    content = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    return content, input_tokens, output_tokens


def parsear_respuesta(respuesta_llm):
    """Parsea respuesta JSON del LLM"""
    try:
        respuesta_llm = respuesta_llm.strip()

        if '```json' in respuesta_llm:
            respuesta_llm = respuesta_llm.split('```json')[1].split('```')[0].strip()
        elif '```' in respuesta_llm:
            respuesta_llm = respuesta_llm.split('```')[1].split('```')[0].strip()

        resultado = json.loads(respuesta_llm)

        label = resultado.get('label', 'NORMAL').upper()
        if label not in ['ANOMALO', 'NORMAL']:
            label = 'NORMAL'

        confidence = float(resultado.get('confidence', 0.5))
        confidence = max(0.0, min(1.0, confidence))

        reason = resultado.get('reason', 'Sin razon')[:100]

        return label, confidence, reason

    except Exception as e:
        logger.warning(f"Error parseando respuesta: {e}. Usando default NORMAL")
        return 'NORMAL', 0.5, 'Error en parseo'


def predecir_con_llm(df, config):
    """Ejecuta predicciones con LLM para todo el dataset"""
    logger.info("Iniciando predicciones con LLM...")

    api_key = os.getenv(config['model_a']['api_key_env'])
    if not api_key:
        raise ValueError(f"API key no encontrada. Define {config['model_a']['api_key_env']} en variables de entorno")

    api_key = api_key.strip()
    client = OpenAI(api_key=api_key)

    predicciones = []
    confidences = []
    reasons = []
    latencias = []
    costos = []
    input_tokens_list = []
    output_tokens_list = []

    total = len(df)

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            logger.info(f"Progreso: {idx}/{total} ({(idx/total)*100:.1f}%)")

        start_time = time.time()

        try:
            prompt = crear_prompt(row)
            respuesta_llm, input_tokens, output_tokens = llamar_llm(prompt, config, client)
            label, confidence, reason = parsear_respuesta(respuesta_llm)

            cost_input = (input_tokens / 1_000_000) * config['model_a']['cost_per_1m_input_tokens']
            cost_output = (output_tokens / 1_000_000) * config['model_a']['cost_per_1m_output_tokens']
            costo = cost_input + cost_output

            input_tokens_list.append(input_tokens)
            output_tokens_list.append(output_tokens)
            costos.append(costo)
        except Exception as e:
            logger.error(f"Error en prediccion {idx}: {e}")
            label, confidence, reason = 'NORMAL', 0.5, 'Error en API'
            input_tokens_list.append(0)
            output_tokens_list.append(0)
            costos.append(0.0)

        latencia = (time.time() - start_time) * 1000

        predicciones.append(label)
        confidences.append(confidence)
        reasons.append(reason)
        latencias.append(latencia)

    logger.info(f"Predicciones completadas: {total}")

    df['PRED_LLM'] = predicciones
    df['CONFIDENCE_LLM'] = confidences
    df['REASON_LLM'] = reasons
    df['LATENCY_LLM'] = latencias
    df['COST_LLM'] = costos
    df['INPUT_TOKENS'] = input_tokens_list
    df['OUTPUT_TOKENS'] = output_tokens_list

    return df


def calcular_metricas_llm(df):
    """Calcula metricas de performance del LLM"""
    logger.info("Calculando metricas de LLM...")

    latencia_promedio = df['LATENCY_LLM'].mean()
    latencia_p50 = df['LATENCY_LLM'].quantile(0.5)
    latencia_p95 = df['LATENCY_LLM'].quantile(0.95)

    total_predicciones = len(df)
    predicciones_anomalas = (df['PRED_LLM'] == 'ANOMALO').sum()
    pct_anomalas = (predicciones_anomalas / total_predicciones) * 100

    confidence_promedio = df['CONFIDENCE_LLM'].mean()

    costo_total = df['COST_LLM'].sum()
    costo_promedio = df['COST_LLM'].mean()
    total_input_tokens = df['INPUT_TOKENS'].sum()
    total_output_tokens = df['OUTPUT_TOKENS'].sum()

    logger.info(f"Latencia promedio: {latencia_promedio:.2f}ms")
    logger.info(f"Latencia P50: {latencia_p50:.2f}ms")
    logger.info(f"Latencia P95: {latencia_p95:.2f}ms")
    logger.info(f"Predicciones ANOMALO: {predicciones_anomalas:,} ({pct_anomalas:.2f}%)")
    logger.info(f"Confidence promedio: {confidence_promedio:.3f}")
    logger.info(f"Costo total: ${costo_total:.6f}")
    logger.info(f"Costo promedio por prediccion: ${costo_promedio:.6f}")
    logger.info(f"Total tokens input: {total_input_tokens:,}")
    logger.info(f"Total tokens output: {total_output_tokens:,}")

    metricas = {
        'latencia_promedio_ms': latencia_promedio,
        'latencia_p50_ms': latencia_p50,
        'latencia_p95_ms': latencia_p95,
        'total_predicciones': total_predicciones,
        'predicciones_anomalas': predicciones_anomalas,
        'pct_anomalas': pct_anomalas,
        'confidence_promedio': confidence_promedio,
        'costo_total_usd': costo_total,
        'costo_promedio_usd': costo_promedio,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens
    }

    return metricas


def guardar_predicciones_llm(df, metricas, config):
    """Guarda predicciones y metricas"""
    logger.info("Guardando predicciones LLM...")

    output_dir = BASE_DIR / config['paths']['results']
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_path = output_dir / 'predicciones_llm.csv'
    df.to_csv(pred_path, index=False)
    logger.info(f"Predicciones guardadas: {pred_path}")

    # Convertir numpy int64 a int Python nativo
    metricas_serializable = {k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v
                             for k, v in metricas.items()}

    metricas_path = output_dir / 'metricas_llm.json'
    with open(metricas_path, 'w') as f:
        json.dump(metricas_serializable, f, indent=2)
    logger.info(f"Metricas guardadas: {metricas_path}")


def ejecutar_modelo_a(config_path=None):
    """Ejecuta pipeline completo del Modelo A - LLM"""
    logger.info("="*60)
    logger.info("MODELO A - LLM (OPENAI)")
    logger.info("="*60)

    config = cargar_configuracion(config_path)

    df_test = cargar_datos_test(config)
    df_test = predecir_con_llm(df_test, config)
    metricas = calcular_metricas_llm(df_test)
    guardar_predicciones_llm(df_test, metricas, config)

    logger.info("="*60)
    logger.info("MODELO A COMPLETADO")
    logger.info("="*60)

    return df_test, metricas


if __name__ == "__main__":
    ejecutar_modelo_a()
