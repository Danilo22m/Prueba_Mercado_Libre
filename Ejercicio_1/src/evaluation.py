import pandas as pd
import numpy as np
import logging
import yaml
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, auc,
    confusion_matrix, classification_report
)

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


def cargar_predicciones(config):
    """Carga predicciones de ambos modelos"""
    logger.info("Cargando predicciones de ambos modelos...")

    results_dir = BASE_DIR / config['paths']['results']

    # Cargar predicciones LLM
    llm_path = results_dir / 'predicciones_llm.csv'
    if not llm_path.exists():
        raise FileNotFoundError(f"No se encontraron predicciones LLM en {llm_path}")

    df_llm = pd.read_csv(llm_path)
    logger.info(f"Predicciones LLM cargadas: {len(df_llm):,} registros")

    # Cargar predicciones Isolation Forest
    if_path = results_dir / 'predicciones_isolation_forest.csv'
    if not if_path.exists():
        raise FileNotFoundError(f"No se encontraron predicciones Isolation Forest en {if_path}")

    df_if = pd.read_csv(if_path)
    logger.info(f"Predicciones Isolation Forest cargadas: {len(df_if):,} registros")

    return df_llm, df_if


def preparar_datasets_independientes(df_llm, df_if):
    """Prepara datasets independientes para evaluación de cada modelo"""
    logger.info("Preparando datasets independientes para evaluación...")
    logger.info(f"Modelo A (LLM) será evaluado en: {len(df_llm):,} registros (muestra 5%)")
    logger.info(f"Modelo B (Isolation Forest) será evaluado en: {len(df_if):,} registros (test completo)")

    return df_llm.copy(), df_if.copy()


def calcular_metricas_clasificacion(y_true, y_pred, model_name):
    """Calcula metricas de clasificacion"""
    logger.info(f"Calculando metricas para {model_name}...")

    # Convertir a binario: ANOMALO=1, NORMAL=0
    y_true_bin = (y_true == 'ANOMALO').astype(int)
    y_pred_bin = (y_pred == 'ANOMALO').astype(int)

    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    tn, fp, fn, tp = cm.ravel()

    logger.info(f"{model_name} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    logger.info(f"{model_name} - TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    metricas = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'confusion_matrix': cm.tolist()
    }

    return metricas


def calcular_pr_auc(y_true, confidences, model_name):
    """Calcula Precision-Recall AUC"""
    logger.info(f"Calculando PR-AUC para {model_name}...")

    y_true_bin = (y_true == 'ANOMALO').astype(int)

    # Calcular curva PR
    precision, recall, thresholds = precision_recall_curve(y_true_bin, confidences)

    # Calcular AUC
    pr_auc = auc(recall, precision)

    logger.info(f"{model_name} - PR-AUC: {pr_auc:.4f}")

    return pr_auc, precision, recall, thresholds


def crear_tabla_comparativa(metricas_llm, metricas_if, pr_auc_llm, pr_auc_if):
    """Crea tabla comparativa de metricas (evaluadas sobre datasets diferentes)"""
    logger.info("Creando tabla comparativa...")

    comparacion = {
        'Modelo A (LLM) [muestra 5%]': {
            'Precision': metricas_llm['precision'],
            'Recall': metricas_llm['recall'],
            'F1-Score': metricas_llm['f1_score'],
            'PR-AUC': pr_auc_llm,
            'TP': metricas_llm['true_positives'],
            'TN': metricas_llm['true_negatives'],
            'FP': metricas_llm['false_positives'],
            'FN': metricas_llm['false_negatives']
        },
        'Modelo B (IF) [test completo]': {
            'Precision': metricas_if['precision'],
            'Recall': metricas_if['recall'],
            'F1-Score': metricas_if['f1_score'],
            'PR-AUC': pr_auc_if,
            'TP': metricas_if['true_positives'],
            'TN': metricas_if['true_negatives'],
            'FP': metricas_if['false_positives'],
            'FN': metricas_if['false_negatives']
        }
    }

    df_comparacion = pd.DataFrame(comparacion).T

    logger.info("\n" + "="*60)
    logger.info("TABLA COMPARATIVA DE MODELOS")
    logger.info("(IMPORTANTE: Evaluados sobre diferentes datasets)")
    logger.info("="*60)
    logger.info("\n" + df_comparacion.to_string())
    logger.info("="*60)

    return df_comparacion


def graficar_confusion_matrices(metricas_llm, metricas_if, config):
    """Grafica matrices de confusion"""
    logger.info("Graficando matrices de confusion...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Modelo A (LLM)
    cm_llm = np.array(metricas_llm['confusion_matrix'])
    sns.heatmap(cm_llm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['NORMAL', 'ANOMALO'],
                yticklabels=['NORMAL', 'ANOMALO'])
    axes[0].set_title('Modelo A (LLM)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Modelo B (Isolation Forest)
    cm_if = np.array(metricas_if['confusion_matrix'])
    sns.heatmap(cm_if, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['NORMAL', 'ANOMALO'],
                yticklabels=['NORMAL', 'ANOMALO'])
    axes[1].set_title('Modelo B (Isolation Forest)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()

    plots_dir = BASE_DIR / config['paths']['plots']
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_path = plots_dir / 'confusion_matrices.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Grafico guardado: {plot_path}")

    plt.close()


def graficar_curvas_pr(pr_llm, pr_if, pr_auc_llm, pr_auc_if, config):
    """Grafica curvas Precision-Recall"""
    logger.info("Graficando curvas PR...")

    plt.figure(figsize=(10, 6))

    precision_llm, recall_llm, _ = pr_llm
    precision_if, recall_if, _ = pr_if

    plt.plot(recall_llm, precision_llm, label=f'Modelo A (LLM) - AUC={pr_auc_llm:.4f}', linewidth=2)
    plt.plot(recall_if, precision_if, label=f'Modelo B (Isolation Forest) - AUC={pr_auc_if:.4f}', linewidth=2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curvas Precision-Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plots_dir = BASE_DIR / config['paths']['plots']
    plot_path = plots_dir / 'precision_recall_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Grafico guardado: {plot_path}")

    plt.close()


def guardar_resultados(df_comparacion, metricas_llm, metricas_if, pr_auc_llm, pr_auc_if, config):
    """Guarda resultados de evaluacion"""
    logger.info("Guardando resultados de evaluacion...")

    results_dir = BASE_DIR / config['paths']['results']

    # Guardar tabla comparativa
    comp_path = results_dir / 'comparacion_modelos.csv'
    df_comparacion.to_csv(comp_path)
    logger.info(f"Tabla comparativa guardada: {comp_path}")

    # Guardar metricas completas
    eval_results = {
        'modelo_a_llm': {
            **metricas_llm,
            'pr_auc': pr_auc_llm
        },
        'modelo_b_isolation_forest': {
            **metricas_if,
            'pr_auc': pr_auc_if
        }
    }

    eval_path = results_dir / 'evaluacion_completa.json'
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"Evaluacion completa guardada: {eval_path}")


def ejecutar_evaluacion(config_path=None):
    """Ejecuta pipeline completo de evaluacion - Métricas independientes por modelo"""
    logger.info("="*60)
    logger.info("EVALUACION COMPARATIVA DE MODELOS")
    logger.info("="*60)

    config = cargar_configuracion(config_path)

    # Cargar predicciones
    df_llm, df_if = cargar_predicciones(config)

    # Preparar datasets independientes (sin merge)
    df_llm_eval, df_if_eval = preparar_datasets_independientes(df_llm, df_if)

    logger.info("\n" + "="*60)
    logger.info("EVALUANDO MODELO A (LLM) - Muestra 5%")
    logger.info("="*60)

    # Calcular metricas de clasificacion para LLM
    metricas_llm = calcular_metricas_clasificacion(
        df_llm_eval['TRUE_LABEL'],
        df_llm_eval['PRED_LLM'],
        'Modelo A (LLM)'
    )

    # Calcular PR-AUC para LLM
    pr_auc_llm, precision_llm, recall_llm, thresholds_llm = calcular_pr_auc(
        df_llm_eval['TRUE_LABEL'],
        df_llm_eval['CONFIDENCE_LLM'],
        'Modelo A (LLM)'
    )

    logger.info("\n" + "="*60)
    logger.info("EVALUANDO MODELO B (ISOLATION FOREST) - Test Completo")
    logger.info("="*60)

    # Calcular metricas de clasificacion para Isolation Forest
    metricas_if = calcular_metricas_clasificacion(
        df_if_eval['TRUE_LABEL'],
        df_if_eval['PRED_IF'],
        'Modelo B (Isolation Forest)'
    )

    # Calcular PR-AUC para Isolation Forest
    pr_auc_if, precision_if, recall_if, thresholds_if = calcular_pr_auc(
        df_if_eval['TRUE_LABEL'],
        df_if_eval['CONFIDENCE_IF'],
        'Modelo B (Isolation Forest)'
    )

    # Crear tabla comparativa
    logger.info("\n" + "="*60)
    logger.info("NOTA: Métricas calculadas sobre datasets diferentes")
    logger.info(f"- Modelo A (LLM): {len(df_llm_eval):,} registros (muestra 5%)")
    logger.info(f"- Modelo B (IF): {len(df_if_eval):,} registros (test completo)")
    logger.info("="*60)

    df_comparacion = crear_tabla_comparativa(metricas_llm, metricas_if, pr_auc_llm, pr_auc_if)

    # Generar graficos
    graficar_confusion_matrices(metricas_llm, metricas_if, config)
    graficar_curvas_pr(
        (precision_llm, recall_llm, thresholds_llm),
        (precision_if, recall_if, thresholds_if),
        pr_auc_llm, pr_auc_if,
        config
    )

    # Guardar resultados
    guardar_resultados(df_comparacion, metricas_llm, metricas_if, pr_auc_llm, pr_auc_if, config)

    logger.info("="*60)
    logger.info("EVALUACION COMPLETADA")
    logger.info("="*60)

    return df_comparacion, metricas_llm, metricas_if


if __name__ == "__main__":
    ejecutar_evaluacion()
