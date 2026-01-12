import pandas as pd
import numpy as np
import logging
import yaml
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score

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


def cargar_predicciones_para_ab_test(config):
    """
    Carga predicciones de ambos modelos.

    IMPORTANTE: Para A/B test necesitamos predicciones sobre los MISMOS registros.
    Como LLM usa muestra del 5% e IF usa todo el test, vamos a:
    - Cargar ambos datasets
    - Hacer merge para encontrar registros comunes
    - Realizar A/B test solo sobre registros comunes
    """
    logger.info("Cargando predicciones para A/B test...")

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

    # Hacer merge para obtener registros comunes
    logger.info("Haciendo merge para encontrar registros comunes...")
    df_merged = df_llm.merge(
        df_if[['ITEM_ID', 'ORD_CLOSED_DT', 'PRICE', 'PRED_IF', 'CONFIDENCE_IF']],
        on=['ITEM_ID', 'ORD_CLOSED_DT', 'PRICE'],
        how='inner'
    )

    logger.info(f"Registros comunes encontrados: {len(df_merged):,}")

    if len(df_merged) < 100:
        logger.warning(f"ADVERTENCIA: Solo {len(df_merged)} registros comunes. "
                      "Resultados de A/B test pueden no ser confiables.")

    return df_merged


def bootstrap_stratified_sample(df, n_bootstrap=1000, random_state=42):
    """
    Genera muestras bootstrap estratificadas por TRUE_LABEL

    Parameters:
    - df: DataFrame con predicciones
    - n_bootstrap: Número de iteraciones bootstrap
    - random_state: Semilla aleatoria

    Returns:
    - Lista de DataFrames con muestras bootstrap
    """
    logger.info(f"Generando {n_bootstrap} muestras bootstrap estratificadas...")

    np.random.seed(random_state)
    bootstrap_samples = []

    # Separar por TRUE_LABEL para estratificar
    df_normal = df[df['TRUE_LABEL'] == 'NORMAL']
    df_anomalo = df[df['TRUE_LABEL'] == 'ANOMALO']

    logger.info(f"Distribución original - NORMAL: {len(df_normal):,}, ANOMALO: {len(df_anomalo):,}")

    for i in range(n_bootstrap):
        # Muestreo con reemplazo, manteniendo proporciones
        sample_normal = df_normal.sample(n=len(df_normal), replace=True, random_state=random_state + i)
        sample_anomalo = df_anomalo.sample(n=len(df_anomalo), replace=True, random_state=random_state + i + 1000)

        # Combinar y mezclar
        sample_combined = pd.concat([sample_normal, sample_anomalo], ignore_index=True)
        sample_combined = sample_combined.sample(frac=1, random_state=random_state + i).reset_index(drop=True)

        bootstrap_samples.append(sample_combined)

    logger.info(f"Generadas {len(bootstrap_samples)} muestras bootstrap")

    return bootstrap_samples


def calcular_metricas_en_muestra(df, modelo_col, true_col='TRUE_LABEL'):
    """
    Calcula métricas de clasificación en una muestra

    Returns:
    - dict con precision, recall, f1_score
    """
    y_true = (df[true_col] == 'ANOMALO').astype(int)
    y_pred = (df[modelo_col] == 'ANOMALO').astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def ejecutar_bootstrap_ab_test(df, n_bootstrap=1000, random_state=42):
    """
    Ejecuta A/B test usando bootstrap stratified sampling

    Returns:
    - Diccionario con resultados del test
    """
    logger.info("="*60)
    logger.info("EJECUTANDO A/B TEST CON BOOTSTRAP STRATIFIED SAMPLING")
    logger.info("="*60)

    # Generar muestras bootstrap
    bootstrap_samples = bootstrap_stratified_sample(df, n_bootstrap, random_state)

    # Calcular métricas en cada muestra
    metricas_llm_bootstrap = []
    metricas_if_bootstrap = []

    logger.info("Calculando métricas en cada muestra bootstrap...")

    for i, sample in enumerate(bootstrap_samples):
        if (i + 1) % 100 == 0:
            logger.info(f"Procesando muestra {i+1}/{n_bootstrap}...")

        metricas_llm = calcular_metricas_en_muestra(sample, 'PRED_LLM')
        metricas_if = calcular_metricas_en_muestra(sample, 'PRED_IF')

        metricas_llm_bootstrap.append(metricas_llm)
        metricas_if_bootstrap.append(metricas_if)

    # Convertir a DataFrames para análisis
    df_llm_bootstrap = pd.DataFrame(metricas_llm_bootstrap)
    df_if_bootstrap = pd.DataFrame(metricas_if_bootstrap)

    # Calcular diferencias
    df_diff = pd.DataFrame({
        'precision_diff': df_llm_bootstrap['precision'] - df_if_bootstrap['precision'],
        'recall_diff': df_llm_bootstrap['recall'] - df_if_bootstrap['recall'],
        'f1_diff': df_llm_bootstrap['f1_score'] - df_if_bootstrap['f1_score']
    })

    # Calcular p-values (prueba bilateral)
    # H0: No hay diferencia entre modelos
    # H1: Hay diferencia significativa
    p_value_precision = (df_diff['precision_diff'] <= 0).sum() / n_bootstrap
    p_value_recall = (df_diff['recall_diff'] <= 0).sum() / n_bootstrap
    p_value_f1 = (df_diff['f1_diff'] <= 0).sum() / n_bootstrap

    # Ajustar para prueba bilateral
    p_value_precision = min(p_value_precision, 1 - p_value_precision) * 2
    p_value_recall = min(p_value_recall, 1 - p_value_recall) * 2
    p_value_f1 = min(p_value_f1, 1 - p_value_f1) * 2

    # Calcular intervalos de confianza (95%)
    ci_precision = (
        df_diff['precision_diff'].quantile(0.025),
        df_diff['precision_diff'].quantile(0.975)
    )
    ci_recall = (
        df_diff['recall_diff'].quantile(0.025),
        df_diff['recall_diff'].quantile(0.975)
    )
    ci_f1 = (
        df_diff['f1_diff'].quantile(0.025),
        df_diff['f1_diff'].quantile(0.975)
    )

    resultados = {
        'n_bootstrap': n_bootstrap,
        'n_registros_comunes': len(df),
        'metricas_llm': {
            'precision_mean': df_llm_bootstrap['precision'].mean(),
            'precision_std': df_llm_bootstrap['precision'].std(),
            'recall_mean': df_llm_bootstrap['recall'].mean(),
            'recall_std': df_llm_bootstrap['recall'].std(),
            'f1_mean': df_llm_bootstrap['f1_score'].mean(),
            'f1_std': df_llm_bootstrap['f1_score'].std()
        },
        'metricas_if': {
            'precision_mean': df_if_bootstrap['precision'].mean(),
            'precision_std': df_if_bootstrap['precision'].std(),
            'recall_mean': df_if_bootstrap['recall'].mean(),
            'recall_std': df_if_bootstrap['recall'].std(),
            'f1_mean': df_if_bootstrap['f1_score'].mean(),
            'f1_std': df_if_bootstrap['f1_score'].std()
        },
        'diferencias': {
            'precision_diff_mean': df_diff['precision_diff'].mean(),
            'precision_ci': ci_precision,
            'precision_p_value': p_value_precision,
            'recall_diff_mean': df_diff['recall_diff'].mean(),
            'recall_ci': ci_recall,
            'recall_p_value': p_value_recall,
            'f1_diff_mean': df_diff['f1_diff'].mean(),
            'f1_ci': ci_f1,
            'f1_p_value': p_value_f1
        },
        'significancia': {
            'alpha': 0.05,
            'precision_significativo': p_value_precision < 0.05,
            'recall_significativo': p_value_recall < 0.05,
            'f1_significativo': p_value_f1 < 0.05
        }
    }

    # Guardar distribuciones para gráficos
    resultados['distribuciones'] = {
        'llm_bootstrap': df_llm_bootstrap,
        'if_bootstrap': df_if_bootstrap,
        'diferencias': df_diff
    }

    return resultados


def mostrar_resultados_ab_test(resultados):
    """Muestra resultados del A/B test en consola"""
    logger.info("\n" + "="*60)
    logger.info("RESULTADOS A/B TEST")
    logger.info("="*60)

    logger.info(f"\nNúmero de iteraciones bootstrap: {resultados['n_bootstrap']:,}")
    logger.info(f"Registros comunes evaluados: {resultados['n_registros_comunes']:,}")

    logger.info("\n" + "-"*60)
    logger.info("MODELO A (LLM) - Métricas Bootstrap")
    logger.info("-"*60)
    llm = resultados['metricas_llm']
    logger.info(f"Precision: {llm['precision_mean']:.4f} ± {llm['precision_std']:.4f}")
    logger.info(f"Recall:    {llm['recall_mean']:.4f} ± {llm['recall_std']:.4f}")
    logger.info(f"F1-Score:  {llm['f1_mean']:.4f} ± {llm['f1_std']:.4f}")

    logger.info("\n" + "-"*60)
    logger.info("MODELO B (ISOLATION FOREST) - Métricas Bootstrap")
    logger.info("-"*60)
    if_m = resultados['metricas_if']
    logger.info(f"Precision: {if_m['precision_mean']:.4f} ± {if_m['precision_std']:.4f}")
    logger.info(f"Recall:    {if_m['recall_mean']:.4f} ± {if_m['recall_std']:.4f}")
    logger.info(f"F1-Score:  {if_m['f1_mean']:.4f} ± {if_m['f1_std']:.4f}")

    logger.info("\n" + "="*60)
    logger.info("DIFERENCIAS (Modelo A - Modelo B)")
    logger.info("="*60)
    diff = resultados['diferencias']
    sig = resultados['significancia']

    logger.info(f"\nPRECISION:")
    logger.info(f"  Diferencia media: {diff['precision_diff_mean']:+.4f}")
    logger.info(f"  IC 95%: [{diff['precision_ci'][0]:+.4f}, {diff['precision_ci'][1]:+.4f}]")
    logger.info(f"  P-value: {diff['precision_p_value']:.4f}")
    logger.info(f"  {'✓ SIGNIFICATIVO' if sig['precision_significativo'] else 'NO significativo'} (α=0.05)")

    logger.info(f"\nRECALL:")
    logger.info(f"  Diferencia media: {diff['recall_diff_mean']:+.4f}")
    logger.info(f"  IC 95%: [{diff['recall_ci'][0]:+.4f}, {diff['recall_ci'][1]:+.4f}]")
    logger.info(f"  P-value: {diff['recall_p_value']:.4f}")
    logger.info(f"  {'✓ SIGNIFICATIVO' if sig['recall_significativo'] else 'NO significativo'} (α=0.05)")

    logger.info(f"\nF1-SCORE:")
    logger.info(f"  Diferencia media: {diff['f1_diff_mean']:+.4f}")
    logger.info(f"  IC 95%: [{diff['f1_ci'][0]:+.4f}, {diff['f1_ci'][1]:+.4f}]")
    logger.info(f"  P-value: {diff['f1_p_value']:.4f}")
    logger.info(f"  {'✓ SIGNIFICATIVO' if sig['f1_significativo'] else 'NO significativo'} (α=0.05)")

    logger.info("\n" + "="*60)
    logger.info("CONCLUSIÓN")
    logger.info("="*60)

    if sig['f1_significativo']:
        if diff['f1_diff_mean'] > 0:
            logger.info("✓ Modelo A (LLM) es SIGNIFICATIVAMENTE MEJOR que Modelo B (IF)")
        else:
            logger.info("✓ Modelo B (IF) es SIGNIFICATIVAMENTE MEJOR que Modelo A (LLM)")
    else:
        logger.info("⚠ NO hay diferencia significativa entre los modelos (α=0.05)")
        logger.info("  Ambos modelos tienen rendimiento estadísticamente similar")

    logger.info("="*60)


def graficar_distribuciones_bootstrap(resultados, config):
    """Genera gráficos de distribuciones bootstrap"""
    logger.info("Generando gráficos de distribuciones bootstrap...")

    df_llm = resultados['distribuciones']['llm_bootstrap']
    df_if = resultados['distribuciones']['if_bootstrap']
    df_diff = resultados['distribuciones']['diferencias']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    metricas = ['precision', 'recall', 'f1_score']
    metricas_labels = ['Precision', 'Recall', 'F1-Score']

    # Fila 1: Distribuciones individuales
    for i, (metrica, label) in enumerate(zip(metricas, metricas_labels)):
        ax = axes[0, i]

        if metrica == 'f1_score':
            col_llm = 'f1_score'
            col_if = 'f1_score'
            col_diff = 'f1_diff'
        else:
            col_llm = metrica
            col_if = metrica
            col_diff = f'{metrica}_diff'

        ax.hist(df_llm[col_llm], bins=30, alpha=0.6, label='Modelo A (LLM)', color='blue')
        ax.hist(df_if[col_if], bins=30, alpha=0.6, label='Modelo B (IF)', color='green')
        ax.axvline(df_llm[col_llm].mean(), color='blue', linestyle='--', linewidth=2)
        ax.axvline(df_if[col_if].mean(), color='green', linestyle='--', linewidth=2)
        ax.set_xlabel(label)
        ax.set_ylabel('Frecuencia')
        ax.set_title(f'Distribución Bootstrap - {label}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Fila 2: Diferencias
    for i, (metrica, label) in enumerate(zip(['precision', 'recall', 'f1'], metricas_labels)):
        ax = axes[1, i]

        col_diff = f'{metrica}_diff'
        ax.hist(df_diff[col_diff], bins=30, alpha=0.7, color='purple')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Sin diferencia')
        ax.axvline(df_diff[col_diff].mean(), color='black', linestyle='-', linewidth=2, label='Diferencia media')

        # Intervalos de confianza
        ci_low, ci_high = resultados['diferencias'][f'{metrica}_ci']
        ax.axvline(ci_low, color='orange', linestyle=':', linewidth=1.5, label='IC 95%')
        ax.axvline(ci_high, color='orange', linestyle=':', linewidth=1.5)

        ax.set_xlabel(f'Diferencia {label} (A - B)')
        ax.set_ylabel('Frecuencia')
        ax.set_title(f'Distribución de Diferencias - {label}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plots_dir = BASE_DIR / config['paths']['plots']
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_path = plots_dir / 'ab_test_bootstrap_distributions.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Gráfico guardado: {plot_path}")

    plt.close()


def guardar_resultados_ab_test(resultados, config):
    """Guarda resultados del A/B test"""
    logger.info("Guardando resultados del A/B test...")

    results_dir = BASE_DIR / config['paths']['results']

    # Preparar resultados para JSON (sin DataFrames)
    # Convertir todos los valores a tipos nativos de Python
    resultados_json = {
        'n_bootstrap': int(resultados['n_bootstrap']),
        'n_registros_comunes': int(resultados['n_registros_comunes']),
        'metricas_llm': {k: float(v) for k, v in resultados['metricas_llm'].items()},
        'metricas_if': {k: float(v) for k, v in resultados['metricas_if'].items()},
        'diferencias': {
            'precision_diff_mean': float(resultados['diferencias']['precision_diff_mean']),
            'precision_ci': [float(x) for x in resultados['diferencias']['precision_ci']],
            'precision_p_value': float(resultados['diferencias']['precision_p_value']),
            'recall_diff_mean': float(resultados['diferencias']['recall_diff_mean']),
            'recall_ci': [float(x) for x in resultados['diferencias']['recall_ci']],
            'recall_p_value': float(resultados['diferencias']['recall_p_value']),
            'f1_diff_mean': float(resultados['diferencias']['f1_diff_mean']),
            'f1_ci': [float(x) for x in resultados['diferencias']['f1_ci']],
            'f1_p_value': float(resultados['diferencias']['f1_p_value'])
        },
        'significancia': {
            'alpha': float(resultados['significancia']['alpha']),
            'precision_significativo': bool(resultados['significancia']['precision_significativo']),
            'recall_significativo': bool(resultados['significancia']['recall_significativo']),
            'f1_significativo': bool(resultados['significancia']['f1_significativo'])
        }
    }

    # Guardar JSON
    ab_test_path = results_dir / 'ab_test_results.json'
    with open(ab_test_path, 'w') as f:
        json.dump(resultados_json, f, indent=2)
    logger.info(f"Resultados guardados: {ab_test_path}")


def ejecutar_ab_test(config_path=None, n_bootstrap=1000):
    """Ejecuta pipeline completo de A/B testing"""
    logger.info("="*60)
    logger.info("A/B TESTING - MODELO A (LLM) VS MODELO B (ISOLATION FOREST)")
    logger.info("="*60)

    config = cargar_configuracion(config_path)

    # Cargar predicciones comunes
    df_merged = cargar_predicciones_para_ab_test(config)

    # Ejecutar bootstrap A/B test
    resultados = ejecutar_bootstrap_ab_test(df_merged, n_bootstrap=n_bootstrap)

    # Mostrar resultados
    mostrar_resultados_ab_test(resultados)

    # Generar gráficos
    graficar_distribuciones_bootstrap(resultados, config)

    # Guardar resultados
    guardar_resultados_ab_test(resultados, config)

    logger.info("="*60)
    logger.info("A/B TEST COMPLETADO")
    logger.info("="*60)

    return resultados


if __name__ == "__main__":
    ejecutar_ab_test(n_bootstrap=1000)
