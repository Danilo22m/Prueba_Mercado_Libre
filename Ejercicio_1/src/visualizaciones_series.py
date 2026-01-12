import pandas as pd
import numpy as np
import logging
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

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


def cargar_datos_completos_con_predicciones(config):
    """
    Carga datos de test con predicciones de ambos modelos.

    Como LLM solo tiene muestra del 5%, hacemos merge para obtener
    registros donde tengamos ambas predicciones.
    """
    logger.info("Cargando datos completos con predicciones...")

    results_dir = BASE_DIR / config['paths']['results']

    # Cargar predicciones LLM
    llm_path = results_dir / 'predicciones_llm.csv'
    df_llm = pd.read_csv(llm_path)
    logger.info(f"Predicciones LLM: {len(df_llm):,} registros")

    # Cargar predicciones Isolation Forest
    if_path = results_dir / 'predicciones_isolation_forest.csv'
    df_if = pd.read_csv(if_path)
    logger.info(f"Predicciones IF: {len(df_if):,} registros")

    # Merge para obtener registros comunes
    df_merged = df_llm.merge(
        df_if[['ITEM_ID', 'ORD_CLOSED_DT', 'PRICE', 'PRED_IF', 'CONFIDENCE_IF']],
        on=['ITEM_ID', 'ORD_CLOSED_DT', 'PRICE'],
        how='inner',
        suffixes=('', '_IF')
    )

    logger.info(f"Registros con ambas predicciones: {len(df_merged):,}")

    # Convertir fecha a datetime
    df_merged['ORD_CLOSED_DT'] = pd.to_datetime(df_merged['ORD_CLOSED_DT'])

    return df_merged


def seleccionar_productos_interesantes(df, n_productos=3):
    """
    Selecciona productos interesantes para visualizar:
    - Productos con al menos una anomalía verdadera
    - Que tengan suficientes registros
    - Donde los modelos difieran en algunas predicciones
    """
    logger.info(f"Seleccionando {n_productos} productos interesantes para visualizar...")

    # Filtrar productos con anomalías
    productos_con_anomalias = df[df['TRUE_LABEL'] == 'ANOMALO']['ITEM_ID'].unique()

    # Contar registros por producto
    registros_por_producto = df['ITEM_ID'].value_counts()

    # Filtrar productos con al menos 10 registros
    productos_suficientes = registros_por_producto[registros_por_producto >= 10].index

    # Intersección
    candidatos = list(set(productos_con_anomalias) & set(productos_suficientes))

    if len(candidatos) == 0:
        logger.warning("No hay productos con anomalías y suficientes registros. Usando productos al azar.")
        candidatos = registros_por_producto.head(10).index.tolist()

    # Calcular diferencias entre modelos para cada producto
    diferencias = []
    for producto in candidatos:
        df_prod = df[df['ITEM_ID'] == producto]
        # Contar cuántas veces difieren los modelos
        diff_count = (df_prod['PRED_LLM'] != df_prod['PRED_IF']).sum()
        diferencias.append((producto, diff_count, len(df_prod)))

    # Ordenar por diferencias (queremos productos donde los modelos difieran)
    diferencias.sort(key=lambda x: x[1], reverse=True)

    # Seleccionar top N
    productos_seleccionados = [p[0] for p in diferencias[:n_productos]]

    logger.info(f"Productos seleccionados: {productos_seleccionados}")
    for prod, diff, total in diferencias[:n_productos]:
        logger.info(f"  {prod}: {diff}/{total} predicciones diferentes entre modelos")

    return productos_seleccionados


def determinar_categoria_prediccion(row):
    """
    Determina la categoría de predicción basada en TRUE_LABEL y predicciones de ambos modelos.

    Categorías:
    - 'ambos_aciertan': Ambos modelos predicen correctamente
    - 'ambos_fallan': Ambos modelos predicen incorrectamente
    - 'solo_llm_acierta': Solo Modelo A acierta
    - 'solo_if_acierta': Solo Modelo B acierta
    """
    true_label = row['TRUE_LABEL']
    pred_llm = row['PRED_LLM']
    pred_if = row['PRED_IF']

    llm_correcto = (pred_llm == true_label)
    if_correcto = (pred_if == true_label)

    if llm_correcto and if_correcto:
        return 'ambos_aciertan'
    elif not llm_correcto and not if_correcto:
        return 'ambos_fallan'
    elif llm_correcto and not if_correcto:
        return 'solo_llm_acierta'
    else:  # if_correcto and not llm_correcto
        return 'solo_if_acierta'


def graficar_serie_temporal_producto(df_producto, producto_id, config, ax):
    """
    Grafica serie temporal de un producto con predicciones de ambos modelos.

    - Serie de precios
    - Mediana/umbral
    - Puntos coloreados según categoría de predicción
    """
    # Ordenar por fecha
    df_prod = df_producto.sort_values('ORD_CLOSED_DT').copy()

    # Calcular mediana global del producto
    mediana = df_prod['PRICE'].median()

    # Determinar categoría de predicción
    df_prod['categoria'] = df_prod.apply(determinar_categoria_prediccion, axis=1)

    # Colores para cada categoría
    colores = {
        'ambos_aciertan': 'green',
        'ambos_fallan': 'red',
        'solo_llm_acierta': 'orange',
        'solo_if_acierta': 'blue'
    }

    # Labels para leyenda
    labels_dict = {
        'ambos_aciertan': 'Ambos aciertan',
        'ambos_fallan': 'Ambos fallan',
        'solo_llm_acierta': 'Solo LLM acierta',
        'solo_if_acierta': 'Solo IF acierta'
    }

    # Graficar línea de precios
    ax.plot(df_prod['ORD_CLOSED_DT'], df_prod['PRICE'],
            color='gray', alpha=0.3, linewidth=1, label='Serie de precios')

    # Graficar mediana
    ax.axhline(y=mediana, color='purple', linestyle='--',
               linewidth=1.5, label=f'Mediana ({mediana:.2f})')

    # Graficar puntos coloreados por categoría
    for categoria in colores.keys():
        df_cat = df_prod[df_prod['categoria'] == categoria]
        if len(df_cat) > 0:
            ax.scatter(df_cat['ORD_CLOSED_DT'], df_cat['PRICE'],
                      c=colores[categoria], s=100, alpha=0.7,
                      label=labels_dict[categoria], edgecolors='black', linewidth=0.5)

    # Marcar anomalías verdaderas con un círculo
    df_anomalo = df_prod[df_prod['TRUE_LABEL'] == 'ANOMALO']
    if len(df_anomalo) > 0:
        ax.scatter(df_anomalo['ORD_CLOSED_DT'], df_anomalo['PRICE'],
                  facecolors='none', edgecolors='black', s=200, linewidths=2,
                  label='Anomalía verdadera')

    # Configurar gráfico
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio')
    ax.set_title(f'Producto: {producto_id}')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)


def crear_visualizaciones_series_temporales(config_path=None, n_productos=3):
    """
    Crea visualizaciones de series temporales para productos seleccionados.

    Muestra:
    - Serie de precios
    - Mediana/umbral
    - Puntos coloreados según acierto/error de ambos modelos
    """
    logger.info("="*60)
    logger.info("VISUALIZACIONES DE SERIES TEMPORALES")
    logger.info("="*60)

    config = cargar_configuracion(config_path)

    # Cargar datos
    df = cargar_datos_completos_con_predicciones(config)

    # Seleccionar productos interesantes
    productos = seleccionar_productos_interesantes(df, n_productos)

    # Crear figura con subplots
    fig, axes = plt.subplots(n_productos, 1, figsize=(14, 5 * n_productos))

    # Si solo hay 1 producto, axes no es array
    if n_productos == 1:
        axes = [axes]

    # Graficar cada producto
    for i, producto_id in enumerate(productos):
        logger.info(f"Graficando producto {i+1}/{n_productos}: {producto_id}")
        df_producto = df[df['ITEM_ID'] == producto_id]
        graficar_serie_temporal_producto(df_producto, producto_id, config, axes[i])

    plt.tight_layout()

    # Guardar gráfico
    plots_dir = BASE_DIR / config['paths']['plots']
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_path = plots_dir / 'series_temporales_comparacion_modelos.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Gráfico guardado: {plot_path}")

    plt.close()

    # Generar reporte de las categorías
    generar_reporte_categorias(df, config)

    logger.info("="*60)
    logger.info("VISUALIZACIONES COMPLETADAS")
    logger.info("="*60)


def generar_reporte_categorias(df, config):
    """Genera reporte de distribución de categorías de predicción"""
    logger.info("\nGenerando reporte de categorías...")

    df['categoria'] = df.apply(determinar_categoria_prediccion, axis=1)

    categoria_counts = df['categoria'].value_counts()
    total = len(df)

    logger.info("\n" + "="*60)
    logger.info("DISTRIBUCIÓN DE PREDICCIONES (Registros comunes)")
    logger.info("="*60)

    for categoria, count in categoria_counts.items():
        pct = (count / total) * 100
        logger.info(f"{categoria:20s}: {count:5d} ({pct:5.2f}%)")

    logger.info(f"{'TOTAL':20s}: {total:5d} (100.00%)")
    logger.info("="*60)

    # Calcular acuerdos
    acuerdos = categoria_counts.get('ambos_aciertan', 0) + categoria_counts.get('ambos_fallan', 0)
    desacuerdos = categoria_counts.get('solo_llm_acierta', 0) + categoria_counts.get('solo_if_acierta', 0)

    logger.info(f"\nACUERDO entre modelos: {acuerdos:,} ({acuerdos/total*100:.2f}%)")
    logger.info(f"DESACUERDO entre modelos: {desacuerdos:,} ({desacuerdos/total*100:.2f}%)")


if __name__ == "__main__":
    crear_visualizaciones_series_temporales(n_productos=3)
