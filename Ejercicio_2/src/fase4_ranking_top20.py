#!/usr/bin/env python3
"""
FASE 4: Ranking y Análisis Top-20

Este módulo genera un ranking de las URLs más influyentes y analiza
POR QUÉ son influyentes mediante:
1. Ranking multi-métrica (PageRank como principal)
2. Clasificación de roles (Autoridad, Hub, Puente, Propagador, Multi-Rol)
3. Análisis comparativo entre Top-20
4. Cálculo de correlaciones entre métricas
5. Análisis de concentración de influencia

Autor: Danilo Melo
Fecha: 2026-01-11
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import json

# Configurar logging
logger = logging.getLogger(__name__)


def cargar_configuracion(config_path: Optional[str] = None) -> dict:
    """
    Carga la configuración desde archivo YAML.

    Args:
        config_path: Ruta al archivo de configuración (None usa default)

    Returns:
        dict: Configuración cargada
    """
    if config_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        config_path = base_dir / 'config' / 'config.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def cargar_metricas(csv_path: str) -> pd.DataFrame:
    """
    Carga las métricas de centralidad desde CSV.

    Args:
        csv_path: Ruta al archivo CSV con métricas

    Returns:
        pd.DataFrame: DataFrame con todas las métricas
    """
    logger.info(f"Cargando métricas desde: {csv_path}")

    df = pd.read_csv(csv_path)

    logger.info(f"Métricas cargadas: {len(df):,} nodos")

    return df


def calcular_ranking(df: pd.DataFrame, metrica_principal: str = 'pagerank') -> pd.DataFrame:
    """
    Ordena el DataFrame por la métrica principal y asigna ranking.

    Args:
        df: DataFrame con métricas
        metrica_principal: Nombre de la columna para ranking principal

    Returns:
        pd.DataFrame: DataFrame ordenado con columna 'ranking'
    """
    logger.info(f"Calculando ranking por: {metrica_principal}")

    # Ordenar descendente por métrica principal
    df_ranked = df.sort_values(by=metrica_principal, ascending=False).reset_index(drop=True)

    # Asignar ranking (1-indexed)
    df_ranked['ranking'] = range(1, len(df_ranked) + 1)

    logger.info(f"Ranking calculado para {len(df_ranked):,} nodos")

    return df_ranked


def calcular_percentiles(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calcula percentiles para todas las métricas.

    Args:
        df: DataFrame con métricas

    Returns:
        Dict con percentiles por métrica
    """
    metricas = ['pagerank', 'in_degree', 'out_degree', 'hits_authority',
                'hits_hub', 'betweenness', 'closeness']

    percentiles = {}

    for metrica in metricas:
        percentiles[metrica] = {
            'p50': df[metrica].quantile(0.50),
            'p75': df[metrica].quantile(0.75),
            'p90': df[metrica].quantile(0.90),
            'p95': df[metrica].quantile(0.95),
            'p99': df[metrica].quantile(0.99)
        }

    return percentiles


def clasificar_rol_nodo(row: pd.Series, percentiles: Dict[str, Dict[str, float]],
                        umbral_percentil: int = 90) -> Dict[str, Any]:
    """
    Clasifica el rol de un nodo según sus métricas.

    Roles posibles:
    - Autoridad Pura: Alta authority + alto PageRank + alto in-degree
    - Hub: Alto hub score + alto out-degree
    - Puente: Alto betweenness (independiente de otros)
    - Propagador: Alta closeness
    - Multi-Rol: Cumple múltiples roles
    - Estándar: No destaca en ninguna métrica

    Args:
        row: Fila del DataFrame con métricas del nodo
        percentiles: Percentiles calculados para comparación
        umbral_percentil: Percentil para considerar "alto" (default: 90)

    Returns:
        Dict con rol_principal, roles_secundarios, y razones
    """
    umbral = f'p{umbral_percentil}'

    # Evaluar cada rol
    roles = []
    razones = []

    # 1. Autoridad Pura
    es_authority = (
        row['hits_authority'] >= percentiles['hits_authority'][umbral] and
        row['pagerank'] >= percentiles['pagerank'][umbral] and
        row['in_degree'] >= percentiles['in_degree'][umbral]
    )
    if es_authority:
        roles.append('Autoridad')
        razones.append(f"Alta authority ({row['hits_authority']:.6f}), PageRank ({row['pagerank']:.6f}), in-degree ({row['in_degree']})")

    # 2. Hub
    es_hub = (
        row['hits_hub'] >= percentiles['hits_hub'][umbral] and
        row['out_degree'] >= percentiles['out_degree'][umbral]
    )
    if es_hub:
        roles.append('Hub')
        razones.append(f"Alto hub score ({row['hits_hub']:.6f}), out-degree ({row['out_degree']})")

    # 3. Puente
    es_puente = row['betweenness'] >= percentiles['betweenness'][umbral]
    if es_puente:
        roles.append('Puente')
        razones.append(f"Alto betweenness ({row['betweenness']:.6f}), conecta comunidades")

    # 4. Propagador
    es_propagador = row['closeness'] >= percentiles['closeness'][umbral]
    if es_propagador:
        roles.append('Propagador')
        razones.append(f"Alta closeness ({row['closeness']:.6f}), alcance rápido")

    # Determinar rol principal
    if len(roles) == 0:
        rol_principal = 'Estándar'
        roles_secundarios = []
        razon_principal = f"Métricas en rango promedio (PageRank: {row['pagerank']:.6f})"
    elif len(roles) == 1:
        rol_principal = roles[0]
        roles_secundarios = []
        razon_principal = razones[0]
    else:
        rol_principal = 'Multi-Rol'
        roles_secundarios = roles
        razon_principal = f"Combina {len(roles)} roles: {', '.join(roles)}"

    return {
        'rol_principal': rol_principal,
        'roles_secundarios': roles_secundarios if len(roles) > 1 else [],
        'razon_principal': razon_principal,
        'razones_detalladas': razones
    }


def generar_explicacion_nodo(row: pd.Series, df_completo: pd.DataFrame,
                             percentiles: Dict[str, Dict[str, float]]) -> str:
    """
    Genera explicación en lenguaje natural de por qué el nodo es influyente.

    Args:
        row: Fila con métricas del nodo
        df_completo: DataFrame completo para comparaciones
        percentiles: Percentiles de las métricas

    Returns:
        str: Explicación en lenguaje natural
    """
    explicacion_partes = []

    # Comparar PageRank con promedio
    pagerank_promedio = df_completo['pagerank'].mean()
    pagerank_ratio = row['pagerank'] / pagerank_promedio if pagerank_promedio > 0 else 0

    if pagerank_ratio > 10:
        explicacion_partes.append(f"PageRank {pagerank_ratio:.1f}x superior al promedio (excepcional)")
    elif pagerank_ratio > 5:
        explicacion_partes.append(f"PageRank {pagerank_ratio:.1f}x superior al promedio (muy alto)")
    elif pagerank_ratio > 2:
        explicacion_partes.append(f"PageRank {pagerank_ratio:.1f}x superior al promedio (alto)")

    # Analizar in-degree
    if row['in_degree'] >= percentiles['in_degree']['p95']:
        explicacion_partes.append(f"{row['in_degree']} enlaces entrantes (top 5%)")
    elif row['in_degree'] >= percentiles['in_degree']['p90']:
        explicacion_partes.append(f"{row['in_degree']} enlaces entrantes (top 10%)")

    # Analizar authority vs hub
    if row['hits_authority'] > row['hits_hub'] * 2:
        explicacion_partes.append("Principalmente destino de navegación (authority > hub)")
    elif row['hits_hub'] > row['hits_authority'] * 2:
        explicacion_partes.append("Principalmente página de referencia (hub > authority)")
    elif row['hits_authority'] >= percentiles['hits_authority']['p90'] and row['hits_hub'] >= percentiles['hits_hub']['p90']:
        explicacion_partes.append("Equilibrio entre destino y referencia")

    # Analizar betweenness si es significativo
    if row['betweenness'] >= percentiles['betweenness']['p95']:
        explicacion_partes.append("Nodo puente crítico (top 5% betweenness)")

    # Analizar closeness
    if row['closeness'] >= percentiles['closeness']['p90']:
        explicacion_partes.append("Alta capacidad de propagación rápida")

    return ". ".join(explicacion_partes) + "."


def analizar_top_n(df_ranked: pd.DataFrame, n: int,
                   percentiles: Dict[str, Dict[str, float]],
                   umbral_percentil: int = 90) -> Dict[str, Any]:
    """
    Análisis detallado de los Top-N nodos.

    Args:
        df_ranked: DataFrame con ranking ya calculado
        n: Número de nodos top a analizar
        percentiles: Percentiles de métricas
        umbral_percentil: Umbral para clasificación de roles

    Returns:
        Dict con análisis completo
    """
    logger.info(f"Analizando Top-{n} nodos...")

    # Seleccionar top-N
    df_top = df_ranked.head(n).copy()

    # Clasificar roles
    clasificaciones = []
    for idx, row in df_top.iterrows():
        clasificacion = clasificar_rol_nodo(row, percentiles, umbral_percentil)
        clasificaciones.append(clasificacion)

    df_top['rol_principal'] = [c['rol_principal'] for c in clasificaciones]
    df_top['roles_secundarios'] = [c['roles_secundarios'] for c in clasificaciones]
    df_top['explicacion_rol'] = [c['razon_principal'] for c in clasificaciones]

    # Generar explicación detallada
    df_top['explicacion_completa'] = df_top.apply(
        lambda row: generar_explicacion_nodo(row, df_ranked, percentiles),
        axis=1
    )

    # Análisis de diversidad de roles
    conteo_roles = df_top['rol_principal'].value_counts().to_dict()

    # Calcular concentración de PageRank
    pagerank_total = df_ranked['pagerank'].sum()
    pagerank_top5 = df_top.head(5)['pagerank'].sum()
    pagerank_top10 = df_top.head(10)['pagerank'].sum()
    pagerank_top_n = df_top['pagerank'].sum()

    concentracion = {
        'top_5': float(pagerank_top5 / pagerank_total) if pagerank_total > 0 else 0,
        'top_10': float(pagerank_top10 / pagerank_total) if pagerank_total > 0 else 0,
        f'top_{n}': float(pagerank_top_n / pagerank_total) if pagerank_total > 0 else 0
    }

    logger.info(f"  Diversidad de roles en Top-{n}: {conteo_roles}")
    logger.info(f"  Concentración PageRank Top-5: {concentracion['top_5']*100:.2f}%")
    logger.info(f"  Concentración PageRank Top-10: {concentracion['top_10']*100:.2f}%")
    logger.info(f"  Concentración PageRank Top-{n}: {concentracion[f'top_{n}']*100:.2f}%")

    # Estadísticas comparativas Top-N vs resto
    df_resto = df_ranked.iloc[n:]

    comparacion = {
        'pagerank': {
            'promedio_top': float(df_top['pagerank'].mean()),
            'promedio_resto': float(df_resto['pagerank'].mean()),
            'ratio': float(df_top['pagerank'].mean() / df_resto['pagerank'].mean()) if df_resto['pagerank'].mean() > 0 else 0
        },
        'in_degree': {
            'promedio_top': float(df_top['in_degree'].mean()),
            'promedio_resto': float(df_resto['in_degree'].mean()),
            'ratio': float(df_top['in_degree'].mean() / df_resto['in_degree'].mean()) if df_resto['in_degree'].mean() > 0 else 0
        },
        'betweenness': {
            'promedio_top': float(df_top['betweenness'].mean()),
            'promedio_resto': float(df_resto['betweenness'].mean()),
            'ratio': float(df_top['betweenness'].mean() / df_resto['betweenness'].mean()) if df_resto['betweenness'].mean() > 0 else 0
        }
    }

    return {
        'df_top': df_top,
        'diversidad_roles': conteo_roles,
        'concentracion_pagerank': concentracion,
        'comparacion_top_vs_resto': comparacion
    }


def calcular_correlaciones(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calcula correlaciones entre métricas principales.

    Args:
        df: DataFrame con métricas

    Returns:
        Dict con correlaciones relevantes
    """
    logger.info("Calculando correlaciones entre métricas...")

    correlaciones = {
        'pagerank_vs_in_degree': float(df['pagerank'].corr(df['in_degree'])),
        'pagerank_vs_authority': float(df['pagerank'].corr(df['hits_authority'])),
        'pagerank_vs_betweenness': float(df['pagerank'].corr(df['betweenness'])),
        'hub_vs_out_degree': float(df['hits_hub'].corr(df['out_degree'])),
        'authority_vs_in_degree': float(df['hits_authority'].corr(df['in_degree'])),
        'betweenness_vs_closeness': float(df['betweenness'].corr(df['closeness']))
    }

    logger.info("  Correlaciones principales:")
    for nombre, valor in correlaciones.items():
        estado = "FUERTE" if abs(valor) > 0.7 else "MODERADA" if abs(valor) > 0.4 else "DÉBIL"
        logger.info(f"    {nombre}: {valor:.4f} ({estado})")

    return correlaciones


def guardar_ranking_csv(df_top: pd.DataFrame, output_path: str) -> None:
    """
    Guarda el ranking Top-N en formato CSV.

    Args:
        df_top: DataFrame con Top-N nodos
        output_path: Ruta de salida
    """
    logger.info(f"Guardando ranking CSV en: {output_path}")

    # Seleccionar columnas relevantes
    columnas = [
        'ranking', 'nodo', 'pagerank', 'in_degree', 'out_degree',
        'hits_authority', 'hits_hub', 'betweenness', 'closeness',
        'rol_principal', 'explicacion_rol', 'explicacion_completa'
    ]

    df_export = df_top[columnas].copy()

    # Formatear columnas numéricas para mejor legibilidad
    df_export['pagerank'] = df_export['pagerank'].map(lambda x: f"{x:.6f}")
    df_export['hits_authority'] = df_export['hits_authority'].map(lambda x: f"{x:.6f}")
    df_export['hits_hub'] = df_export['hits_hub'].map(lambda x: f"{x:.6f}")
    df_export['betweenness'] = df_export['betweenness'].map(lambda x: f"{x:.6f}")
    df_export['closeness'] = df_export['closeness'].map(lambda x: f"{x:.6f}")

    df_export.to_csv(output_path, index=False, encoding='utf-8')

    logger.info("  CSV guardado exitosamente")


def guardar_analisis_json(analisis: Dict[str, Any], correlaciones: Dict[str, float],
                          output_path: str) -> None:
    """
    Guarda el análisis completo en formato JSON.

    Args:
        analisis: Diccionario con análisis Top-N
        correlaciones: Correlaciones entre métricas
        output_path: Ruta de salida
    """
    logger.info(f"Guardando análisis JSON en: {output_path}")

    df_top = analisis['df_top']

    # Convertir DataFrame a lista de diccionarios
    top_nodos = []
    for idx, row in df_top.iterrows():
        top_nodos.append({
            'ranking': int(row['ranking']),
            'nodo': int(row['nodo']),
            'metricas': {
                'pagerank': float(row['pagerank']),
                'in_degree': int(row['in_degree']),
                'out_degree': int(row['out_degree']),
                'hits_authority': float(row['hits_authority']),
                'hits_hub': float(row['hits_hub']),
                'betweenness': float(row['betweenness']),
                'closeness': float(row['closeness'])
            },
            'rol_principal': row['rol_principal'],
            'roles_secundarios': row['roles_secundarios'],
            'explicacion_rol': row['explicacion_rol'],
            'explicacion_completa': row['explicacion_completa']
        })

    resumen = {
        'resumen': {
            'total_nodos_analizados': len(df_top),
            'top_n': len(df_top),
            'diversidad_roles': analisis['diversidad_roles'],
            'concentracion_pagerank': analisis['concentracion_pagerank'],
            'comparacion_top_vs_resto': analisis['comparacion_top_vs_resto']
        },
        'correlaciones': correlaciones,
        'top_nodos': top_nodos
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    logger.info("  JSON guardado exitosamente")


def generar_insights_texto(analisis: Dict[str, Any], correlaciones: Dict[str, float],
                           output_path: str) -> None:
    """
    Genera archivo de texto con insights principales.

    Args:
        analisis: Análisis Top-N
        correlaciones: Correlaciones entre métricas
        output_path: Ruta de salida
    """
    logger.info(f"Generando insights de texto en: {output_path}")

    df_top = analisis['df_top']
    n = len(df_top)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"ANÁLISIS TOP-{n} URLS MÁS INFLUYENTES\n")
        f.write("="*80 + "\n\n")

        # 1. Resumen ejecutivo
        f.write("1. RESUMEN EJECUTIVO\n")
        f.write("-" * 80 + "\n\n")

        concentracion = analisis['concentracion_pagerank']
        f.write(f"- Los Top-5 nodos concentran {concentracion['top_5']*100:.2f}% del PageRank total\n")
        f.write(f"- Los Top-10 nodos concentran {concentracion['top_10']*100:.2f}% del PageRank total\n")
        f.write(f"- Los Top-{n} nodos concentran {concentracion[f'top_{n}']*100:.2f}% del PageRank total\n\n")

        # Diversidad de roles
        f.write("Diversidad de roles:\n")
        for rol, cantidad in analisis['diversidad_roles'].items():
            f.write(f"  - {rol}: {cantidad} nodos ({cantidad/n*100:.1f}%)\n")
        f.write("\n")

        # 2. Correlaciones principales
        f.write("2. CORRELACIONES ENTRE MÉTRICAS\n")
        f.write("-" * 80 + "\n\n")

        for nombre, valor in correlaciones.items():
            interpretacion = ""
            if abs(valor) > 0.7:
                interpretacion = "FUERTE - Las métricas están altamente relacionadas"
            elif abs(valor) > 0.4:
                interpretacion = "MODERADA - Existe relación pero con variabilidad"
            else:
                interpretacion = "DÉBIL - Métricas relativamente independientes"

            f.write(f"{nombre}: {valor:.4f} ({interpretacion})\n")
        f.write("\n")

        # 3. Top-5 más detallado
        f.write("3. TOP-5 NODOS MÁS INFLUYENTES\n")
        f.write("-" * 80 + "\n\n")

        for idx, row in df_top.head(5).iterrows():
            f.write(f"[{int(row['ranking'])}] Nodo {int(row['nodo'])}\n")
            f.write(f"    Rol: {row['rol_principal']}\n")
            f.write(f"    PageRank: {row['pagerank']:.6f}\n")
            f.write(f"    In-Degree: {int(row['in_degree'])} | Out-Degree: {int(row['out_degree'])}\n")
            f.write(f"    Authority: {row['hits_authority']:.6f} | Hub: {row['hits_hub']:.6f}\n")
            f.write(f"    Explicación: {row['explicacion_completa']}\n")
            f.write("\n")

        # 4. Hallazgos principales
        f.write("4. HALLAZGOS PRINCIPALES\n")
        f.write("-" * 80 + "\n\n")

        comparacion = analisis['comparacion_top_vs_resto']

        f.write(f"- PageRank promedio Top-{n}: {comparacion['pagerank']['ratio']:.2f}x superior al resto\n")
        f.write(f"- In-Degree promedio Top-{n}: {comparacion['in_degree']['ratio']:.2f}x superior al resto\n")
        f.write(f"- Betweenness promedio Top-{n}: {comparacion['betweenness']['ratio']:.2f}x superior al resto\n\n")

        # Interpretación según correlaciones
        if correlaciones['pagerank_vs_authority'] > 0.8:
            f.write("- Alta correlación PageRank-Authority: Los nodos influyentes son principalmente\n")
            f.write("  destinos de alta calidad (contenido valioso).\n")

        if correlaciones['pagerank_vs_betweenness'] < 0.3:
            f.write("- Baja correlación PageRank-Betweenness: Los nodos más influyentes no necesariamente\n")
            f.write("  son los puentes principales. Existen nodos ocultos con alto betweenness.\n")

        if concentracion['top_5'] > 0.3:
            f.write(f"- Alta concentración en Top-5 ({concentracion['top_5']*100:.1f}%): Distribución tipo 'winner-takes-all'.\n")
            f.write("  Recomendación: Estrategias de link building hacia estos nodos tienen alto ROI.\n")

        f.write("\n")

        # 5. Recomendaciones iniciales
        f.write("5. RECOMENDACIONES INICIALES\n")
        f.write("-" * 80 + "\n\n")

        num_autoridades = analisis['diversidad_roles'].get('Autoridad', 0)
        num_hubs = analisis['diversidad_roles'].get('Hub', 0)
        num_puentes = analisis['diversidad_roles'].get('Puente', 0)

        if num_autoridades > n * 0.4:
            f.write(f"- Alta presencia de Autoridades ({num_autoridades}): Priorizar creación de contenido\n")
            f.write("  de alta calidad en estas URLs. Son destinos naturales de enlaces.\n\n")

        if num_hubs > n * 0.2:
            f.write(f"- Presencia de Hubs ({num_hubs}): Optimizar páginas de navegación/categoría.\n")
            f.write("  Asegurar que apunten a las mejores autoridades.\n\n")

        if num_puentes > 0:
            f.write(f"- Nodos Puente identificados ({num_puentes}): Monitoreo prioritario.\n")
            f.write("  Su caída fragmentaría el grafo. Considerar rutas alternativas.\n\n")

        f.write("="*80 + "\n")

    logger.info("  Insights de texto generados exitosamente")


def ejecutar_fase4(config_path: Optional[str] = None) -> pd.DataFrame:
    """
    Ejecuta la FASE 4: Ranking y Análisis Top-N.

    Args:
        config_path: Ruta a archivo de configuración (None usa default)

    Returns:
        pd.DataFrame: DataFrame con Top-N nodos analizados
    """
    logger.info("="*80)
    logger.info("FASE 4: RANKING Y ANÁLISIS TOP-N")
    logger.info("="*80)

    # 1. Cargar configuración
    config = cargar_configuracion(config_path)

    base_dir = Path(__file__).resolve().parent.parent
    results_dir = base_dir / config['outputs']['results_dir']
    metricas_csv = results_dir / 'fase3_metricas_centralidad.csv'

    top_n = config['ranking']['top_n']
    metrica_principal = config['ranking']['metrica_principal']
    umbral_percentil = config['ranking']['umbrales']['authority_percentile']

    # 2. Cargar métricas
    df_metricas = cargar_metricas(metricas_csv)

    # 3. Calcular ranking
    df_ranked = calcular_ranking(df_metricas, metrica_principal)

    # 4. Calcular percentiles
    percentiles = calcular_percentiles(df_ranked)

    # 5. Analizar Top-N
    analisis = analizar_top_n(df_ranked, top_n, percentiles, umbral_percentil)

    # 6. Calcular correlaciones
    correlaciones = calcular_correlaciones(df_ranked)

    # 7. Guardar resultados
    output_csv = results_dir / 'fase4_top_20_urls.csv'
    output_json = results_dir / 'fase4_analisis_top20.json'
    output_txt = results_dir / 'fase4_insights.txt'

    guardar_ranking_csv(analisis['df_top'], output_csv)
    guardar_analisis_json(analisis, correlaciones, output_json)
    generar_insights_texto(analisis, correlaciones, output_txt)

    logger.info("")
    logger.info("="*80)
    logger.info("FASE 4 COMPLETADA")
    logger.info("="*80)

    return analisis['df_top']


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Ejecutar
    df_top = ejecutar_fase4()

    print(f"\n[COMPLETADO] Análisis Top-{len(df_top)} generado")
    print(f"\nTop-5 nodos por PageRank:")
    print(df_top.head(5)[['ranking', 'nodo', 'pagerank', 'in_degree', 'rol_principal']])
