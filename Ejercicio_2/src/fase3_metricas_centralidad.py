#!/usr/bin/env python3
"""
FASE 3: Métricas de Centralidad

Este módulo calcula las 5 métricas de centralidad principales para el subgrafo:
1. PageRank - Autoridad basada en calidad y cantidad de enlaces entrantes
2. Degree Centrality - In-degree (popularidad) y Out-degree (conectividad)
3. HITS - Hubs (nodos que apuntan a autoridades) y Authorities (nodos apuntados por hubs)
4. Betweenness Centrality - Nodos que actúan como puentes entre comunidades
5. Closeness Centrality - Cercanía promedio a otros nodos

Autor: Danilo Melo
Fecha: 2026-01-11
"""

import logging
import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import yaml
import pickle
import json
from tqdm import tqdm

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


def cargar_subgrafo(subgrafo_path: str) -> nx.DiGraph:
    """
    Carga el subgrafo desde archivo pickle.

    Args:
        subgrafo_path: Ruta al archivo .gpickle del subgrafo

    Returns:
        nx.DiGraph: Subgrafo cargado
    """
    logger.info(f"Cargando subgrafo desde: {subgrafo_path}")

    with open(subgrafo_path, 'rb') as f:
        G = pickle.load(f)

    logger.info(f"Subgrafo cargado: {G.number_of_nodes():,} nodos, {G.number_of_edges():,} aristas")

    return G


def calcular_pagerank(G: nx.DiGraph, alpha: float = 0.85,
                      max_iter: int = 100, tol: float = 1e-6) -> Dict[int, float]:
    """
    Calcula PageRank para todos los nodos del grafo.

    PageRank mide la autoridad de una página web basándose en:
    - Cantidad de enlaces entrantes (popularidad)
    - Calidad de las páginas que enlazan (autoridad transitiva)

    Args:
        G: Grafo dirigido
        alpha: Factor de amortiguación (probabilidad de seguir enlaces, default=0.85)
        max_iter: Máximo de iteraciones
        tol: Tolerancia para convergencia

    Returns:
        Dict[int, float]: Diccionario {nodo: pagerank_score}
    """
    logger.info("Calculando PageRank...")
    logger.info(f"  Parámetros: alpha={alpha}, max_iter={max_iter}, tol={tol}")

    pagerank = nx.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol)

    logger.info(f"  PageRank calculado para {len(pagerank):,} nodos")

    # Mostrar top-5 nodos
    top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info("  Top-5 nodos por PageRank:")
    for i, (node, score) in enumerate(top_nodes, 1):
        logger.info(f"    {i}. Nodo {node}: {score:.6f}")

    return pagerank


def calcular_degree_centrality(G: nx.DiGraph) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Calcula In-Degree y Out-Degree Centrality.

    - In-Degree: Número de enlaces entrantes (popularidad)
    - Out-Degree: Número de enlaces salientes (conectividad)

    Args:
        G: Grafo dirigido

    Returns:
        Tuple[Dict, Dict]: (in_degree, out_degree)
    """
    logger.info("Calculando Degree Centrality...")

    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())

    logger.info(f"  In-Degree calculado para {len(in_degree):,} nodos")
    logger.info(f"  Out-Degree calculado para {len(out_degree):,} nodos")

    # Mostrar top-5 por in-degree
    top_in = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info("  Top-5 nodos por In-Degree:")
    for i, (node, degree) in enumerate(top_in, 1):
        logger.info(f"    {i}. Nodo {node}: {degree} enlaces entrantes")

    # Mostrar top-5 por out-degree
    top_out = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info("  Top-5 nodos por Out-Degree:")
    for i, (node, degree) in enumerate(top_out, 1):
        logger.info(f"    {i}. Nodo {node}: {degree} enlaces salientes")

    return in_degree, out_degree


def calcular_hits(G: nx.DiGraph, max_iter: int = 100,
                  tol: float = 1e-6) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Calcula HITS (Hyperlink-Induced Topic Search) - Algoritmo de Kleinberg.

    - Hubs: Nodos que apuntan a muchas autoridades (páginas de referencia)
    - Authorities: Nodos apuntados por muchos hubs (páginas con contenido de valor)

    Args:
        G: Grafo dirigido
        max_iter: Máximo de iteraciones
        tol: Tolerancia para convergencia

    Returns:
        Tuple[Dict, Dict]: (hubs, authorities)
    """
    logger.info("Calculando HITS (Hubs and Authorities)...")
    logger.info(f"  Parámetros: max_iter={max_iter}, tol={tol}")

    hubs, authorities = nx.hits(G, max_iter=max_iter, tol=tol)

    logger.info(f"  HITS calculado para {len(hubs):,} nodos")

    # Mostrar top-5 authorities
    top_auth = sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info("  Top-5 Authorities:")
    for i, (node, score) in enumerate(top_auth, 1):
        logger.info(f"    {i}. Nodo {node}: {score:.6f}")

    # Mostrar top-5 hubs
    top_hubs = sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info("  Top-5 Hubs:")
    for i, (node, score) in enumerate(top_hubs, 1):
        logger.info(f"    {i}. Nodo {node}: {score:.6f}")

    return hubs, authorities


def calcular_betweenness(G: nx.DiGraph, k: Optional[int] = None,
                         normalized: bool = True) -> Dict[int, float]:
    """
    Calcula Betweenness Centrality.

    Mide qué tan frecuentemente un nodo aparece en los caminos más cortos
    entre otros nodos. Identifica nodos "puente" que conectan diferentes
    comunidades o secciones del grafo.

    Args:
        G: Grafo dirigido
        k: Si no es None, usa muestra aleatoria de k nodos (más rápido)
        normalized: Si True, normaliza valores entre 0 y 1

    Returns:
        Dict[int, float]: Diccionario {nodo: betweenness_score}
    """
    logger.info("Calculando Betweenness Centrality...")

    if k is not None:
        logger.info(f"  Usando muestra de {k} nodos (aproximación)")
    else:
        logger.info("  Calculando para todos los nodos (puede tardar varios minutos)")

    logger.info(f"  Normalized: {normalized}")

    betweenness = nx.betweenness_centrality(G, k=k, normalized=normalized)

    logger.info(f"  Betweenness calculado para {len(betweenness):,} nodos")

    # Mostrar top-5 nodos
    top_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info("  Top-5 nodos por Betweenness:")
    for i, (node, score) in enumerate(top_nodes, 1):
        logger.info(f"    {i}. Nodo {node}: {score:.6f}")

    return betweenness


def calcular_closeness(G: nx.DiGraph, wf_improved: bool = True) -> Dict[int, float]:
    """
    Calcula Closeness Centrality.

    Mide qué tan cerca está un nodo de todos los demás nodos en el grafo.
    Nodos con alta closeness pueden difundir información rápidamente.

    NOTA: En grafos dirigidos no fuertemente conexos, algunos nodos pueden
    tener closeness = 0 si no pueden alcanzar otros nodos.

    Args:
        G: Grafo dirigido
        wf_improved: Si True, usa fórmula mejorada de Wasserman-Faust

    Returns:
        Dict[int, float]: Diccionario {nodo: closeness_score}
    """
    logger.info("Calculando Closeness Centrality...")
    logger.info(f"  Wasserman-Faust improved formula: {wf_improved}")

    closeness = nx.closeness_centrality(G, wf_improved=wf_improved)

    logger.info(f"  Closeness calculado para {len(closeness):,} nodos")

    # Contar nodos con closeness = 0
    zero_closeness = sum(1 for score in closeness.values() if score == 0)
    if zero_closeness > 0:
        logger.info(f"  ADVERTENCIA: {zero_closeness:,} nodos tienen closeness=0 (no alcanzan otros nodos)")

    # Mostrar top-5 nodos (excluyendo closeness=0)
    non_zero = [(node, score) for node, score in closeness.items() if score > 0]
    if non_zero:
        top_nodes = sorted(non_zero, key=lambda x: x[1], reverse=True)[:5]
        logger.info("  Top-5 nodos por Closeness:")
        for i, (node, score) in enumerate(top_nodes, 1):
            logger.info(f"    {i}. Nodo {node}: {score:.6f}")

    return closeness


def consolidar_metricas(
    pagerank: Dict[int, float],
    in_degree: Dict[int, int],
    out_degree: Dict[int, int],
    hubs: Dict[int, float],
    authorities: Dict[int, float],
    betweenness: Dict[int, float],
    closeness: Dict[int, float]
) -> pd.DataFrame:
    """
    Consolida todas las métricas en un único DataFrame.

    Args:
        pagerank: Scores de PageRank
        in_degree: In-degree de cada nodo
        out_degree: Out-degree de cada nodo
        hubs: Scores de HITS Hubs
        authorities: Scores de HITS Authorities
        betweenness: Scores de Betweenness
        closeness: Scores de Closeness

    Returns:
        pd.DataFrame: DataFrame con todas las métricas por nodo
    """
    logger.info("Consolidando métricas en DataFrame...")

    # Obtener lista de nodos (deberían ser los mismos en todas las métricas)
    nodes = sorted(pagerank.keys())

    # Crear DataFrame
    df = pd.DataFrame({
        'nodo': nodes,
        'pagerank': [pagerank[node] for node in nodes],
        'in_degree': [in_degree[node] for node in nodes],
        'out_degree': [out_degree[node] for node in nodes],
        'hits_hub': [hubs[node] for node in nodes],
        'hits_authority': [authorities[node] for node in nodes],
        'betweenness': [betweenness[node] for node in nodes],
        'closeness': [closeness[node] for node in nodes]
    })

    logger.info(f"  DataFrame creado: {len(df):,} filas, {len(df.columns)} columnas")

    # Calcular estadísticas básicas
    logger.info("\n  Estadísticas de las métricas:")
    logger.info(f"    PageRank - min: {df['pagerank'].min():.6f}, max: {df['pagerank'].max():.6f}, promedio: {df['pagerank'].mean():.6f}")
    logger.info(f"    In-Degree - min: {df['in_degree'].min()}, max: {df['in_degree'].max()}, promedio: {df['in_degree'].mean():.1f}")
    logger.info(f"    Out-Degree - min: {df['out_degree'].min()}, max: {df['out_degree'].max()}, promedio: {df['out_degree'].mean():.1f}")
    logger.info(f"    Authority - min: {df['hits_authority'].min():.6f}, max: {df['hits_authority'].max():.6f}, promedio: {df['hits_authority'].mean():.6f}")
    logger.info(f"    Hub - min: {df['hits_hub'].min():.6f}, max: {df['hits_hub'].max():.6f}, promedio: {df['hits_hub'].mean():.6f}")
    logger.info(f"    Betweenness - min: {df['betweenness'].min():.6f}, max: {df['betweenness'].max():.6f}, promedio: {df['betweenness'].mean():.6f}")
    logger.info(f"    Closeness - min: {df['closeness'].min():.6f}, max: {df['closeness'].max():.6f}, promedio: {df['closeness'].mean():.6f}")

    return df


def guardar_metricas(df: pd.DataFrame, output_dir: str) -> None:
    """
    Guarda las métricas en formato CSV y JSON.

    Args:
        df: DataFrame con todas las métricas
        output_dir: Directorio de salida
    """
    logger.info(f"Guardando métricas en: {output_dir}")

    # Crear directorio si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Guardar CSV completo
    csv_path = Path(output_dir) / 'fase3_metricas_centralidad.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"  CSV guardado: {csv_path}")

    # Guardar resumen JSON con estadísticas
    resumen = {
        'num_nodos': len(df),
        'metricas': {
            'pagerank': {
                'min': float(df['pagerank'].min()),
                'max': float(df['pagerank'].max()),
                'mean': float(df['pagerank'].mean()),
                'std': float(df['pagerank'].std())
            },
            'in_degree': {
                'min': int(df['in_degree'].min()),
                'max': int(df['in_degree'].max()),
                'mean': float(df['in_degree'].mean()),
                'std': float(df['in_degree'].std())
            },
            'out_degree': {
                'min': int(df['out_degree'].min()),
                'max': int(df['out_degree'].max()),
                'mean': float(df['out_degree'].mean()),
                'std': float(df['out_degree'].std())
            },
            'hits_authority': {
                'min': float(df['hits_authority'].min()),
                'max': float(df['hits_authority'].max()),
                'mean': float(df['hits_authority'].mean()),
                'std': float(df['hits_authority'].std())
            },
            'hits_hub': {
                'min': float(df['hits_hub'].min()),
                'max': float(df['hits_hub'].max()),
                'mean': float(df['hits_hub'].mean()),
                'std': float(df['hits_hub'].std())
            },
            'betweenness': {
                'min': float(df['betweenness'].min()),
                'max': float(df['betweenness'].max()),
                'mean': float(df['betweenness'].mean()),
                'std': float(df['betweenness'].std())
            },
            'closeness': {
                'min': float(df['closeness'].min()),
                'max': float(df['closeness'].max()),
                'mean': float(df['closeness'].mean()),
                'std': float(df['closeness'].std())
            }
        },
        'top_10_por_metrica': {
            'pagerank': df.nlargest(10, 'pagerank')[['nodo', 'pagerank']].to_dict('records'),
            'in_degree': df.nlargest(10, 'in_degree')[['nodo', 'in_degree']].to_dict('records'),
            'out_degree': df.nlargest(10, 'out_degree')[['nodo', 'out_degree']].to_dict('records'),
            'hits_authority': df.nlargest(10, 'hits_authority')[['nodo', 'hits_authority']].to_dict('records'),
            'hits_hub': df.nlargest(10, 'hits_hub')[['nodo', 'hits_hub']].to_dict('records'),
            'betweenness': df.nlargest(10, 'betweenness')[['nodo', 'betweenness']].to_dict('records'),
            'closeness': df.nlargest(10, 'closeness')[['nodo', 'closeness']].to_dict('records')
        }
    }

    json_path = Path(output_dir) / 'fase3_resumen_metricas.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    logger.info(f"  JSON guardado: {json_path}")

    logger.info("Métricas guardadas exitosamente")


def ejecutar_fase3(config_path: Optional[str] = None) -> pd.DataFrame:
    """
    Ejecuta la FASE 3: Cálculo de Métricas de Centralidad.

    Args:
        config_path: Ruta a archivo de configuración (None usa default)

    Returns:
        pd.DataFrame: DataFrame con todas las métricas calculadas
    """
    logger.info("="*80)
    logger.info("FASE 3: METRICAS DE CENTRALIDAD")
    logger.info("="*80)

    # 1. Cargar configuración
    config = cargar_configuracion(config_path)

    base_dir = Path(__file__).resolve().parent.parent
    subgrafo_path = base_dir / config['data']['processed_dir'] / 'subgrafo.gpickle'
    output_dir = base_dir / config['outputs']['results_dir']

    # 2. Cargar subgrafo
    G = cargar_subgrafo(subgrafo_path)

    # 3. Calcular PageRank
    pagerank = calcular_pagerank(
        G,
        alpha=config['metricas']['pagerank']['alpha'],
        max_iter=config['metricas']['pagerank']['max_iter'],
        tol=config['metricas']['pagerank']['tol']
    )

    # 4. Calcular Degree Centrality
    in_degree, out_degree = calcular_degree_centrality(G)

    # 5. Calcular HITS
    hubs, authorities = calcular_hits(
        G,
        max_iter=config['metricas']['hits']['max_iter'],
        tol=config['metricas']['hits']['tol']
    )

    # 6. Calcular Betweenness Centrality
    betweenness = calcular_betweenness(
        G,
        k=config['metricas']['betweenness']['k'],
        normalized=config['metricas']['betweenness']['normalized']
    )

    # 7. Calcular Closeness Centrality
    closeness = calcular_closeness(
        G,
        wf_improved=config['metricas']['closeness']['wf_improved']
    )

    # 8. Consolidar métricas en DataFrame
    df_metricas = consolidar_metricas(
        pagerank, in_degree, out_degree,
        hubs, authorities, betweenness, closeness
    )

    # 9. Guardar resultados
    guardar_metricas(df_metricas, output_dir)

    logger.info("="*80)
    logger.info("FASE 3 COMPLETADA")
    logger.info("="*80)

    return df_metricas


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Ejecutar
    df = ejecutar_fase3()

    print(f"\n[COMPLETADO] Métricas calculadas para {len(df):,} nodos")
    print(f"\nTop-5 nodos por PageRank:")
    print(df.nlargest(5, 'pagerank')[['nodo', 'pagerank', 'in_degree', 'hits_authority']])
