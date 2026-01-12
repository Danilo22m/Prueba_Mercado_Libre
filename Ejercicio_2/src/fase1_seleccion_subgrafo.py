#!/usr/bin/env python3
"""
FASE 1: Selección de Subgrafo

Este módulo implementa la selección de un subgrafo manejable desde el
archivo web-Stanford.txt mediante:
1. Carga del grafo completo
2. Extracción del Largest Weakly Connected Component (WCC)
3. Reducción a ~30k aristas
4. Validación de tamaño y diversidad de grados

Autor: Danilo Melo
Fecha: 2026-01-11
"""

import logging
import networkx as nx
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import yaml
import pickle

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


def cargar_grafo_completo(file_path: str) -> nx.DiGraph:
    """
    Carga el grafo completo desde el archivo web-Stanford.txt.

    Args:
        file_path: Ruta al archivo de datos

    Returns:
        nx.DiGraph: Grafo dirigido completo
    """
    logger.info(f"Cargando grafo desde: {file_path}")

    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue

            parts = line.strip().split()
            if len(parts) == 2:
                from_node = int(parts[0])
                to_node = int(parts[1])
                edges.append((from_node, to_node))

    G = nx.DiGraph()
    G.add_edges_from(edges)

    logger.info(f"✓ Grafo cargado: {G.number_of_nodes():,} nodos, {G.number_of_edges():,} aristas")

    return G


def extraer_largest_wcc(G: nx.DiGraph) -> nx.DiGraph:
    """
    Extrae el Largest Weakly Connected Component (WCC) del grafo.

    Args:
        G: Grafo dirigido completo

    Returns:
        nx.DiGraph: Subgrafo correspondiente al WCC más grande
    """
    logger.info("Extrayendo Largest Weakly Connected Component...")

    wccs = list(nx.weakly_connected_components(G))
    logger.info(f"  Encontrados {len(wccs)} componentes débilmente conexas")

    wccs_sorted = sorted(wccs, key=len, reverse=True)

    for i, wcc in enumerate(wccs_sorted[:5]):
        logger.info(f"  WCC {i+1}: {len(wcc):,} nodos")

    largest_wcc = wccs_sorted[0]

    logger.info(f"✓ Largest WCC seleccionado: {len(largest_wcc):,} nodos ({100*len(largest_wcc)/G.number_of_nodes():.1f}% del total)")

    G_wcc = G.subgraph(largest_wcc).copy()

    logger.info(f"  Aristas en WCC: {G_wcc.number_of_edges():,}")

    return G_wcc


def seleccionar_subgrafo_por_degree(G: nx.DiGraph, target_edges: int,
                                     include_neighbors: bool = True) -> nx.DiGraph:
    """
    Selecciona un subgrafo basado en los nodos con mayor degree.

    Args:
        G: Grafo de entrada
        target_edges: Número objetivo de aristas
        include_neighbors: Si True, agrega vecinos de 1er nivel

    Returns:
        nx.DiGraph: Subgrafo seleccionado
    """
    logger.info(f"Seleccionando subgrafo con ~{target_edges:,} aristas...")

    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

    best_subgraph = None
    best_diff = float('inf')

    for top_n in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]:
        core_nodes = [node for node, degree in sorted_nodes[:top_n]]

        if include_neighbors:
            extended_nodes = set(core_nodes)
            for node in core_nodes:
                extended_nodes.update(G.successors(node))
                extended_nodes.update(G.predecessors(node))

            selected_nodes = list(extended_nodes)
        else:
            selected_nodes = core_nodes

        G_sub = G.subgraph(selected_nodes).copy()

        num_edges = G_sub.number_of_edges()
        diff = abs(num_edges - target_edges)

        logger.info(f"  top_n={top_n:,}: {G_sub.number_of_nodes():,} nodos, {num_edges:,} aristas (diff={diff:,})")

        if diff < best_diff:
            best_diff = diff
            best_subgraph = G_sub

        if diff < 0.1 * target_edges:
            break

    logger.info(f"✓ Subgrafo seleccionado: {best_subgraph.number_of_nodes():,} nodos, {best_subgraph.number_of_edges():,} aristas")

    return best_subgraph


def validar_subgrafo(G: nx.DiGraph, min_edges: int, max_edges: int) -> bool:
    """
    Valida que el subgrafo cumpla los requisitos de tamaño.

    Args:
        G: Subgrafo a validar
        min_edges: Mínimo de aristas aceptable
        max_edges: Máximo de aristas aceptable

    Returns:
        bool: True si cumple los requisitos
    """
    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()

    logger.info("Validando subgrafo...")
    logger.info(f"  Nodos: {num_nodes:,}")
    logger.info(f"  Aristas: {num_edges:,}")

    if num_edges < min_edges:
        logger.warning(f"⚠️  Subgrafo muy pequeño: {num_edges:,} < {min_edges:,} aristas")
        return False
    elif num_edges > max_edges:
        logger.warning(f"⚠️  Subgrafo muy grande: {num_edges:,} > {max_edges:,} aristas")
        return False

    logger.info(f"✓ Tamaño correcto: {min_edges:,} ≤ {num_edges:,} ≤ {max_edges:,}")

    degrees = [degree for node, degree in G.degree()]
    min_degree = min(degrees)
    max_degree = max(degrees)
    avg_degree = sum(degrees) / len(degrees)

    logger.info(f"  Grado mínimo: {min_degree}")
    logger.info(f"  Grado máximo: {max_degree:,}")
    logger.info(f"  Grado promedio: {avg_degree:.1f}")

    if max_degree < 10:
        logger.warning("⚠️  No hay nodos con grado alto (max < 10)")
        return False

    logger.info("✓ Diversidad de grados adecuada")

    n = G.number_of_nodes()
    max_possible_edges = n * (n - 1)
    density = num_edges / max_possible_edges if max_possible_edges > 0 else 0

    logger.info(f"  Densidad: {density:.6f} ({100*density:.4f}%)")

    return True


def guardar_subgrafo(G: nx.DiGraph, output_path: str) -> None:
    """
    Guarda el subgrafo en formato pickle (NetworkX).

    Args:
        G: Grafo a guardar
        output_path: Ruta de salida
    """
    logger.info(f"Guardando subgrafo en: {output_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("✓ Subgrafo guardado exitosamente")


def ejecutar_fase1(config_path: Optional[str] = None) -> nx.DiGraph:
    """
    Ejecuta la FASE 1: Selección de Subgrafo.

    Args:
        config_path: Ruta a archivo de configuración (None usa default)

    Returns:
        nx.DiGraph: Subgrafo seleccionado
    """
    logger.info("="*80)
    logger.info("FASE 1: SELECCIÓN DE SUBGRAFO")
    logger.info("="*80)

    # 1. Cargar configuración
    config = cargar_configuracion(config_path)

    base_dir = Path(__file__).resolve().parent.parent
    raw_file = base_dir / config['data']['raw_file']
    processed_dir = base_dir / config['data']['processed_dir']

    # 2. Cargar grafo completo
    G_full = cargar_grafo_completo(raw_file)

    # 3. Extraer Largest WCC
    if config['grafo']['use_largest_wcc']:
        G_wcc = extraer_largest_wcc(G_full)
    else:
        G_wcc = G_full

    # 4. Reducir a subgrafo objetivo
    target_edges = config['grafo']['target_edges']
    include_neighbors = config['grafo']['include_neighbors']

    if config['grafo']['target_min_edges'] <= G_wcc.number_of_edges() <= config['grafo']['target_max_edges']:
        logger.info("✓ WCC ya tiene tamaño objetivo, no se requiere reducción adicional")
        G_sub = G_wcc
    else:
        G_sub = seleccionar_subgrafo_por_degree(
            G_wcc,
            target_edges=target_edges,
            include_neighbors=include_neighbors
        )

    # 5. Validar subgrafo
    is_valid = validar_subgrafo(
        G_sub,
        min_edges=config['grafo']['target_min_edges'],
        max_edges=config['grafo']['target_max_edges']
    )

    if not is_valid:
        logger.warning("⚠️  El subgrafo no cumple todos los requisitos, pero se procederá")

    # 6. Guardar subgrafo
    subgrafo_path = processed_dir / 'subgrafo.gpickle'
    guardar_subgrafo(G_sub, subgrafo_path)

    logger.info("="*80)
    logger.info("FASE 1 COMPLETADA")
    logger.info("="*80)

    return G_sub


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Ejecutar
    G = ejecutar_fase1()

    print(f"\n✓ Subgrafo final: {G.number_of_nodes():,} nodos, {G.number_of_edges():,} aristas")
