#!/usr/bin/env python3
"""
FASE 5: Subgrafo Explicativo y Visualización

Este módulo genera visualizaciones estáticas e interactivas del subgrafo
centrado en los nodos más influyentes:

1. Inducción de subgrafo (Top-N + vecinos)
2. Visualización estática (Matplotlib + NetworkX)
3. Visualización interactiva (Pyvis)
4. Análisis de patrones visuales
5. Reporte de insights

Autor: Danilo Melo
Fecha: 2026-01-11
"""

import logging
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple
import yaml
import pickle
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


def cargar_grafo(grafo_path: str) -> nx.DiGraph:
    """
    Carga el grafo completo desde archivo pickle.

    Args:
        grafo_path: Ruta al archivo .gpickle

    Returns:
        nx.DiGraph: Grafo cargado
    """
    logger.info(f"Cargando grafo desde: {grafo_path}")

    with open(grafo_path, 'rb') as f:
        G = pickle.load(f)

    logger.info(f"Grafo cargado: {G.number_of_nodes():,} nodos, {G.number_of_edges():,} aristas")

    return G


def cargar_metricas_csv(csv_path: str) -> pd.DataFrame:
    """
    Carga métricas desde CSV.

    Args:
        csv_path: Ruta al CSV con métricas

    Returns:
        pd.DataFrame: Métricas
    """
    logger.info(f"Cargando métricas desde: {csv_path}")

    df = pd.read_csv(csv_path)

    logger.info(f"Métricas cargadas: {len(df):,} nodos")

    return df


def inducir_subgrafo(G: nx.DiGraph, df_top: pd.DataFrame,
                     top_n: int = 10,
                     incluir_vecinos: bool = True,
                     incluir_vecinos_2do: bool = False) -> nx.DiGraph:
    """
    Induce subgrafo centrado en Top-N nodos más influyentes.

    Args:
        G: Grafo completo
        df_top: DataFrame con ranking (debe tener columna 'nodo')
        top_n: Número de nodos top a incluir como núcleo
        incluir_vecinos: Si True, agrega vecinos de 1er nivel
        incluir_vecinos_2do: Si True, agrega vecinos de 2do nivel para Top-3

    Returns:
        nx.DiGraph: Subgrafo inducido
    """
    logger.info(f"Induciendo subgrafo con Top-{top_n}...")

    # Seleccionar núcleo (Top-N)
    core_nodes = set(df_top.head(top_n)['nodo'].tolist())
    logger.info(f"  Núcleo: {len(core_nodes)} nodos")

    selected_nodes = set(core_nodes)

    if incluir_vecinos:
        # Agregar vecinos de 1er nivel
        for node in core_nodes:
            if node in G:
                selected_nodes.update(G.successors(node))
                selected_nodes.update(G.predecessors(node))

        logger.info(f"  + Vecinos 1er nivel: {len(selected_nodes)} nodos totales")

    if incluir_vecinos_2do and top_n >= 3:
        # Agregar vecinos de 2do nivel solo para Top-3
        top_3 = set(df_top.head(3)['nodo'].tolist())
        vecinos_1er_nivel = set()

        for node in top_3:
            if node in G:
                vecinos_1er_nivel.update(G.successors(node))
                vecinos_1er_nivel.update(G.predecessors(node))

        # Ahora agregar vecinos de esos vecinos
        for node in vecinos_1er_nivel:
            if node in G:
                selected_nodes.update(G.successors(node))
                selected_nodes.update(G.predecessors(node))

        logger.info(f"  + Vecinos 2do nivel (Top-3): {len(selected_nodes)} nodos totales")

    # Filtrar nodos que existen en G
    selected_nodes = {n for n in selected_nodes if n in G}

    # Inducir subgrafo
    G_sub = G.subgraph(selected_nodes).copy()

    logger.info(f"Subgrafo inducido: {G_sub.number_of_nodes():,} nodos, {G_sub.number_of_edges():,} aristas")

    return G_sub


def preparar_metricas_subgrafo(G_sub: nx.DiGraph, df_metricas: pd.DataFrame) -> Dict[str, Dict[int, float]]:
    """
    Prepara diccionarios de métricas solo para nodos del subgrafo.

    Args:
        G_sub: Subgrafo
        df_metricas: DataFrame con métricas de todos los nodos

    Returns:
        Dict con métricas por nodo
    """
    logger.info("Preparando métricas para subgrafo...")

    nodos_sub = set(G_sub.nodes())

    # Filtrar solo nodos presentes en el subgrafo
    df_sub = df_metricas[df_metricas['nodo'].isin(nodos_sub)].copy()

    metricas = {
        'pagerank': dict(zip(df_sub['nodo'], df_sub['pagerank'])),
        'in_degree': dict(zip(df_sub['nodo'], df_sub['in_degree'])),
        'out_degree': dict(zip(df_sub['nodo'], df_sub['out_degree'])),
        'authority': dict(zip(df_sub['nodo'], df_sub['hits_authority'])),
        'hub': dict(zip(df_sub['nodo'], df_sub['hits_hub'])),
        'betweenness': dict(zip(df_sub['nodo'], df_sub['betweenness'])),
        'closeness': dict(zip(df_sub['nodo'], df_sub['closeness']))
    }

    logger.info(f"  Métricas preparadas para {len(nodos_sub):,} nodos")

    return metricas


def get_color_from_authority(authority: float, min_auth: float, max_auth: float) -> str:
    """
    Convierte authority score a color en escala Amarillo-Naranja-Rojo.

    Args:
        authority: Valor de authority
        min_auth: Mínimo authority en el dataset
        max_auth: Máximo authority en el dataset

    Returns:
        str: Color en formato hex
    """
    if max_auth == min_auth:
        normalized = 0.5
    else:
        normalized = (authority - min_auth) / (max_auth - min_auth)

    # Escala de colores: amarillo (#FFFF00) -> naranja (#FF8C00) -> rojo (#FF0000)
    cmap = plt.cm.YlOrRd
    rgba = cmap(normalized)
    hex_color = mcolors.rgb2hex(rgba[:3])

    return hex_color


def visualizar_estatico(G_sub: nx.DiGraph, metricas: Dict[str, Dict[int, float]],
                        top_nodes: List[int], config: dict, output_path: str) -> None:
    """
    Genera visualización estática con Matplotlib + NetworkX.

    Args:
        G_sub: Subgrafo a visualizar
        metricas: Diccionario con métricas por nodo
        top_nodes: Lista de nodos Top-N (para etiquetas)
        config: Configuración de visualización
        output_path: Ruta de salida para imagen
    """
    logger.info("Generando visualización estática...")

    # Configuración
    figsize = tuple(config['visualizacion']['estatica']['figsize'])
    dpi = config['visualizacion']['estatica']['dpi']
    layout_type = config['visualizacion']['estatica']['layout']
    node_size_scale = config['visualizacion']['estatica']['node_size_scale']
    edge_alpha = config['visualizacion']['estatica']['edge_alpha']
    show_labels_n = config['visualizacion']['estatica']['show_labels_top_n']

    # Crear figura
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Calcular layout
    logger.info(f"  Calculando layout: {layout_type}")
    if layout_type == 'spring':
        pos = nx.spring_layout(G_sub, k=0.5, iterations=50, seed=42)
    elif layout_type == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G_sub)
    elif layout_type == 'circular':
        pos = nx.circular_layout(G_sub)
    elif layout_type == 'shell':
        pos = nx.shell_layout(G_sub)
    else:
        pos = nx.spring_layout(G_sub, k=0.5, iterations=50, seed=42)

    # Preparar tamaños de nodos (según PageRank)
    node_sizes = []
    for node in G_sub.nodes():
        pr = metricas['pagerank'].get(node, 0)
        node_sizes.append(pr * node_size_scale)

    # Preparar colores de nodos (según Authority)
    authorities = [metricas['authority'].get(node, 0) for node in G_sub.nodes()]
    min_auth = min(authorities) if authorities else 0
    max_auth = max(authorities) if authorities else 1

    node_colors = [metricas['authority'].get(node, 0) for node in G_sub.nodes()]

    # Dibujar aristas
    nx.draw_networkx_edges(
        G_sub, pos,
        edge_color='gray',
        alpha=edge_alpha,
        arrows=True,
        arrowsize=10,
        arrowstyle='->',
        connectionstyle='arc3,rad=0.1',
        width=0.5,
        ax=ax
    )

    # Dibujar nodos
    nx.draw_networkx_nodes(
        G_sub, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.YlOrRd,
        vmin=min_auth,
        vmax=max_auth,
        alpha=0.9,
        ax=ax
    )

    # Dibujar etiquetas solo para Top-N
    labels_dict = {node: str(node) for node in top_nodes[:show_labels_n] if node in G_sub}

    nx.draw_networkx_labels(
        G_sub, pos,
        labels=labels_dict,
        font_size=8,
        font_weight='bold',
        font_color='black',
        ax=ax
    )

    # Título
    ax.set_title(
        f'Subgrafo de Top-{len(top_nodes)} URLs más Influyentes\n'
        f'({G_sub.number_of_nodes():,} nodos, {G_sub.number_of_edges():,} aristas)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # Leyenda de colores (Authority)
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.YlOrRd,
        norm=plt.Normalize(vmin=min_auth, vmax=max_auth)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Authority Score (HITS)', rotation=270, labelpad=20, fontsize=10)

    # Agregar texto explicativo
    info_text = (
        f'Tamaño de nodo = PageRank\n'
        f'Color de nodo = Authority Score\n'
        f'Etiquetas = Top-{show_labels_n} nodos'
    )
    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    ax.axis('off')
    plt.tight_layout()

    # Guardar
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"  Visualización guardada: {output_path}")


def visualizar_interactivo(G_sub: nx.DiGraph, metricas: Dict[str, Dict[int, float]],
                           top_nodes: List[int], config: dict, output_path: str) -> None:
    """
    Genera visualización interactiva con Pyvis.

    Args:
        G_sub: Subgrafo a visualizar
        metricas: Diccionario con métricas por nodo
        top_nodes: Lista de nodos Top-N
        config: Configuración de visualización
        output_path: Ruta de salida para HTML
    """
    logger.info("Generando visualización interactiva...")

    try:
        from pyvis.network import Network
    except ImportError:
        logger.error("pyvis no está instalado. Ejecutar: pip install pyvis")
        return

    # Configuración
    height = config['visualizacion']['interactiva']['height']
    width = config['visualizacion']['interactiva']['width']
    node_size_scale = config['visualizacion']['interactiva']['node_size_scale']
    physics_enabled = config['visualizacion']['interactiva']['physics_enabled']
    edge_color = config['visualizacion']['interactiva']['edge_color']

    # Crear red
    net = Network(
        height=height,
        width=width,
        directed=True,
        notebook=False
    )

    # Configurar física
    if physics_enabled:
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.09
            }
          }
        }
        """)
    else:
        net.toggle_physics(False)

    # Calcular rango de authority para colores
    authorities = [metricas['authority'].get(node, 0) for node in G_sub.nodes()]
    min_auth = min(authorities) if authorities else 0
    max_auth = max(authorities) if authorities else 1

    # Agregar nodos
    top_nodes_set = set(top_nodes[:10])

    for node in G_sub.nodes():
        # Tooltip con todas las métricas
        title = f"""
        <b>Nodo: {node}</b><br>
        PageRank: {metricas['pagerank'].get(node, 0):.6f}<br>
        In-Degree: {metricas['in_degree'].get(node, 0)}<br>
        Out-Degree: {metricas['out_degree'].get(node, 0)}<br>
        Authority: {metricas['authority'].get(node, 0):.6f}<br>
        Hub: {metricas['hub'].get(node, 0):.6f}<br>
        Betweenness: {metricas['betweenness'].get(node, 0):.6f}<br>
        Closeness: {metricas['closeness'].get(node, 0):.6f}
        """

        # Tamaño según PageRank
        size = metricas['pagerank'].get(node, 0) * node_size_scale

        # Color según Authority
        auth = metricas['authority'].get(node, 0)
        color = get_color_from_authority(auth, min_auth, max_auth)

        # Label solo para Top-N
        label = str(node) if node in top_nodes_set else ""

        net.add_node(
            node,
            label=label,
            title=title,
            size=max(size, 5),  # Tamaño mínimo 5
            color=color,
            borderWidth=3 if node in top_nodes_set else 1,
            borderWidthSelected=5
        )

    # Agregar aristas
    for edge in G_sub.edges():
        net.add_edge(edge[0], edge[1], color=edge_color, width=1)

    # Guardar
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output_path))

    logger.info(f"  Visualización interactiva guardada: {output_path}")


def analizar_patrones_subgrafo(G_sub: nx.DiGraph, metricas: Dict[str, Dict[int, float]],
                                top_nodes: List[int]) -> Dict[str, Any]:
    """
    Analiza patrones y estructura del subgrafo.

    Args:
        G_sub: Subgrafo
        metricas: Métricas de nodos
        top_nodes: Lista de Top-N nodos

    Returns:
        Dict con análisis de patrones
    """
    logger.info("Analizando patrones del subgrafo...")

    # Métricas básicas
    N = G_sub.number_of_nodes()
    E = G_sub.number_of_edges()
    density = E / (N * (N - 1)) if N > 1 else 0

    # Grado promedio
    degrees = [d for n, d in G_sub.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0

    # Componentes
    num_wccs = nx.number_weakly_connected_components(G_sub)
    num_sccs = nx.number_strongly_connected_components(G_sub)

    # Analizar Top-N en el subgrafo
    top_in_sub = [n for n in top_nodes if n in G_sub]
    total_in_degree_top = sum(G_sub.in_degree(n) for n in top_in_sub)
    total_edges = G_sub.number_of_edges()
    pct_edges_to_top = (total_in_degree_top / total_edges * 100) if total_edges > 0 else 0

    # Diámetro (solo si el grafo es débilmente conexo)
    try:
        if nx.is_weakly_connected(G_sub):
            # Convertir a no dirigido para calcular diámetro
            G_undirected = G_sub.to_undirected()
            diameter = nx.diameter(G_undirected)
        else:
            diameter = None
    except:
        diameter = None

    # Coeficiente de clustering promedio
    try:
        avg_clustering = nx.average_clustering(G_sub.to_undirected())
    except:
        avg_clustering = 0

    analisis = {
        'composicion': {
            'nodos_totales': N,
            'top_n_incluidos': len(top_in_sub),
            'vecinos': N - len(top_in_sub),
            'aristas': E,
            'densidad': float(density)
        },
        'metricas_estructura': {
            'grado_promedio': float(avg_degree),
            'diametro': diameter,
            'clustering_promedio': float(avg_clustering),
            'componentes_debiles': num_wccs,
            'componentes_fuertes': num_sccs
        },
        'concentracion': {
            'pct_aristas_hacia_top_n': float(pct_edges_to_top),
            'in_degree_promedio_top_n': float(total_in_degree_top / len(top_in_sub)) if top_in_sub else 0
        }
    }

    logger.info(f"  Densidad: {density:.6f}")
    logger.info(f"  Grado promedio: {avg_degree:.2f}")
    logger.info(f"  {pct_edges_to_top:.2f}% de aristas apuntan hacia Top-{len(top_in_sub)}")

    return analisis


def generar_reporte_insights(G_sub: nx.DiGraph, analisis: Dict[str, Any],
                             top_nodes: List[int], output_path: str) -> None:
    """
    Genera reporte de texto con insights visuales.

    Args:
        G_sub: Subgrafo
        analisis: Análisis de patrones
        top_nodes: Top-N nodos
        output_path: Ruta de salida
    """
    logger.info(f"Generando reporte de insights en: {output_path}")

    comp = analisis['composicion']
    estructura = analisis['metricas_estructura']
    concentracion = analisis['concentracion']

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ANÁLISIS VISUAL DEL SUBGRAFO\n")
        f.write("="*80 + "\n\n")

        # 1. Composición
        f.write("1. COMPOSICIÓN DEL SUBGRAFO\n")
        f.write("-" * 80 + "\n")
        f.write(f"- Nodos totales: {comp['nodos_totales']:,}\n")
        f.write(f"- Top-N incluidos: {comp['top_n_incluidos']}\n")
        f.write(f"- Vecinos: {comp['vecinos']:,}\n")
        f.write(f"- Aristas: {comp['aristas']:,}\n")
        f.write(f"- Densidad: {comp['densidad']:.6f} ({comp['densidad']*100:.4f}%)\n\n")

        # 2. Métricas de estructura
        f.write("2. MÉTRICAS DE ESTRUCTURA\n")
        f.write("-" * 80 + "\n")
        f.write(f"- Grado promedio: {estructura['grado_promedio']:.2f}\n")
        if estructura['diametro'] is not None:
            f.write(f"- Diámetro: {estructura['diametro']} (camino más largo)\n")
        else:
            f.write(f"- Diámetro: No calculable (grafo no conexo)\n")
        f.write(f"- Coeficiente clustering promedio: {estructura['clustering_promedio']:.4f}\n")
        f.write(f"- Componentes débilmente conexas: {estructura['componentes_debiles']}\n")
        f.write(f"- Componentes fuertemente conexas: {estructura['componentes_fuertes']}\n\n")

        # 3. Concentración de influencia
        f.write("3. CONCENTRACIÓN DE INFLUENCIA\n")
        f.write("-" * 80 + "\n")
        f.write(f"- {concentracion['pct_aristas_hacia_top_n']:.2f}% de las aristas apuntan hacia Top-{comp['top_n_incluidos']}\n")
        f.write(f"- In-degree promedio de Top-N: {concentracion['in_degree_promedio_top_n']:.2f}\n\n")

        # 4. Patrones observados
        f.write("4. PATRONES IDENTIFICADOS\n")
        f.write("-" * 80 + "\n")

        # Analizar tipo de estructura
        if concentracion['pct_aristas_hacia_top_n'] > 60:
            f.write("- Estructura dominante: Hub-and-Spoke (estrella)\n")
            f.write(f"  * Los Top-{comp['top_n_incluidos']} actúan como super-hubs centrales\n")
            f.write(f"  * Mayoría de nodos apuntan hacia el núcleo ({concentracion['pct_aristas_hacia_top_n']:.1f}%)\n\n")
        elif concentracion['pct_aristas_hacia_top_n'] > 40:
            f.write("- Estructura dominante: Semi-jerárquica\n")
            f.write(f"  * Los Top-{comp['top_n_incluidos']} son importantes pero no dominantes totales\n")
            f.write("  * Existe diversidad de conexiones entre vecinos\n\n")
        else:
            f.write("- Estructura dominante: Distribuida\n")
            f.write("  * Los Top-N no concentran la mayoría de las conexiones\n")
            f.write("  * Estructura más plana con múltiples sub-hubs\n\n")

        # Clustering
        if estructura['clustering_promedio'] > 0.3:
            f.write(f"- Alto clustering ({estructura['clustering_promedio']:.3f}): Los nodos forman comunidades densas\n")
        elif estructura['clustering_promedio'] > 0.1:
            f.write(f"- Clustering moderado ({estructura['clustering_promedio']:.3f}): Algunas comunidades locales\n")
        else:
            f.write(f"- Bajo clustering ({estructura['clustering_promedio']:.3f}): Estructura tipo árbol o cadena\n")

        f.write("\n")

        # 5. Hallazgos clave
        f.write("5. HALLAZGOS CLAVE\n")
        f.write("-" * 80 + "\n")

        if comp['densidad'] < 0.01:
            f.write(f"- Grafo disperso (densidad {comp['densidad']*100:.4f}%): Pocas conexiones relativas\n")
        elif comp['densidad'] > 0.1:
            f.write(f"- Grafo denso (densidad {comp['densidad']*100:.2f}%): Alta interconexión\n")

        if estructura['componentes_debiles'] > 1:
            f.write(f"- Grafo fragmentado en {estructura['componentes_debiles']} componentes\n")
            f.write("  Recomendación: Agregar enlaces puente entre componentes\n")
        else:
            f.write("- Grafo bien conectado (1 componente débil)\n")

        f.write("\n")

        # 6. Recomendaciones visuales
        f.write("6. RECOMENDACIONES BASADAS EN VISUALIZACIÓN\n")
        f.write("-" * 80 + "\n")

        if concentracion['pct_aristas_hacia_top_n'] > 70:
            f.write(f"- PRIORIDAD ALTA: Optimizar nodos Top-{comp['top_n_incluidos']} (concentran >70% influencia)\n")
            f.write("  Acción: Reforzar contenido, monitoreo prioritario, redundancia de enlaces\n\n")

        if estructura['componentes_debiles'] > 1:
            f.write(f"- Fragmentación detectada: {estructura['componentes_debiles']} componentes separados\n")
            f.write("  Acción: Identificar nodos puente y reforzar conexiones inter-componentes\n\n")

        if comp['densidad'] < 0.005:
            f.write("- Grafo muy disperso: Oportunidad de link building interno\n")
            f.write("  Acción: Agregar enlaces desde vecinos hacia Top-N y entre vecinos\n\n")

        f.write("="*80 + "\n")

    logger.info("  Reporte de insights generado exitosamente")


def guardar_subgrafo(G_sub: nx.DiGraph, output_path: str) -> None:
    """
    Guarda el subgrafo en formato pickle para análisis posterior.

    Args:
        G_sub: Subgrafo
        output_path: Ruta de salida
    """
    logger.info(f"Guardando subgrafo en: {output_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(G_sub, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("  Subgrafo guardado exitosamente")


def ejecutar_fase5(config_path: Optional[str] = None) -> nx.DiGraph:
    """
    Ejecuta la FASE 5: Visualización del Subgrafo.

    Args:
        config_path: Ruta a archivo de configuración (None usa default)

    Returns:
        nx.DiGraph: Subgrafo generado
    """
    logger.info("="*80)
    logger.info("FASE 5: VISUALIZACIÓN DEL SUBGRAFO")
    logger.info("="*80)

    # 1. Cargar configuración
    config = cargar_configuracion(config_path)

    base_dir = Path(__file__).resolve().parent.parent
    subgrafo_path = base_dir / config['data']['processed_dir'] / 'subgrafo.gpickle'
    metricas_csv = base_dir / config['outputs']['results_dir'] / 'fase3_metricas_centralidad.csv'
    ranking_csv = base_dir / config['outputs']['results_dir'] / 'fase4_top_20_urls.csv'
    viz_dir = base_dir / config['outputs']['visualizaciones_dir']

    # 2. Cargar grafo completo
    G = cargar_grafo(subgrafo_path)

    # 3. Cargar métricas
    df_metricas = cargar_metricas_csv(metricas_csv)

    # 4. Cargar ranking Top-N
    df_ranking = pd.read_csv(ranking_csv)
    top_n = config['visualizacion']['top_n_subgrafo']
    top_nodes = df_ranking.head(top_n)['nodo'].tolist()

    # 5. Inducir subgrafo
    G_sub = inducir_subgrafo(
        G,
        df_ranking,
        top_n=top_n,
        incluir_vecinos=config['visualizacion']['incluir_vecinos'],
        incluir_vecinos_2do=config['visualizacion']['incluir_vecinos_2do_nivel']
    )

    # 6. Preparar métricas del subgrafo
    metricas = preparar_metricas_subgrafo(G_sub, df_metricas)

    # 7. Generar visualización estática
    viz_estatica_path = viz_dir / 'subgrafo_top_urls.png'
    visualizar_estatico(G_sub, metricas, top_nodes, config, viz_estatica_path)

    # 8. Generar visualización interactiva
    viz_interactiva_path = viz_dir / 'subgrafo_interactivo.html'
    visualizar_interactivo(G_sub, metricas, top_nodes, config, viz_interactiva_path)

    # 9. Analizar patrones del subgrafo
    analisis = analizar_patrones_subgrafo(G_sub, metricas, top_nodes)

    # 10. Generar reporte de insights
    insights_path = viz_dir / 'fase5_insights_visualizacion.txt'
    generar_reporte_insights(G_sub, analisis, top_nodes, insights_path)

    # 11. Guardar subgrafo
    subgrafo_output_path = viz_dir / 'subgrafo_top.gpickle'
    guardar_subgrafo(G_sub, subgrafo_output_path)

    # 12. Guardar análisis en JSON
    analisis_json_path = viz_dir / 'fase5_analisis_subgrafo.json'
    with open(analisis_json_path, 'w', encoding='utf-8') as f:
        json.dump(analisis, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info("="*80)
    logger.info("FASE 5 COMPLETADA")
    logger.info("="*80)
    logger.info(f"Visualizaciones guardadas en: {viz_dir}")

    return G_sub


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Ejecutar
    G_sub = ejecutar_fase5()

    print(f"\n[COMPLETADO] Subgrafo visualizado: {G_sub.number_of_nodes():,} nodos, {G_sub.number_of_edges():,} aristas")
    print(f"\nArchivos generados:")
    print(f"  - subgrafo_top_urls.png (visualización estática)")
    print(f"  - subgrafo_interactivo.html (visualización interactiva)")
    print(f"  - fase5_insights_visualizacion.txt (reporte de insights)")
