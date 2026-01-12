#!/usr/bin/env python3
"""
FASE 2: Análisis del Grafo Dirigido

Este módulo carga el subgrafo generado en FASE 1 y calcula estadísticas
descriptivas para entender su estructura:
1. Cargar subgrafo desde pickle
2. Calcular estadísticas básicas (N, E, densidad)
3. Analizar distribución de grados (in/out)
4. Identificar nodos especiales (huérfanos, sinks)
5. Guardar estadísticas en JSON

Autor: Danilo Melo
Fecha: 2026-01-11
"""

import logging
import networkx as nx
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
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


def cargar_subgrafo(file_path: str) -> nx.DiGraph:
    """
    Carga el subgrafo desde archivo pickle.

    Args:
        file_path: Ruta al archivo pickle

    Returns:
        nx.DiGraph: Grafo cargado
    """
    logger.info(f"Cargando subgrafo desde: {file_path}")

    with open(file_path, 'rb') as f:
        G = pickle.load(f)

    logger.info(f"✓ Subgrafo cargado exitosamente")

    return G


def calcular_estadisticas_basicas(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Calcula estadísticas básicas del grafo.

    Args:
        G: Grafo dirigido

    Returns:
        dict: Estadísticas básicas
    """
    logger.info("Calculando estadísticas básicas...")

    N = G.number_of_nodes()
    E = G.number_of_edges()

    # Densidad: E / (N * (N-1))
    max_edges = N * (N - 1) if N > 1 else 1
    density = E / max_edges

    stats = {
        'num_nodos': N,
        'num_aristas': E,
        'densidad': density,
        'densidad_porcentaje': density * 100
    }

    logger.info(f"  Nodos (N): {N:,}")
    logger.info(f"  Aristas (E): {E:,}")
    logger.info(f"  Densidad: {density:.6f} ({density*100:.4f}%)")

    return stats


def analizar_distribucion_grados(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Analiza la distribución de grados (in-degree, out-degree, total).

    Args:
        G: Grafo dirigido

    Returns:
        dict: Estadísticas de distribución de grados
    """
    logger.info("Analizando distribución de grados...")

    # In-degree
    in_degrees = dict(G.in_degree())
    in_degree_values = list(in_degrees.values())

    # Out-degree
    out_degrees = dict(G.out_degree())
    out_degree_values = list(out_degrees.values())

    # Total degree
    total_degrees = dict(G.degree())
    total_degree_values = list(total_degrees.values())

    stats = {
        'in_degree': {
            'min': min(in_degree_values) if in_degree_values else 0,
            'max': max(in_degree_values) if in_degree_values else 0,
            'promedio': sum(in_degree_values) / len(in_degree_values) if in_degree_values else 0,
        },
        'out_degree': {
            'min': min(out_degree_values) if out_degree_values else 0,
            'max': max(out_degree_values) if out_degree_values else 0,
            'promedio': sum(out_degree_values) / len(out_degree_values) if out_degree_values else 0,
        },
        'total_degree': {
            'min': min(total_degree_values) if total_degree_values else 0,
            'max': max(total_degree_values) if total_degree_values else 0,
            'promedio': sum(total_degree_values) / len(total_degree_values) if total_degree_values else 0,
        }
    }

    logger.info(f"  In-degree:  min={stats['in_degree']['min']}, max={stats['in_degree']['max']:,}, avg={stats['in_degree']['promedio']:.2f}")
    logger.info(f"  Out-degree: min={stats['out_degree']['min']}, max={stats['out_degree']['max']:,}, avg={stats['out_degree']['promedio']:.2f}")
    logger.info(f"  Total:      min={stats['total_degree']['min']}, max={stats['total_degree']['max']:,}, avg={stats['total_degree']['promedio']:.2f}")

    return stats


def identificar_nodos_especiales(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Identifica nodos con características especiales.

    Args:
        G: Grafo dirigido

    Returns:
        dict: Información sobre nodos especiales
    """
    logger.info("Identificando nodos especiales...")

    # Nodos huérfanos (in-degree = 0)
    orphans = [node for node in G.nodes() if G.in_degree(node) == 0]
    num_orphans = len(orphans)
    pct_orphans = (num_orphans / G.number_of_nodes() * 100) if G.number_of_nodes() > 0 else 0

    # Nodos sink (out-degree = 0)
    sinks = [node for node in G.nodes() if G.out_degree(node) == 0]
    num_sinks = len(sinks)
    pct_sinks = (num_sinks / G.number_of_nodes() * 100) if G.number_of_nodes() > 0 else 0

    # Nodos aislados (degree = 0)
    isolated = list(nx.isolates(G))
    num_isolated = len(isolated)
    pct_isolated = (num_isolated / G.number_of_nodes() * 100) if G.number_of_nodes() > 0 else 0

    stats = {
        'huerfanos': {
            'cantidad': num_orphans,
            'porcentaje': pct_orphans
        },
        'sinks': {
            'cantidad': num_sinks,
            'porcentaje': pct_sinks
        },
        'aislados': {
            'cantidad': num_isolated,
            'porcentaje': pct_isolated
        }
    }

    logger.info(f"  Nodos huérfanos (in-degree=0): {num_orphans:,} ({pct_orphans:.2f}%)")
    logger.info(f"  Nodos sink (out-degree=0): {num_sinks:,} ({pct_sinks:.2f}%)")
    logger.info(f"  Nodos aislados (degree=0): {num_isolated:,} ({pct_isolated:.2f}%)")

    return stats


def analizar_componentes(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Analiza componentes débilmente y fuertemente conexas.

    Args:
        G: Grafo dirigido

    Returns:
        dict: Información sobre componentes
    """
    logger.info("Analizando componentes conexas...")

    # Weakly Connected Components
    wccs = list(nx.weakly_connected_components(G))
    num_wccs = len(wccs)
    largest_wcc_size = max([len(wcc) for wcc in wccs]) if wccs else 0

    # Strongly Connected Components
    sccs = list(nx.strongly_connected_components(G))
    num_sccs = len(sccs)
    largest_scc_size = max([len(scc) for scc in sccs]) if sccs else 0

    stats = {
        'weakly_connected': {
            'num_componentes': num_wccs,
            'tamano_mayor': largest_wcc_size,
            'porcentaje_mayor': (largest_wcc_size / G.number_of_nodes() * 100) if G.number_of_nodes() > 0 else 0
        },
        'strongly_connected': {
            'num_componentes': num_sccs,
            'tamano_mayor': largest_scc_size,
            'porcentaje_mayor': (largest_scc_size / G.number_of_nodes() * 100) if G.number_of_nodes() > 0 else 0
        }
    }

    logger.info(f"  Componentes débiles (WCC): {num_wccs}, mayor={largest_wcc_size:,} ({stats['weakly_connected']['porcentaje_mayor']:.2f}%)")
    logger.info(f"  Componentes fuertes (SCC): {num_sccs}, mayor={largest_scc_size:,} ({stats['strongly_connected']['porcentaje_mayor']:.2f}%)")

    return stats


def validar_calidad_grafo(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida si las métricas del grafo son adecuadas para continuar con las siguientes fases.

    Args:
        stats: Estadísticas del grafo

    Returns:
        dict: Resultados de validación con recomendaciones
    """
    logger.info("")
    logger.info("VALIDACIÓN DE CALIDAD DEL GRAFO:")
    logger.info("-" * 80)

    validaciones = []
    es_valido = True

    # 1. Validar tamaño mínimo
    num_nodos = stats['basicas']['num_nodos']
    num_aristas = stats['basicas']['num_aristas']

    if num_nodos < 100:
        validaciones.append({
            'metrica': 'Número de nodos',
            'valor': num_nodos,
            'estado': 'CRÍTICO',
            'mensaje': f'Muy pocos nodos ({num_nodos}). Se recomienda al menos 100 nodos para análisis significativo.'
        })
        es_valido = False
    elif num_nodos < 1000:
        validaciones.append({
            'metrica': 'Número de nodos',
            'valor': num_nodos,
            'estado': 'ADVERTENCIA',
            'mensaje': f'Pocos nodos ({num_nodos:,}). Los resultados pueden no ser muy representativos.'
        })
    else:
        validaciones.append({
            'metrica': 'Número de nodos',
            'valor': num_nodos,
            'estado': 'BUENO',
            'mensaje': f'Suficientes nodos ({num_nodos:,}) para análisis robusto.'
        })

    # 2. Validar densidad
    densidad = stats['basicas']['densidad']

    if densidad < 0.00001:  # Muy disperso
        validaciones.append({
            'metrica': 'Densidad',
            'valor': f"{densidad:.6f}",
            'estado': 'ADVERTENCIA',
            'mensaje': f'Grafo muy disperso ({densidad*100:.4f}%). PageRank y HITS pueden tener baja varianza.'
        })
    elif densidad > 0.1:  # Muy denso
        validaciones.append({
            'metrica': 'Densidad',
            'valor': f"{densidad:.6f}",
            'estado': 'ADVERTENCIA',
            'mensaje': f'Grafo muy denso ({densidad*100:.2f}%). Betweenness será costoso computacionalmente.'
        })
    else:
        validaciones.append({
            'metrica': 'Densidad',
            'valor': f"{densidad:.6f}",
            'estado': 'BUENO',
            'mensaje': f'Densidad adecuada ({densidad*100:.4f}%) para análisis de centralidad.'
        })

    # 3. Validar distribución de grados
    max_in_degree = stats['distribucion_grados']['in_degree']['max']
    avg_in_degree = stats['distribucion_grados']['in_degree']['promedio']

    if max_in_degree < 10:
        validaciones.append({
            'metrica': 'In-degree máximo',
            'valor': max_in_degree,
            'estado': 'CRÍTICO',
            'mensaje': f'In-degree máximo muy bajo ({max_in_degree}). No hay nodos claramente influyentes.'
        })
        es_valido = False
    elif max_in_degree > avg_in_degree * 10:  # Hay nodos hub claros
        validaciones.append({
            'metrica': 'In-degree máximo',
            'valor': max_in_degree,
            'estado': 'EXCELENTE',
            'mensaje': f'Existe diversidad clara de grados (max={max_in_degree:,}, avg={avg_in_degree:.2f}). Ideal para análisis de autoridad.'
        })
    else:
        validaciones.append({
            'metrica': 'In-degree máximo',
            'valor': max_in_degree,
            'estado': 'BUENO',
            'mensaje': f'Distribución de grados adecuada (max={max_in_degree:,}, avg={avg_in_degree:.2f}).'
        })

    # 4. Validar nodos huérfanos
    pct_huerfanos = stats['nodos_especiales']['huerfanos']['porcentaje']

    if pct_huerfanos > 50:
        validaciones.append({
            'metrica': 'Nodos huérfanos',
            'valor': f"{pct_huerfanos:.2f}%",
            'estado': 'CRÍTICO',
            'mensaje': f'Demasiados nodos huérfanos ({pct_huerfanos:.2f}%). El grafo está muy fragmentado.'
        })
        es_valido = False
    elif pct_huerfanos > 20:
        validaciones.append({
            'metrica': 'Nodos huérfanos',
            'valor': f"{pct_huerfanos:.2f}%",
            'estado': 'ADVERTENCIA',
            'mensaje': f'Muchos nodos huérfanos ({pct_huerfanos:.2f}%). Considerar estrategia de link building.'
        })
    else:
        validaciones.append({
            'metrica': 'Nodos huérfanos',
            'valor': f"{pct_huerfanos:.2f}%",
            'estado': 'BUENO',
            'mensaje': f'Bajo porcentaje de nodos huérfanos ({pct_huerfanos:.2f}%).'
        })

    # 5. Validar conectividad (si existe)
    if stats.get('componentes'):
        num_wccs = stats['componentes']['weakly_connected']['num_componentes']
        pct_mayor_wcc = stats['componentes']['weakly_connected']['porcentaje_mayor']

        if num_wccs > 10:
            validaciones.append({
                'metrica': 'Conectividad (WCC)',
                'valor': f"{num_wccs} componentes",
                'estado': 'ADVERTENCIA',
                'mensaje': f'Grafo fragmentado en {num_wccs} componentes. Considerar usar solo el componente principal.'
            })
        elif pct_mayor_wcc > 95:
            validaciones.append({
                'metrica': 'Conectividad (WCC)',
                'valor': f"{pct_mayor_wcc:.2f}% en componente principal",
                'estado': 'EXCELENTE',
                'mensaje': f'Grafo bien conectado: {pct_mayor_wcc:.2f}% de nodos en componente principal.'
            })
        else:
            validaciones.append({
                'metrica': 'Conectividad (WCC)',
                'valor': f"{num_wccs} componentes",
                'estado': 'BUENO',
                'mensaje': f'Conectividad aceptable con {num_wccs} componentes.'
            })

    # Imprimir resultados
    for val in validaciones:
        prefijo = {
            'EXCELENTE': '[EXCELENTE]',
            'BUENO': '[BUENO]',
            'ADVERTENCIA': '[ADVERTENCIA]',
            'CRÍTICO': '[CRÍTICO]'
        }.get(val['estado'], '')

        logger.info(f"{prefijo} {val['metrica']}: {val['valor']}")
        logger.info(f"   {val['mensaje']}")

    logger.info("")
    if es_valido:
        logger.info("VALIDACIÓN: El grafo es APTO para continuar con FASE 3 (Métricas de Centralidad)")
    else:
        logger.warning("VALIDACIÓN: El grafo tiene PROBLEMAS CRÍTICOS. Se recomienda ajustar FASE 1.")

    return {
        'es_valido': es_valido,
        'validaciones': validaciones
    }


def guardar_estadisticas(stats: Dict[str, Any], output_path: str) -> None:
    """
    Guarda las estadísticas en formato JSON.

    Args:
        stats: Diccionario con estadísticas
        output_path: Ruta de salida
    """
    logger.info(f"Guardando estadísticas en: {output_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info("✓ Estadísticas guardadas exitosamente")


def ejecutar_fase2(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Ejecuta la FASE 2: Análisis del Grafo Dirigido.

    Args:
        config_path: Ruta a archivo de configuración (None usa default)

    Returns:
        dict: Estadísticas consolidadas
    """
    logger.info("="*80)
    logger.info("FASE 2: ANÁLISIS DEL GRAFO DIRIGIDO")
    logger.info("="*80)

    # 1. Cargar configuración
    config = cargar_configuracion(config_path)

    base_dir = Path(__file__).resolve().parent.parent
    processed_dir = base_dir / config['data']['processed_dir']
    results_dir = base_dir / config['outputs']['results_dir']

    # 2. Cargar subgrafo
    subgrafo_path = processed_dir / 'subgrafo.gpickle'
    G = cargar_subgrafo(subgrafo_path)

    logger.info("")
    logger.info("ESTADÍSTICAS DEL GRAFO:")
    logger.info("-" * 80)

    # 3. Calcular estadísticas básicas
    stats_basicas = calcular_estadisticas_basicas(G)

    logger.info("")
    logger.info("DISTRIBUCIÓN DE GRADOS:")
    logger.info("-" * 80)

    # 4. Analizar distribución de grados
    stats_grados = analizar_distribucion_grados(G)

    logger.info("")
    logger.info("NODOS ESPECIALES:")
    logger.info("-" * 80)

    # 5. Identificar nodos especiales
    stats_especiales = identificar_nodos_especiales(G)

    # 6. Analizar componentes (opcional)
    stats_componentes = {}
    if config.get('analisis', {}).get('calcular_componentes', True):
        logger.info("")
        logger.info("COMPONENTES CONEXAS:")
        logger.info("-" * 80)
        stats_componentes = analizar_componentes(G)

    # 7. Consolidar todas las estadísticas
    stats_completas = {
        'basicas': stats_basicas,
        'distribucion_grados': stats_grados,
        'nodos_especiales': stats_especiales,
        'componentes': stats_componentes
    }

    # 8. Validar calidad del grafo
    resultado_validacion = validar_calidad_grafo(stats_completas)
    stats_completas['validacion'] = resultado_validacion

    # 9. Guardar estadísticas
    output_path = results_dir / 'fase2_estadisticas.json'
    guardar_estadisticas(stats_completas, output_path)

    logger.info("")
    logger.info("="*80)
    logger.info("FASE 2 COMPLETADA")
    logger.info("="*80)

    return stats_completas


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Ejecutar
    stats = ejecutar_fase2()

    print(f"\n✓ Análisis completado. Ver resultados en: outputs/results/fase2_estadisticas.json")
