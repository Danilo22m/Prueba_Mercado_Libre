"""
FASE 6: Simulación de Propagación de Información
=================================================
Simula propagación usando Independent Cascade Model para evaluar
qué estrategias de selección de semillas son más efectivas.

Autor: Danilo Melo
Fecha: 2026-01-11
"""

import sys
import os
import logging
import yaml
import json
import random
import pickle
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
from collections import defaultdict

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def cargar_configuracion(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Carga configuración desde YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def cargar_grafo_y_metricas(processed_dir: str) -> Tuple[nx.DiGraph, pd.DataFrame]:
    """Carga grafo y métricas de fases anteriores"""
    logger.info("Cargando grafo y métricas de fases anteriores...")

    # Cargar grafo - probar múltiples nombres posibles
    grafo_path = Path(processed_dir) / "subgrafo.gpickle"
    if not grafo_path.exists():
        grafo_path = Path(processed_dir) / "subgrafo_final.gpickle"

    with open(grafo_path, 'rb') as f:
        G = pickle.load(f)
    logger.info(f"Grafo cargado: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")

    # Cargar métricas - buscar en outputs/results primero
    metricas_path = Path("outputs/results") / "fase3_metricas_centralidad.csv"
    if not metricas_path.exists():
        metricas_path = Path(processed_dir) / "fase3_metricas_centralidad.csv"

    df_metricas = pd.read_csv(metricas_path)
    logger.info(f"Métricas cargadas: {len(df_metricas)} nodos")

    return G, df_metricas


def seleccionar_semillas_estrategia(
    df_metricas: pd.DataFrame,
    estrategia: str,
    num_semillas: int = 5
) -> List[int]:
    """
    Selecciona nodos semilla según estrategia

    Estrategias disponibles:
    - pagerank_top5: Top-5 por PageRank
    - in_degree_top5: Top-5 por In-Degree
    - betweenness_top5: Top-5 por Betweenness
    - authority_top5: Top-5 por Authority (HITS)
    - random: 5 nodos aleatorios
    """
    if estrategia == "random":
        semillas = df_metricas['nodo'].sample(n=num_semillas, random_state=42).tolist()
        logger.info(f"[Estrategia: random] Semillas aleatorias seleccionadas: {semillas[:3]}...")

    elif estrategia == "pagerank_top5":
        semillas = df_metricas.nlargest(num_semillas, 'pagerank')['nodo'].tolist()
        logger.info(f"[Estrategia: pagerank_top5] Top-{num_semillas} PageRank: {semillas[:3]}...")

    elif estrategia == "in_degree_top5":
        semillas = df_metricas.nlargest(num_semillas, 'in_degree')['nodo'].tolist()
        logger.info(f"[Estrategia: in_degree_top5] Top-{num_semillas} In-Degree: {semillas[:3]}...")

    elif estrategia == "betweenness_top5":
        semillas = df_metricas.nlargest(num_semillas, 'betweenness')['nodo'].tolist()
        logger.info(f"[Estrategia: betweenness_top5] Top-{num_semillas} Betweenness: {semillas[:3]}...")

    elif estrategia == "authority_top5":
        semillas = df_metricas.nlargest(num_semillas, 'hits_authority')['nodo'].tolist()
        logger.info(f"[Estrategia: authority_top5] Top-{num_semillas} Authority: {semillas[:3]}...")

    else:
        raise ValueError(f"Estrategia desconocida: {estrategia}")

    return semillas


def simular_independent_cascade(
    G: nx.DiGraph,
    semillas: List[int],
    p: float = 0.1,
    T_max: int = 5
) -> Dict[str, Any]:
    """
    Simula propagación usando Independent Cascade Model

    Algoritmo:
    1. t=0: Los nodos semilla se activan
    2. t=1,...,T_max: Cada nodo recién activado intenta activar a sus vecinos
       - Con probabilidad p, un vecino inactivo se activa
       - Cada intento de activación se hace UNA SOLA VEZ
    3. El proceso termina cuando no hay nuevas activaciones o se alcanza T_max

    Parámetros:
    -----------
    G : nx.DiGraph
        Grafo dirigido
    semillas : List[int]
        Nodos semilla iniciales
    p : float
        Probabilidad de activación (0.0-1.0)
    T_max : int
        Número máximo de pasos de simulación

    Retorna:
    --------
    Dict con resultados de la simulación
    """
    # Inicializar
    activos = set(semillas)
    nuevos = set(semillas)
    cobertura_por_paso = [len(semillas)]
    intentos_realizados = set()  # Para no repetir intentos de activación

    for t in range(1, T_max + 1):
        proximos = set()

        # Cada nodo recién activado intenta activar a sus vecinos
        for nodo in nuevos:
            for vecino in G.successors(nodo):
                # Solo intentar si el vecino no está activo y no hemos intentado antes
                edge = (nodo, vecino)
                if vecino not in activos and edge not in intentos_realizados:
                    intentos_realizados.add(edge)

                    # Intentar activación con probabilidad p
                    if random.random() < p:
                        proximos.add(vecino)

        # Si no hay nuevas activaciones, terminar
        if not proximos:
            break

        # Actualizar conjuntos
        activos.update(proximos)
        nuevos = proximos
        cobertura_por_paso.append(len(activos))

    # Calcular resultados
    total_nodos = G.number_of_nodes()
    total_activados = len(activos)
    cobertura_pct = (total_activados / total_nodos) * 100

    return {
        'total_activados': total_activados,
        'cobertura_pct': cobertura_pct,
        'pasos_usados': len(cobertura_por_paso),
        'cobertura_por_paso': cobertura_por_paso,
        'nodos_activados': list(activos)
    }


def comparar_estrategias(
    G: nx.DiGraph,
    df_metricas: pd.DataFrame,
    estrategias: List[str],
    num_semillas: int = 5,
    p: float = 0.1,
    T_max: int = 5,
    num_simulaciones: int = 100
) -> Dict[str, Any]:
    """
    Compara múltiples estrategias de selección de semillas

    Para estrategias determinísticas (top-N), las semillas son fijas pero
    la propagación es estocástica, por lo que promediamos sobre num_simulaciones.

    Para estrategia random, tanto las semillas como la propagación son aleatorias.
    """
    logger.info(f"\n{'='*80}")
    logger.info("COMPARACIÓN DE ESTRATEGIAS DE PROPAGACIÓN")
    logger.info(f"{'='*80}")
    logger.info(f"Probabilidad de activación: {p}")
    logger.info(f"Número de semillas por estrategia: {num_semillas}")
    logger.info(f"Máximo de pasos: {T_max}")
    logger.info(f"Simulaciones por estrategia: {num_simulaciones}")
    logger.info(f"{'='*80}\n")

    resultados = {}

    for estrategia in estrategias:
        logger.info(f"\n[INFO] Evaluando estrategia: {estrategia}")

        # Seleccionar semillas
        semillas = seleccionar_semillas_estrategia(df_metricas, estrategia, num_semillas)

        # Ejecutar múltiples simulaciones
        coberturas = []
        pasos = []
        coberturas_temporales = []

        for sim in tqdm(range(num_simulaciones), desc=f"Simulando {estrategia}"):
            # Para estrategia random, seleccionar semillas diferentes cada vez
            if estrategia == "random":
                semillas = seleccionar_semillas_estrategia(df_metricas, estrategia, num_semillas)

            resultado = simular_independent_cascade(G, semillas, p, T_max)
            coberturas.append(resultado['cobertura_pct'])
            pasos.append(resultado['pasos_usados'])
            coberturas_temporales.append(resultado['cobertura_por_paso'])

        # Calcular estadísticas
        cobertura_promedio = np.mean(coberturas)
        cobertura_std = np.std(coberturas)
        cobertura_mediana = np.median(coberturas)
        pasos_promedio = np.mean(pasos)

        # Calcular cobertura temporal promedio (alinear longitudes)
        max_len = max(len(c) for c in coberturas_temporales)
        coberturas_alineadas = []
        for c in coberturas_temporales:
            # Rellenar con el último valor si la simulación terminó antes
            c_extendida = c + [c[-1]] * (max_len - len(c))
            coberturas_alineadas.append(c_extendida)

        cobertura_temporal_promedio = np.mean(coberturas_alineadas, axis=0).tolist()
        cobertura_temporal_std = np.std(coberturas_alineadas, axis=0).tolist()

        resultados[estrategia] = {
            'semillas': semillas,  # Últimas semillas usadas
            'cobertura_promedio_pct': cobertura_promedio,
            'cobertura_std_pct': cobertura_std,
            'cobertura_mediana_pct': cobertura_mediana,
            'pasos_promedio': pasos_promedio,
            'cobertura_temporal_promedio': cobertura_temporal_promedio,
            'cobertura_temporal_std': cobertura_temporal_std,
            'num_simulaciones': num_simulaciones
        }

        logger.info(f"[INFO] Cobertura promedio: {cobertura_promedio:.2f}% (+/- {cobertura_std:.2f}%)")
        logger.info(f"[INFO] Pasos promedio: {pasos_promedio:.2f}")

    # Ordenar por cobertura promedio
    ranking = sorted(
        resultados.items(),
        key=lambda x: x[1]['cobertura_promedio_pct'],
        reverse=True
    )

    logger.info(f"\n{'='*80}")
    logger.info("RANKING DE ESTRATEGIAS (por cobertura promedio)")
    logger.info(f"{'='*80}")
    for i, (estrategia, datos) in enumerate(ranking, 1):
        logger.info(
            f"{i}. {estrategia}: {datos['cobertura_promedio_pct']:.2f}% "
            f"(+/- {datos['cobertura_std_pct']:.2f}%)"
        )
    logger.info(f"{'='*80}\n")

    return {
        'resultados': resultados,
        'ranking': [(est, datos['cobertura_promedio_pct']) for est, datos in ranking]
    }


def analisis_sensibilidad(
    G: nx.DiGraph,
    df_metricas: pd.DataFrame,
    estrategia_base: str,
    probabilidades: List[float],
    num_semillas: int = 5,
    T_max: int = 5,
    num_simulaciones: int = 100
) -> Dict[str, Any]:
    """
    Analiza cómo varía la cobertura con diferentes probabilidades de activación
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"ANÁLISIS DE SENSIBILIDAD - Estrategia: {estrategia_base}")
    logger.info(f"{'='*80}")
    logger.info(f"Probabilidades a probar: {probabilidades}")
    logger.info(f"Simulaciones por probabilidad: {num_simulaciones}")
    logger.info(f"{'='*80}\n")

    # Seleccionar semillas fijas para comparación justa
    semillas = seleccionar_semillas_estrategia(df_metricas, estrategia_base, num_semillas)
    logger.info(f"Semillas fijas: {semillas[:3]}...")

    resultados = {}

    for p in probabilidades:
        logger.info(f"\n[INFO] Probando probabilidad p={p}")

        coberturas = []
        coberturas_temporales = []

        for sim in tqdm(range(num_simulaciones), desc=f"p={p}"):
            resultado = simular_independent_cascade(G, semillas, p, T_max)
            coberturas.append(resultado['cobertura_pct'])
            coberturas_temporales.append(resultado['cobertura_por_paso'])

        # Calcular estadísticas
        cobertura_promedio = np.mean(coberturas)
        cobertura_std = np.std(coberturas)

        # Cobertura temporal promedio
        max_len = max(len(c) for c in coberturas_temporales)
        coberturas_alineadas = []
        for c in coberturas_temporales:
            c_extendida = c + [c[-1]] * (max_len - len(c))
            coberturas_alineadas.append(c_extendida)

        cobertura_temporal_promedio = np.mean(coberturas_alineadas, axis=0).tolist()

        resultados[p] = {
            'cobertura_promedio_pct': cobertura_promedio,
            'cobertura_std_pct': cobertura_std,
            'cobertura_temporal_promedio': cobertura_temporal_promedio
        }

        logger.info(f"[INFO] Cobertura promedio con p={p}: {cobertura_promedio:.2f}% (+/- {cobertura_std:.2f}%)")

    logger.info(f"\n{'='*80}")
    logger.info("RESUMEN ANÁLISIS DE SENSIBILIDAD")
    logger.info(f"{'='*80}")
    for p in sorted(resultados.keys()):
        logger.info(f"p={p}: {resultados[p]['cobertura_promedio_pct']:.2f}% (+/- {resultados[p]['cobertura_std_pct']:.2f}%)")
    logger.info(f"{'='*80}\n")

    return {
        'estrategia': estrategia_base,
        'semillas': semillas,
        'resultados': resultados
    }


def visualizar_comparacion_estrategias(
    resultados_comparacion: Dict[str, Any],
    output_path: str
):
    """Visualiza comparación de estrategias"""
    resultados = resultados_comparacion['resultados']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Cobertura promedio con barras de error
    ax1 = axes[0]
    estrategias = list(resultados.keys())
    coberturas = [resultados[e]['cobertura_promedio_pct'] for e in estrategias]
    stds = [resultados[e]['cobertura_std_pct'] for e in estrategias]

    colores = plt.cm.Set2(range(len(estrategias)))
    bars = ax1.bar(estrategias, coberturas, yerr=stds, capsize=5, color=colores, alpha=0.8)
    ax1.set_xlabel('Estrategia', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cobertura Promedio (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Comparación de Estrategias de Propagación', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    ax1.grid(axis='y', alpha=0.3)

    # Anotar valores
    for i, (bar, cob, std) in enumerate(zip(bars, coberturas, stds)):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.5,
                f'{cob:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. Evolución temporal de cobertura
    ax2 = axes[1]
    for i, estrategia in enumerate(estrategias):
        temporal = resultados[estrategia]['cobertura_temporal_promedio']
        pasos = list(range(len(temporal)))
        ax2.plot(pasos, temporal, marker='o', linewidth=2, label=estrategia, color=colores[i])

    ax2.set_xlabel('Paso de Simulación', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Nodos Activados (Promedio)', fontsize=12, fontweight='bold')
    ax2.set_title('Evolución Temporal de la Propagación', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"[INFO] Visualización guardada en: {output_path}")
    plt.close()


def visualizar_sensibilidad(
    resultados_sensibilidad: Dict[str, Any],
    output_path: str
):
    """Visualiza análisis de sensibilidad"""
    resultados = resultados_sensibilidad['resultados']
    estrategia = resultados_sensibilidad['estrategia']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Cobertura vs Probabilidad
    ax1 = axes[0]
    probabilidades = sorted(resultados.keys())
    coberturas = [resultados[p]['cobertura_promedio_pct'] for p in probabilidades]
    stds = [resultados[p]['cobertura_std_pct'] for p in probabilidades]

    ax1.errorbar(probabilidades, coberturas, yerr=stds, marker='o', linewidth=2,
                 markersize=8, capsize=5, color='steelblue', label='Cobertura Promedio')
    ax1.fill_between(probabilidades,
                     [c - s for c, s in zip(coberturas, stds)],
                     [c + s for c, s in zip(coberturas, stds)],
                     alpha=0.2, color='steelblue')
    ax1.set_xlabel('Probabilidad de Activación (p)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cobertura (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Análisis de Sensibilidad - {estrategia}', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=10)

    # 2. Evolución temporal para diferentes probabilidades
    ax2 = axes[1]
    colores = plt.cm.viridis(np.linspace(0, 1, len(probabilidades)))
    for i, p in enumerate(probabilidades):
        temporal = resultados[p]['cobertura_temporal_promedio']
        pasos = list(range(len(temporal)))
        ax2.plot(pasos, temporal, marker='o', linewidth=2, label=f'p={p}', color=colores[i])

    ax2.set_xlabel('Paso de Simulación', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Nodos Activados (Promedio)', fontsize=12, fontweight='bold')
    ax2.set_title('Evolución Temporal por Probabilidad', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"[INFO] Visualización de sensibilidad guardada en: {output_path}")
    plt.close()


def generar_insights_propagacion(
    resultados_comparacion: Dict[str, Any],
    resultados_sensibilidad: Dict[str, Any],
    config: Dict[str, Any]
) -> str:
    """Genera análisis textual de los resultados de propagación"""
    insights = []
    insights.append("="*80)
    insights.append("INSIGHTS - SIMULACIÓN DE PROPAGACIÓN DE INFORMACIÓN")
    insights.append("="*80)
    insights.append("")

    # 1. Configuración de simulación
    insights.append("1. CONFIGURACIÓN DE SIMULACIÓN")
    insights.append("-" * 80)
    insights.append(f"Modelo: {config['propagacion']['modelo']}")
    insights.append(f"Probabilidad base: {config['propagacion']['probabilidad']}")
    insights.append(f"Máximo de pasos: {config['propagacion']['max_pasos']}")
    insights.append(f"Simulaciones por estrategia: {config['propagacion']['num_simulaciones']}")
    insights.append(f"Número de semillas: {config['propagacion']['num_semillas']}")
    insights.append("")

    # 2. Ranking de estrategias
    insights.append("2. RANKING DE ESTRATEGIAS")
    insights.append("-" * 80)
    ranking = resultados_comparacion['ranking']
    for i, (estrategia, cobertura) in enumerate(ranking, 1):
        detalles = resultados_comparacion['resultados'][estrategia]
        insights.append(f"{i}. {estrategia}")
        insights.append(f"   - Cobertura promedio: {cobertura:.2f}%")
        insights.append(f"   - Desviación estándar: {detalles['cobertura_std_pct']:.2f}%")
        insights.append(f"   - Cobertura mediana: {detalles['cobertura_mediana_pct']:.2f}%")
        insights.append(f"   - Pasos promedio: {detalles['pasos_promedio']:.2f}")
        insights.append("")

    # 3. Análisis de la mejor estrategia
    insights.append("3. ANÁLISIS DE LA MEJOR ESTRATEGIA")
    insights.append("-" * 80)
    mejor_estrategia, mejor_cobertura = ranking[0]
    peor_estrategia, peor_cobertura = ranking[-1]
    mejora = mejor_cobertura - peor_cobertura

    insights.append(f"Estrategia más efectiva: {mejor_estrategia}")
    insights.append(f"Cobertura: {mejor_cobertura:.2f}%")
    insights.append(f"")
    insights.append(f"Estrategia menos efectiva: {peor_estrategia}")
    insights.append(f"Cobertura: {peor_cobertura:.2f}%")
    insights.append(f"")
    insights.append(f"Mejora relativa: {mejora:.2f} puntos porcentuales")
    insights.append(f"Mejora relativa: {(mejora / peor_cobertura * 100):.1f}%")
    insights.append("")

    # 4. Análisis de sensibilidad
    if resultados_sensibilidad:
        insights.append("4. ANÁLISIS DE SENSIBILIDAD A LA PROBABILIDAD")
        insights.append("-" * 80)
        estrategia_sens = resultados_sensibilidad['estrategia']
        resultados_sens = resultados_sensibilidad['resultados']

        insights.append(f"Estrategia analizada: {estrategia_sens}")
        insights.append("")

        probabilidades = sorted(resultados_sens.keys())
        coberturas = [resultados_sens[p]['cobertura_promedio_pct'] for p in probabilidades]

        insights.append("Cobertura por probabilidad:")
        for p, c in zip(probabilidades, coberturas):
            std = resultados_sens[p]['cobertura_std_pct']
            insights.append(f"  p={p:.2f}: {c:.2f}% (+/- {std:.2f}%)")

        insights.append("")
        insights.append(f"Cobertura mínima (p={min(probabilidades)}): {min(coberturas):.2f}%")
        insights.append(f"Cobertura máxima (p={max(probabilidades)}): {max(coberturas):.2f}%")

        # Calcular elasticidad
        delta_p = max(probabilidades) - min(probabilidades)
        delta_c = max(coberturas) - min(coberturas)
        elasticidad = delta_c / delta_p
        insights.append(f"")
        insights.append(f"Elasticidad promedio: {elasticidad:.2f} puntos porcentuales por 0.01 de incremento en p")
        insights.append("")

    # 5. Recomendaciones
    insights.append("5. RECOMENDACIONES")
    insights.append("-" * 80)
    insights.append(f"[RECOMENDACIÓN 1] Usar estrategia '{mejor_estrategia}' para maximizar alcance")
    insights.append(f"   Esta estrategia alcanza {mejor_cobertura:.2f}% del grafo en promedio.")
    insights.append("")

    if mejor_estrategia != "random":
        insights.append(f"[RECOMENDACIÓN 2] La estrategia '{mejor_estrategia}' supera significativamente")
        insights.append(f"   a la selección aleatoria, demostrando el valor de las métricas de centralidad.")
    insights.append("")

    insights.append(f"[RECOMENDACIÓN 3] Ajustar probabilidad de activación según objetivo:")
    if resultados_sensibilidad:
        insights.append(f"   - Para propagación conservadora: p={min(probabilidades)}")
        insights.append(f"   - Para propagación agresiva: p={max(probabilidades)}")
    insights.append("")

    insights.append("="*80)

    return "\n".join(insights)


def main():
    """Función principal de FASE 6"""
    logger.info("="*80)
    logger.info("FASE 6: SIMULACIÓN DE PROPAGACIÓN DE INFORMACIÓN")
    logger.info("="*80)
    logger.info("")

    # 1. Cargar configuración
    config = cargar_configuracion()
    logger.info("[INFO] Configuración cargada correctamente")

    # 2. Crear directorios de salida
    output_dir = Path(config['outputs']['propagacion_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[INFO] Directorio de salida: {output_dir}")

    # 3. Cargar grafo y métricas
    G, df_metricas = cargar_grafo_y_metricas(config['data']['processed_dir'])

    # 4. Comparar estrategias de propagación
    logger.info("\n" + "="*80)
    logger.info("FASE 6.1: COMPARACIÓN DE ESTRATEGIAS")
    logger.info("="*80 + "\n")

    resultados_comparacion = comparar_estrategias(
        G=G,
        df_metricas=df_metricas,
        estrategias=config['propagacion']['estrategias'],
        num_semillas=config['propagacion']['num_semillas'],
        p=config['propagacion']['probabilidad'],
        T_max=config['propagacion']['max_pasos'],
        num_simulaciones=config['propagacion']['num_simulaciones']
    )

    # Guardar resultados de comparación
    output_path_json = output_dir / "fase6_comparacion_estrategias.json"
    with open(output_path_json, 'w', encoding='utf-8') as f:
        json.dump(resultados_comparacion, f, indent=2, ensure_ascii=False)
    logger.info(f"[INFO] Resultados de comparación guardados en: {output_path_json}")

    # Visualizar comparación
    output_path_viz = output_dir / "propagacion_comparacion_estrategias.png"
    visualizar_comparacion_estrategias(resultados_comparacion, str(output_path_viz))

    # 5. Análisis de sensibilidad (si está habilitado)
    resultados_sensibilidad = None
    if config['propagacion'].get('analisis_sensibilidad', False):
        logger.info("\n" + "="*80)
        logger.info("FASE 6.2: ANÁLISIS DE SENSIBILIDAD")
        logger.info("="*80 + "\n")

        # Usar la mejor estrategia para análisis de sensibilidad
        mejor_estrategia = resultados_comparacion['ranking'][0][0]

        resultados_sensibilidad = analisis_sensibilidad(
            G=G,
            df_metricas=df_metricas,
            estrategia_base=mejor_estrategia,
            probabilidades=config['propagacion']['probabilidades_test'],
            num_semillas=config['propagacion']['num_semillas'],
            T_max=config['propagacion']['max_pasos'],
            num_simulaciones=config['propagacion']['num_simulaciones']
        )

        # Guardar resultados de sensibilidad
        output_path_sens_json = output_dir / "fase6_analisis_sensibilidad.json"
        with open(output_path_sens_json, 'w', encoding='utf-8') as f:
            json.dump(resultados_sensibilidad, f, indent=2, ensure_ascii=False)
        logger.info(f"[INFO] Resultados de sensibilidad guardados en: {output_path_sens_json}")

        # Visualizar sensibilidad
        output_path_sens_viz = output_dir / "propagacion_sensibilidad.png"
        visualizar_sensibilidad(resultados_sensibilidad, str(output_path_sens_viz))

    # 6. Generar insights
    logger.info("\n" + "="*80)
    logger.info("GENERANDO INSIGHTS FINALES")
    logger.info("="*80 + "\n")

    insights = generar_insights_propagacion(
        resultados_comparacion=resultados_comparacion,
        resultados_sensibilidad=resultados_sensibilidad,
        config=config
    )

    # Guardar insights
    output_path_insights = output_dir / "fase6_insights_propagacion.txt"
    with open(output_path_insights, 'w', encoding='utf-8') as f:
        f.write(insights)
    logger.info(f"[INFO] Insights guardados en: {output_path_insights}")

    # Mostrar insights en consola
    print("\n" + insights)

    logger.info("\n" + "="*80)
    logger.info("FASE 6 COMPLETADA EXITOSAMENTE")
    logger.info("="*80)
    logger.info(f"\nArchivos generados en: {output_dir}")
    logger.info("  - fase6_comparacion_estrategias.json")
    logger.info("  - propagacion_comparacion_estrategias.png")
    if resultados_sensibilidad:
        logger.info("  - fase6_analisis_sensibilidad.json")
        logger.info("  - propagacion_sensibilidad.png")
    logger.info("  - fase6_insights_propagacion.txt")
    logger.info("")


if __name__ == "__main__":
    main()
