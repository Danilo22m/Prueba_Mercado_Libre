#!/usr/bin/env python3
"""
FASE 7: Recomendaciones Accionables para E-Commerce
====================================================

Genera recomendaciones concretas basadas en el análisis de grafos de URLs,
optimizadas para equipos de e-commerce (Product, Marketing, Tech).

Autor: Danilo Melo
Fecha: 2026-01-11
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import yaml
import pandas as pd
import numpy as np


# =============================================================================
# CONFIGURACIÓN DE LOGGING
# =============================================================================
def setup_logging(config: Dict) -> logging.Logger:
    """Configura el sistema de logging."""
    log_config = config.get('logging', {})
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '[%(asctime)s] %(levelname)s - %(message)s'),
        datefmt=log_config.get('date_format', '%Y-%m-%d %H:%M:%S')
    )
    return logging.getLogger(__name__)


# =============================================================================
# CARGA DE RESULTADOS DE FASES PREVIAS
# =============================================================================
def cargar_resultados_fase2(ruta: str, logger: logging.Logger) -> Dict:
    """Carga estadísticas del grafo (FASE 2)."""
    logger.info(f"Cargando resultados de FASE 2: {ruta}")
    with open(ruta, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"  - Nodos totales: {data['basicas']['num_nodos']}")
    logger.info(f"  - Aristas totales: {data['basicas']['num_aristas']}")
    logger.info(f"  - Densidad: {data['basicas']['densidad']:.6f}")
    return data


def cargar_resultados_fase3(ruta: str, logger: logging.Logger) -> pd.DataFrame:
    """Carga métricas de centralidad (FASE 3)."""
    logger.info(f"Cargando resultados de FASE 3: {ruta}")
    df = pd.read_csv(ruta)
    logger.info(f"  - Métricas cargadas para {len(df)} nodos")
    logger.info(f"  - Columnas: {', '.join(df.columns.tolist())}")
    return df


def cargar_resultados_fase4(ruta: str, logger: logging.Logger) -> Dict:
    """Carga análisis de roles del Top-20 (FASE 4)."""
    logger.info(f"Cargando resultados de FASE 4: {ruta}")
    with open(ruta, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"  - Top-{len(data['top_nodos'])} nodos analizados")
    logger.info(f"  - Diversidad de roles: {data['resumen']['diversidad_roles']}")
    return data


def cargar_resultados_fase6(ruta: str, logger: logging.Logger) -> Dict:
    """Carga resultados de simulación de propagación (FASE 6)."""
    logger.info(f"Cargando resultados de FASE 6: {ruta}")
    with open(ruta, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"  - Estrategias comparadas: {len(data['resultados'])}")
    # La mejor estrategia es la primera del ranking
    mejor_estrategia_nombre = data['ranking'][0][0]
    mejor_cobertura = data['ranking'][0][1]
    logger.info(f"  - Mejor estrategia: {mejor_estrategia_nombre} (cobertura: {mejor_cobertura:.2f}%)")
    return data


# =============================================================================
# ANÁLISIS Y DETECCIÓN DE OPORTUNIDADES
# =============================================================================
def analizar_nodos_huerfanos(df_metricas: pd.DataFrame, logger: logging.Logger) -> Dict:
    """
    Detecta nodos con bajo in-degree (páginas sin enlaces internos).

    Oportunidad: Link Building Interno
    """
    logger.info("[ANÁLISIS] Detectando nodos huérfanos (bajo in-degree)...")

    total_nodos = len(df_metricas)

    # Nodos con in-degree = 0
    huerfanos_absolutos = df_metricas[df_metricas['in_degree'] == 0]

    # Nodos con in-degree <= 2 (muy poco enlazados)
    huerfanos_relativos = df_metricas[df_metricas['in_degree'] <= 2]

    pct_huerfanos_abs = (len(huerfanos_absolutos) / total_nodos) * 100
    pct_huerfanos_rel = (len(huerfanos_relativos) / total_nodos) * 100

    logger.info(f"  - Nodos huérfanos absolutos (in-degree=0): {len(huerfanos_absolutos)} ({pct_huerfanos_abs:.2f}%)")
    logger.info(f"  - Nodos poco enlazados (in-degree<=2): {len(huerfanos_relativos)} ({pct_huerfanos_rel:.2f}%)")

    # Top-10 nodos huérfanos con mayor out-degree (páginas que enlazan a otras pero no reciben)
    top_huerfanos = huerfanos_absolutos.nlargest(10, 'out_degree')[['nodo', 'out_degree', 'pagerank']].to_dict('records')

    return {
        'total_nodos': total_nodos,
        'huerfanos_absolutos': len(huerfanos_absolutos),
        'huerfanos_relativos': len(huerfanos_relativos),
        'pct_huerfanos_absolutos': pct_huerfanos_abs,
        'pct_huerfanos_relativos': pct_huerfanos_rel,
        'top_huerfanos': top_huerfanos
    }


def analizar_concentracion_pagerank(df_metricas: pd.DataFrame, top_n: int, logger: logging.Logger) -> Dict:
    """
    Analiza la concentración de PageRank en los top-N nodos.

    Oportunidad: Diversificación de Autoridad
    """
    logger.info(f"[ANÁLISIS] Analizando concentración de PageRank en Top-{top_n}...")

    df_sorted = df_metricas.sort_values('pagerank', ascending=False)

    pr_total = df_metricas['pagerank'].sum()
    pr_top_n = df_sorted.head(top_n)['pagerank'].sum()
    concentracion = (pr_top_n / pr_total) * 100

    logger.info(f"  - PageRank total: {pr_total:.6f}")
    logger.info(f"  - PageRank en Top-{top_n}: {pr_top_n:.6f}")
    logger.info(f"  - Concentración: {concentracion:.2f}%")

    top_nodos = df_sorted.head(top_n)[['nodo', 'pagerank', 'in_degree', 'out_degree']].to_dict('records')

    return {
        'top_n': top_n,
        'pagerank_total': pr_total,
        'pagerank_top_n': pr_top_n,
        'concentracion_pct': concentracion,
        'top_nodos': top_nodos
    }


def analizar_mejor_estrategia_propagacion(resultados_fase6: Dict, logger: logging.Logger) -> Dict:
    """
    Identifica la mejor estrategia de propagación basada en cobertura.

    Oportunidad: Optimización de Campañas Virales
    """
    logger.info("[ANÁLISIS] Identificando mejor estrategia de propagación...")

    # La mejor estrategia está en el ranking (primera posición)
    mejor_estrategia_nombre = resultados_fase6['ranking'][0][0]
    mejor_cobertura = resultados_fase6['ranking'][0][1]

    # Obtener datos completos de la mejor estrategia
    datos_mejor = resultados_fase6['resultados'][mejor_estrategia_nombre]

    logger.info(f"  - Mejor estrategia: {mejor_estrategia_nombre}")
    logger.info(f"  - Cobertura promedio: {mejor_cobertura:.2f}%")
    logger.info(f"  - Semillas: {datos_mejor['semillas'][:3]}...")

    # Construir ranking de estrategias
    ranking_estrategias = [
        {'nombre': nombre, 'cobertura': cobertura}
        for nombre, cobertura in resultados_fase6['ranking']
    ]

    return {
        'mejor_estrategia': mejor_estrategia_nombre,
        'cobertura_promedio': mejor_cobertura,
        'nodos_activados_promedio': mejor_cobertura * 20,  # Estimación: 2000 nodos * cobertura%
        'semillas_mejor_estrategia': datos_mejor['semillas'],
        'ranking_estrategias': ranking_estrategias
    }


def analizar_roles_autoridades(resultados_fase4: Dict, df_metricas: pd.DataFrame, logger: logging.Logger) -> Dict:
    """
    Identifica páginas con rol de Autoridad para potenciar.

    Oportunidad: Potenciar Páginas Autoridad
    """
    logger.info("[ANÁLISIS] Analizando roles de autoridades...")

    # Extraer nodos del top_nodos
    top_nodos = resultados_fase4['top_nodos']

    # Filtrar nodos que tengan "Autoridad" en el rol principal o roles secundarios
    autoridades = []
    for nodo_info in top_nodos:
        rol_principal = nodo_info.get('rol_principal', '')
        roles_secundarios = nodo_info.get('roles_secundarios', [])

        # Verificar si "Autoridad" está en alguno de los roles
        if 'Autoridad' in rol_principal or any('Autoridad' in r for r in roles_secundarios):
            autoridades.append(nodo_info)

    logger.info(f"  - Autoridades identificadas: {len(autoridades)}")

    # Enriquecer con métricas adicionales (tomar Top-10)
    autoridades_enriquecidas = []
    for auth in autoridades[:10]:
        autoridades_enriquecidas.append({
            'nodo': auth['nodo'],
            'rol': auth['rol_principal'],
            'pagerank': auth['metricas']['pagerank'],
            'in_degree': auth['metricas']['in_degree'],
            'authority': auth['metricas']['hits_authority'],
            'betweenness': auth['metricas']['betweenness']
        })

    return {
        'total_autoridades': len(autoridades),
        'top_autoridades': autoridades_enriquecidas
    }


# =============================================================================
# GENERACIÓN DE ACCIONABLES
# =============================================================================
def generar_accionable_link_building(analisis_huerfanos: Dict, config: Dict) -> Dict:
    """
    Accionable 1: Link Building Interno para Nodos Huérfanos
    """
    umbral = config['recomendaciones']['umbrales']['min_nodos_huerfanos_pct']

    # Determinar severidad
    pct_huerfanos = analisis_huerfanos['pct_huerfanos_absolutos']
    if pct_huerfanos >= umbral:
        prioridad = "ALTA"
        impacto_estimado = "Alto"
    else:
        prioridad = "MEDIA"
        impacto_estimado = "Medio"

    accionable = {
        'id': 'ACT-001',
        'titulo': 'Implementar Link Building Interno para Páginas Huérfanas',
        'descripcion': (
            f"Se detectaron {analisis_huerfanos['huerfanos_absolutos']} páginas "
            f"({pct_huerfanos:.2f}%) sin enlaces internos entrantes. "
            "Esto reduce su visibilidad en crawleos y su PageRank."
        ),
        'categoria': 'SEO Interno',
        'prioridad': prioridad,
        'impacto_estimado': impacto_estimado,
        'esfuerzo_estimado': 'Medio',
        'roi_esperado': 'Alto',

        'acciones_concretas': [
            f"Identificar las {min(50, analisis_huerfanos['huerfanos_absolutos'])} páginas huérfanas prioritarias",
            "Agregar enlaces contextuales desde páginas relacionadas de alta autoridad",
            "Incluir enlaces en breadcrumbs, menús de navegación y secciones 'Relacionados'",
            "Monitorear incremento de PageRank en páginas enlazadas"
        ],

        'kpis': [
            {
                'metrica': 'Páginas huérfanas (in-degree=0)',
                'valor_actual': analisis_huerfanos['huerfanos_absolutos'],
                'objetivo': max(0, analisis_huerfanos['huerfanos_absolutos'] - 50),
                'plazo': '2 semanas'
            },
            {
                'metrica': 'PageRank promedio de páginas objetivo',
                'valor_actual': 'Baseline actual',
                'objetivo': '+20% vs baseline',
                'plazo': '4 semanas'
            }
        ],

        'recursos_necesarios': [
            'Equipo de Contenidos (priorización de páginas)',
            'Equipo de Desarrollo (implementación de enlaces)',
            'Herramienta de análisis de grafos (validación post-implementación)'
        ],

        'paginas_ejemplo': [
            {'nodo': p['nodo'], 'out_degree': int(p['out_degree'])}
            for p in analisis_huerfanos['top_huerfanos'][:5]
        ]
    }

    return accionable


def generar_accionable_propagacion(analisis_propagacion: Dict) -> Dict:
    """
    Accionable 2: Optimizar Campañas usando Estrategia de Propagación Óptima
    """
    mejor_estrategia = analisis_propagacion['mejor_estrategia']
    cobertura = analisis_propagacion['cobertura_promedio']

    # Mapeo de nombre técnico a descripción
    estrategia_map = {
        'pagerank_top5': 'PageRank',
        'in_degree_top5': 'In-Degree',
        'betweenness_top5': 'Betweenness',
        'authority_top5': 'Authority (HITS)',
        'random': 'Aleatorio'
    }

    estrategia_legible = estrategia_map.get(mejor_estrategia, mejor_estrategia)

    accionable = {
        'id': 'ACT-002',
        'titulo': f'Optimizar Campañas Virales con Estrategia {estrategia_legible}',
        'descripcion': (
            f"La estrategia '{estrategia_legible}' alcanzó {cobertura:.2f}% de cobertura "
            f"en simulaciones de propagación (modelo Independent Cascade). "
            "Usar estos nodos como semillas maximiza el alcance viral."
        ),
        'categoria': 'Marketing Viral',
        'prioridad': 'ALTA',
        'impacto_estimado': 'Muy Alto',
        'esfuerzo_estimado': 'Bajo',
        'roi_esperado': 'Muy Alto',

        'acciones_concretas': [
            f"Identificar las páginas top según {estrategia_legible}",
            "Ejecutar campañas de promoción inicial en estas páginas (banners, emails, push notifications)",
            "Implementar mecanismos de sharing fácil (botones sociales, referrals)",
            "Medir alcance viral post-campaña (vistas, shares, conversiones)"
        ],

        'kpis': [
            {
                'metrica': 'Cobertura viral (% usuarios alcanzados)',
                'valor_actual': 'Baseline pre-campaña',
                'objetivo': f'+{int(cobertura)}% vs baseline',
                'plazo': '1 mes post-campaña'
            },
            {
                'metrica': 'Tasa de viralización (shares/usuario)',
                'valor_actual': 'Baseline histórico',
                'objetivo': '+30% vs histórico',
                'plazo': '1 mes post-campaña'
            }
        ],

        'recursos_necesarios': [
            'Equipo de Marketing (diseño de campaña)',
            'Equipo de Producto (implementación de features virales)',
            'Herramienta de tracking (analytics, atribución)'
        ],

        'semillas_recomendadas': [
            int(s) for s in analisis_propagacion['semillas_mejor_estrategia'][:5]
        ],

        'comparacion_estrategias': analisis_propagacion['ranking_estrategias']
    }

    return accionable


def generar_accionable_autoridades(analisis_autoridades: Dict) -> Dict:
    """
    Accionable 3: Potenciar Páginas con Rol de Autoridad
    """
    total_autoridades = analisis_autoridades['total_autoridades']
    top_autoridades = analisis_autoridades['top_autoridades']

    accionable = {
        'id': 'ACT-003',
        'titulo': 'Potenciar Contenido y UX de Páginas Autoridad',
        'descripcion': (
            f"Se identificaron {total_autoridades} páginas con rol de Autoridad "
            "(alta Authority HITS, alto in-degree). Estas páginas son confiables "
            "y reciben muchos enlaces. Optimizarlas maximiza conversión."
        ),
        'categoria': 'Optimización de Conversión',
        'prioridad': 'ALTA',
        'impacto_estimado': 'Alto',
        'esfuerzo_estimado': 'Medio-Alto',
        'roi_esperado': 'Alto',

        'acciones_concretas': [
            "Auditar contenido y UX de las páginas autoridad top-10",
            "Actualizar contenido (descripciones, imágenes, reviews)",
            "Optimizar CTAs y flujo de conversión",
            "Implementar A/B tests en estas páginas prioritarias",
            "Agregar señales de confianza (reviews, ratings, garantías)"
        ],

        'kpis': [
            {
                'metrica': 'Tasa de conversión (páginas autoridad)',
                'valor_actual': 'Baseline actual',
                'objetivo': '+15% vs baseline',
                'plazo': '6 semanas'
            },
            {
                'metrica': 'Tiempo en página (engagement)',
                'valor_actual': 'Baseline actual',
                'objetivo': '+25% vs baseline',
                'plazo': '6 semanas'
            },
            {
                'metrica': 'Bounce rate',
                'valor_actual': 'Baseline actual',
                'objetivo': '-10% vs baseline',
                'plazo': '6 semanas'
            }
        ],

        'recursos_necesarios': [
            'Equipo de Producto (priorización de mejoras)',
            'Equipo de Diseño (optimización de UX)',
            'Equipo de Contenidos (actualización de copy)',
            'Equipo de Data (A/B testing y análisis)'
        ],

        'paginas_prioritarias': [
            {
                'nodo': auth['nodo'],
                'in_degree': auth['in_degree'],
                'authority': auth['authority'],
                'pagerank': auth['pagerank']
            }
            for auth in top_autoridades[:10]
        ]
    }

    return accionable


# =============================================================================
# GUARDADO DE RECOMENDACIONES
# =============================================================================
def guardar_json(accionables: List[Dict], metadata: Dict, ruta: str, logger: logging.Logger):
    """Guarda recomendaciones en formato JSON."""
    logger.info(f"Guardando recomendaciones en JSON: {ruta}")

    output = {
        'metadata': metadata,
        'accionables': accionables
    }

    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"  ✓ JSON guardado: {len(accionables)} accionables")


def guardar_markdown(accionables: List[Dict], metadata: Dict, ruta: str, logger: logging.Logger):
    """Guarda recomendaciones en formato Markdown (para PMs)."""
    logger.info(f"Guardando recomendaciones en Markdown: {ruta}")

    lines = [
        "# Recomendaciones Accionables - Optimización de Grafo de URLs",
        "",
        f"**Fecha de Generación:** {metadata['fecha_generacion']}  ",
        f"**Grafo Analizado:** {metadata['num_nodos']} nodos, {metadata['num_aristas']} aristas  ",
        f"**Método:** Análisis de Grafos con NetworkX + Simulación de Propagación",
        "",
        "---",
        ""
    ]

    for i, acc in enumerate(accionables, 1):
        lines.extend([
            f"## {i}. {acc['titulo']}",
            "",
            f"**ID:** `{acc['id']}`  ",
            f"**Categoría:** {acc['categoria']}  ",
            f"**Prioridad:** {acc['prioridad']}  ",
            f"**Impacto Estimado:** {acc['impacto_estimado']} | **Esfuerzo:** {acc['esfuerzo_estimado']} | **ROI:** {acc['roi_esperado']}",
            "",
            "### Descripción",
            "",
            acc['descripcion'],
            "",
            "### Acciones Concretas",
            ""
        ])

        for accion in acc['acciones_concretas']:
            lines.append(f"- {accion}")

        lines.extend(["", "### KPIs de Éxito", ""])

        for kpi in acc['kpis']:
            lines.extend([
                f"- **{kpi['metrica']}**",
                f"  - Actual: {kpi['valor_actual']}",
                f"  - Objetivo: {kpi['objetivo']}",
                f"  - Plazo: {kpi['plazo']}",
                ""
            ])

        lines.extend(["### Recursos Necesarios", ""])
        for recurso in acc['recursos_necesarios']:
            lines.append(f"- {recurso}")

        lines.extend(["", "---", ""])

    with open(ruta, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    logger.info(f"  ✓ Markdown guardado: {len(accionables)} accionables")


def guardar_csv(accionables: List[Dict], ruta: str, logger: logging.Logger):
    """Guarda resumen de accionables en CSV (para tracking)."""
    logger.info(f"Guardando resumen en CSV: {ruta}")

    rows = []
    for acc in accionables:
        rows.append({
            'ID': acc['id'],
            'Titulo': acc['titulo'],
            'Categoria': acc['categoria'],
            'Prioridad': acc['prioridad'],
            'Impacto': acc['impacto_estimado'],
            'Esfuerzo': acc['esfuerzo_estimado'],
            'ROI': acc['roi_esperado'],
            'Num_Acciones': len(acc['acciones_concretas']),
            'Num_KPIs': len(acc['kpis'])
        })

    df = pd.DataFrame(rows)
    df.to_csv(ruta, index=False, encoding='utf-8')

    logger.info(f"  ✓ CSV guardado: {len(rows)} accionables")


def guardar_txt_resumen_ejecutivo(accionables: List[Dict], metadata: Dict, ruta: str, logger: logging.Logger):
    """Guarda resumen ejecutivo en TXT (para directivos)."""
    logger.info(f"Guardando resumen ejecutivo en TXT: {ruta}")

    lines = [
        "=" * 80,
        "RESUMEN EJECUTIVO - RECOMENDACIONES DE OPTIMIZACIÓN",
        "=" * 80,
        "",
        f"Fecha: {metadata['fecha_generacion']}",
        f"Grafo: {metadata['num_nodos']} nodos, {metadata['num_aristas']} aristas",
        "",
        "-" * 80,
        "ACCIONABLES PRIORITARIOS",
        "-" * 80,
        ""
    ]

    for i, acc in enumerate(accionables, 1):
        lines.extend([
            f"{i}. {acc['titulo'].upper()}",
            f"   Prioridad: {acc['prioridad']} | ROI: {acc['roi_esperado']}",
            "",
            f"   {acc['descripcion']}",
            "",
            "   Próximos pasos:",
        ])

        for j, accion in enumerate(acc['acciones_concretas'][:3], 1):  # Top-3 acciones
            lines.append(f"   {j}) {accion}")

        lines.extend(["", ""])

    lines.extend([
        "-" * 80,
        "RESUMEN DE IMPACTO",
        "-" * 80,
        "",
        f"Total de accionables: {len(accionables)}",
        f"Alta prioridad: {sum(1 for a in accionables if a['prioridad'] == 'ALTA')}",
        f"ROI muy alto/alto: {sum(1 for a in accionables if 'Alto' in a['roi_esperado'])}",
        "",
        "=" * 80
    ])

    with open(ruta, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    logger.info(f"  ✓ TXT guardado: resumen ejecutivo")


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================
def main():
    """Función principal de FASE 7."""

    # Cargar configuración
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logger = setup_logging(config)

    logger.info("=" * 80)
    logger.info("FASE 7: RECOMENDACIONES ACCIONABLES PARA E-COMMERCE")
    logger.info("=" * 80)

    # Crear directorio de salida
    output_dir = Path(config['outputs']['recomendaciones_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directorio de salida: {output_dir}")

    # Rutas de resultados de fases previas
    results_dir = Path(config['outputs']['results_dir'])
    propagacion_dir = Path(config['outputs']['propagacion_dir'])

    ruta_fase2 = results_dir / "fase2_estadisticas.json"
    ruta_fase3 = results_dir / "fase3_metricas_centralidad.csv"
    ruta_fase4 = results_dir / "fase4_analisis_top20.json"
    ruta_fase6 = propagacion_dir / "fase6_comparacion_estrategias.json"

    # -------------------------------------------------------------------------
    # 1. CARGAR RESULTADOS DE FASES PREVIAS
    # -------------------------------------------------------------------------
    logger.info("\n[PASO 1] Cargando resultados de fases previas...")

    resultados_fase2 = cargar_resultados_fase2(ruta_fase2, logger)
    df_metricas = cargar_resultados_fase3(ruta_fase3, logger)
    resultados_fase4 = cargar_resultados_fase4(ruta_fase4, logger)
    resultados_fase6 = cargar_resultados_fase6(ruta_fase6, logger)

    # -------------------------------------------------------------------------
    # 2. ANÁLISIS Y DETECCIÓN DE OPORTUNIDADES
    # -------------------------------------------------------------------------
    logger.info("\n[PASO 2] Analizando oportunidades de optimización...")

    analisis_huerfanos = analizar_nodos_huerfanos(df_metricas, logger)

    top_n_ranking = config['ranking']['top_n']
    analisis_concentracion = analizar_concentracion_pagerank(df_metricas, top_n_ranking, logger)

    analisis_propagacion = analizar_mejor_estrategia_propagacion(resultados_fase6, logger)

    analisis_autoridades = analizar_roles_autoridades(resultados_fase4, df_metricas, logger)

    # -------------------------------------------------------------------------
    # 3. GENERACIÓN DE ACCIONABLES
    # -------------------------------------------------------------------------
    logger.info("\n[PASO 3] Generando accionables concretos...")

    accionables = []

    # Accionable 1: Link Building Interno
    accionable_1 = generar_accionable_link_building(analisis_huerfanos, config)
    accionables.append(accionable_1)
    logger.info(f"  ✓ {accionable_1['id']}: {accionable_1['titulo']}")

    # Accionable 2: Optimización de Propagación
    accionable_2 = generar_accionable_propagacion(analisis_propagacion)
    accionables.append(accionable_2)
    logger.info(f"  ✓ {accionable_2['id']}: {accionable_2['titulo']}")

    # Accionable 3: Potenciar Autoridades
    accionable_3 = generar_accionable_autoridades(analisis_autoridades)
    accionables.append(accionable_3)
    logger.info(f"  ✓ {accionable_3['id']}: {accionable_3['titulo']}")

    # -------------------------------------------------------------------------
    # 4. GUARDADO EN MÚLTIPLES FORMATOS
    # -------------------------------------------------------------------------
    logger.info("\n[PASO 4] Guardando recomendaciones en múltiples formatos...")

    from datetime import datetime
    metadata = {
        'fecha_generacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_nodos': resultados_fase2['basicas']['num_nodos'],
        'num_aristas': resultados_fase2['basicas']['num_aristas'],
        'densidad': resultados_fase2['basicas']['densidad'],
        'num_accionables': len(accionables)
    }

    # JSON (para developers)
    if 'json' in config['recomendaciones']['formatos_salida']:
        ruta_json = output_dir / "fase7_recomendaciones.json"
        guardar_json(accionables, metadata, ruta_json, logger)

    # Markdown (para PMs)
    if 'markdown' in config['recomendaciones']['formatos_salida']:
        ruta_md = output_dir / "fase7_recomendaciones.md"
        guardar_markdown(accionables, metadata, ruta_md, logger)

    # CSV (para tracking)
    if 'csv' in config['recomendaciones']['formatos_salida']:
        ruta_csv = output_dir / "fase7_accionables.csv"
        guardar_csv(accionables, ruta_csv, logger)

    # TXT (resumen ejecutivo)
    if 'txt' in config['recomendaciones']['formatos_salida']:
        ruta_txt = output_dir / "fase7_resumen_ejecutivo.txt"
        guardar_txt_resumen_ejecutivo(accionables, metadata, ruta_txt, logger)

    # -------------------------------------------------------------------------
    # 5. RESUMEN FINAL
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("FASE 7 COMPLETADA")
    logger.info("=" * 80)
    logger.info(f"✓ {len(accionables)} accionables generados")
    logger.info(f"✓ Archivos guardados en: {output_dir}")
    logger.info("")
    logger.info("PRÓXIMOS PASOS:")
    logger.info("  1. Revisar recomendaciones en formato Markdown (para PMs)")
    logger.info("  2. Priorizar accionables según recursos disponibles")
    logger.info("  3. Implementar accionables de alta prioridad")
    logger.info("  4. Monitorear KPIs definidos en cada accionable")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
