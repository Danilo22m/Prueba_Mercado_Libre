# Conclusiones - Ejercicio 1: LLM vs Modelo Clásico

## Resumen Ejecutivo

Este documento presenta las conclusiones del análisis comparativo entre un modelo basado en LLM (Modelo A) y un modelo clásico de Machine Learning (Modelo B - Isolation Forest) para la detección de anomalías en series temporales de precios.

**Fecha del Análisis**: 2026-01-12
**Autor**: Danilo Melo

---

## 1. Resultados de Evaluación

### Modelo A - LLM (Groq API)

**Dataset**: 300 registros (muestra aleatoria del test set)

| Métrica | Valor |
|---------|-------|
| **Precision** | 2.76% |
| **Recall** | 100% |
| **F1-Score** | 5.36% |
| **PR-AUC** | 0.461 |

**Matriz de Confusión**:
```
                Predicted
                Normal  Anomalo
Actual Normal     46      247
       Anomalo     0        7
```

**Análisis**:
- ✅ **Recall perfecto (100%)**: Detecta TODAS las anomalías reales
- ❌ **Precision muy baja (2.76%)**: Alta tasa de falsos positivos (247 FP)
- ⚠️ **Comportamiento conservador**: El modelo prefiere marcar como anómalo cuando tiene duda

### Modelo B - Isolation Forest

**Dataset**: 35,183 registros (test set completo)

| Métrica | Valor |
|---------|-------|
| **Precision** | 18.93% |
| **Recall** | 21.41% |
| **F1-Score** | 20.09% |
| **PR-AUC** | 0.214 |

**Matriz de Confusión**:
```
                  Predicted
                  Normal  Anomalo
Actual Normal    33,983    574
       Anomalo      492    134
```

**Análisis**:
- ✅ **Mejor precision (18.93% vs 2.76%)**: Menos falsos positivos
- ✅ **Mejor F1-Score (20.09% vs 5.36%)**: Balance superior precision-recall
- ❌ **Recall moderado (21.41%)**: Pierde ~79% de anomalías reales
- ✅ **Escalable**: Procesa 35k registros vs 300 del LLM

---

## 2. A/B Testing Estadístico

**Método**: Bootstrap estratificado (1,000 iteraciones)
**Dataset Común**: 300 registros
**Nivel de Confianza**: 95% (α = 0.05)

### Resultados del Bootstrap

| Métrica | LLM (Media ± SD) | IF (Media ± SD) | Diferencia | p-value | Significativo |
|---------|------------------|-----------------|------------|---------|---------------|
| **Precision** | 2.76% ± 0.07% | 50.67% ± 27.49% | -47.91% | 0.178 | ❌ No |
| **Recall** | 100% ± 0% | 27.59% ± 16.20% | +72.41% | <0.001 | ✅ **Sí** |
| **F1-Score** | 5.37% ± 0.13% | 34.01% ± 17.78% | -28.64% | 0.178 | ❌ No |

### Intervalos de Confianza (95%)

- **Precision**: [-97.30%, +2.79%] - No significativo
- **Recall**: [+42.86%, +100%] - **Significativo** (LLM superior)
- **F1-Score**: [-61.38%, +5.43%] - No significativo

### Interpretación

1. **Recall significativamente superior en LLM**: Con 100% de confianza estadística, el LLM tiene mejor recall
2. **Precision e F1 no significativos**: Alta variabilidad impide conclusiones definitivas
3. **Alta varianza en IF**: Desviaciones estándar grandes (27.49% en precision) indican inestabilidad

---

## 3. Análisis Comparativo Detallado

### 3.1. Trade-offs Precision vs Recall

```
┌─────────────────────────────────────────────────────────────┐
│                 Precision vs Recall                         │
├─────────────────────────────────────────────────────────────┤
│  LLM:    ████████████████████████████████████████ 100% Recall
│          ██ 2.76% Precision                                 │
│                                                              │
│  IF:     █████████ 21.41% Recall                           │
│          ███████████ 18.93% Precision                       │
└─────────────────────────────────────────────────────────────┘
```

**LLM**: Maximiza recall a costa de precision (estrategia "catch-all")
**IF**: Balance intermedio pero pierde muchas anomalías reales

### 3.2. Costo Computacional

| Aspecto | LLM | Isolation Forest |
|---------|-----|------------------|
| **Tiempo de procesamiento** | ~20 min (600 registros) | ~5 seg (35k registros) |
| **Latencia por predicción** | ~2000 ms | <0.1 ms |
| **Escalabilidad** | ❌ Limitada (rate limits) | ✅ Excelente |
| **Costo** | $0.00 (Groq gratis) | $0.00 (local) |
| **Infraestructura** | ☁️ Requiere API externa | 💻 Local |

### 3.3. Explicabilidad

| Aspecto | LLM | Isolation Forest |
|---------|-----|------------------|
| **Explicaciones** | ✅ Razones en lenguaje natural | ❌ Solo anomaly score |
| **Confidence** | ✅ Score de 0.0-1.0 | ✅ Anomaly score |
| **Interpretabilidad** | ✅ Alta (humanos entienden) | ⚠️ Media (técnico) |

**Ejemplo de explicación LLM**:
```json
{
  "label": "ANOMALO",
  "confidence": 0.95,
  "reason": "Precio 3.5x por encima de la media histórica"
}
```

---

## 4. Casos de Uso Recomendados

### ✅ Cuándo usar LLM (Modelo A)

1. **Recall crítico**: Cuando NO detectar una anomalía es muy costoso
   - Ejemplo: Fraude financiero, alertas de seguridad

2. **Explicabilidad requerida**: Cuando necesitas justificar decisiones a humanos
   - Ejemplo: Auditorías, compliance, customer support

3. **Baja frecuencia**: Pocos registros por día/hora
   - Ejemplo: Precios de productos premium

4. **Prototipado rápido**: Exploración inicial sin entrenar modelos

### ✅ Cuándo usar Isolation Forest (Modelo B)

1. **Alto volumen**: Miles/millones de predicciones por día
   - Ejemplo: Streaming de precios en tiempo real

2. **Latencia crítica**: Respuestas en <1ms requeridas
   - Ejemplo: Trading algorítmico, sistemas de recomendación

3. **Offline**: Sin acceso a APIs externas
   - Ejemplo: Edge computing, sistemas on-premise

4. **Balance precision-recall**: F1-Score es métrica clave

---

## 5. Limitaciones del Estudio

### Modelo A (LLM)

1. **Muestra pequeña**: Solo 300 registros (vs 35k del IF)
   - Sesgo de muestreo posible
   - Intervalos de confianza amplios

2. **Prompt engineering**: Resultados sensibles al prompt usado
   - Un prompt diferente podría cambiar precision/recall
   - No se hizo optimización de prompt

3. **Rate limits**: 30 req/min limita experimentación
   - No se probaron múltiples configuraciones
   - No se hizo tuning de temperatura/top_p

### Modelo B (Isolation Forest)

1. **Contamination fijo**: 2% basado en proporción real
   - No se optimizó este hiperparámetro
   - Podría mejorarse con tuning

2. **Features limitadas**: Solo 10 features engineered
   - Más features podrían mejorar rendimiento
   - No se probaron transformaciones no lineales

### A/B Testing

1. **Dataset común pequeño**: Solo 300 registros
   - Poder estadístico limitado
   - Alta variabilidad en IF

2. **Distribución diferente**: LLM sobre muestra aleatoria vs IF sobre test completo
   - Comparación no perfectamente justa

---

## 6. Recomendaciones

### Arquitectura Híbrida (Recomendada)

Combinar ambos modelos para maximizar fortalezas:

```
┌────────────────────────────────────────────────────────────┐
│                   Sistema Híbrido                          │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Isolation Forest (Filtro rápido)                       │
│     ↓                                                       │
│     Detecta anomalías candidatas (alta recall)             │
│     ↓                                                       │
│  2. LLM (Validación selectiva)                            │
│     ↓                                                       │
│     Valida solo anomalías candidatas con explicación       │
│     ↓                                                       │
│  3. Output final con explicación                          │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

**Ventajas**:
- ✅ Velocidad de IF para filtrado inicial
- ✅ Explicabilidad de LLM para casos importantes
- ✅ Balance costo-beneficio óptimo

### Mejoras Futuras

#### Modelo A (LLM)
1. **Optimizar prompt**: Experimentar con diferentes estructuras
2. **Few-shot learning**: Incluir ejemplos en el prompt
3. **Ensemble de LLMs**: Combinar múltiples modelos (Groq, OpenAI, Anthropic)
4. **Fine-tuning**: Entrenar modelo específico en datos históricos

#### Modelo B (Isolation Forest)
1. **Hyperparameter tuning**: GridSearch/RandomSearch para contamination
2. **Feature engineering avanzado**: Más features estadísticas
3. **Ensemble**: Combinar con otros modelos (LOF, One-Class SVM)
4. **Deep learning**: Probar Autoencoders o LSTMs

#### A/B Testing
1. **Aumentar muestra**: 1,000+ registros para LLM
2. **Estratificación**: Asegurar distribución similar de anomalías
3. **Métricas adicionales**: MCC, Cohen's Kappa, ROC-AUC

---

## 7. Conclusiones Finales

### Hallazgos Clave

1. **LLM tiene recall superior (100%)** pero precision muy baja (2.76%)
   - Estrategia "catch-all" detecta todas las anomalías pero con muchos falsos positivos

2. **Isolation Forest tiene mejor balance** (F1: 20.09% vs 5.36%)
   - Mejor para producción por velocidad y escalabilidad

3. **Diferencia en recall es estadísticamente significativa** (p < 0.001)
   - LLM es significativamente mejor para no perder anomalías reales

4. **Precision e F1 no son estadísticamente diferentes** (p = 0.178)
   - Alta variabilidad impide conclusiones definitivas

### Respuesta a la Pregunta de Investigación

**¿Puede un LLM reemplazar un modelo clásico para detección de anomalías?**

**Respuesta**: **NO** para uso general, **SÍ** para casos específicos.

**Razones**:
- ❌ **Costo computacional**: 400x más lento (2000ms vs 0.1ms por predicción)
- ❌ **Escalabilidad**: Rate limits impiden alto volumen
- ❌ **Precision baja**: 97% de falsos positivos es inviable
- ✅ **Recall perfecto**: Útil cuando no detectar es crítico
- ✅ **Explicabilidad**: Valor agregado en escenarios de compliance

### Recomendación Final

**Para producción en e-commerce**: **Isolation Forest** con monitoreo humano

**Para casos críticos con bajo volumen**: **Sistema Híbrido** (IF + LLM)

**Para exploración y prototipado**: **LLM** por rapidez de implementación

---

## 8. Referencias

### Archivos de Resultados

- **Métricas completas**: `outputs/results/evaluacion_completa.json`
- **A/B Test**: `outputs/results/ab_test_results.json`
- **Comparación**: `outputs/results/comparacion_modelos.csv`
- **Predicciones LLM**: `outputs/results/predicciones_llm.csv`
- **Predicciones IF**: `outputs/results/predicciones_isolation_forest.csv`

### Visualizaciones

- **Matrices de confusión**: `outputs/plots/confusion_matrices.png`
- **Curvas PR**: `outputs/plots/precision_recall_curves.png`
- **Series temporales**: `outputs/plots/series_temporales_comparacion_modelos.png`

### Configuración

- **Config YAML**: `config/config.yaml`
- **Modelo A**: 600 registros, llama-3.3-70b-versatile, temperature=0.0
- **Modelo B**: contamination=0.02, n_estimators=100

---

**Documento generado**: 2026-01-12
**Proyecto**: Prueba Técnica Mercado Libre - Ejercicio 1
**Autor**: Danilo Melo
