#!/usr/bin/env python3
"""
Analisis detallado del rendimiento de Isolation Forest
"""

# Analisis usando comandos basicos (no requiere pandas)
import csv

# Leer CSV
with open('outputs/results/predicciones_isolation_forest.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

total = len(rows)
print(f'=== ANALISIS MODELO B (ISOLATION FOREST) ===\n')
print(f'Total registros: {total:,}\n')

# Contar verdaderos positivos, falsos positivos, etc
tp = sum(1 for r in rows if r['TRUE_LABEL'] == 'ANOMALO' and r['PRED_IF'] == 'ANOMALO')
tn = sum(1 for r in rows if r['TRUE_LABEL'] == 'NORMAL' and r['PRED_IF'] == 'NORMAL')
fp = sum(1 for r in rows if r['TRUE_LABEL'] == 'NORMAL' and r['PRED_IF'] == 'ANOMALO')
fn = sum(1 for r in rows if r['TRUE_LABEL'] == 'ANOMALO' and r['PRED_IF'] == 'NORMAL')

true_anomalies = sum(1 for r in rows if r['TRUE_LABEL'] == 'ANOMALO')
pred_anomalies = sum(1 for r in rows if r['PRED_IF'] == 'ANOMALO')

print(f'=== CONFUSION MATRIX ===')
print(f'                  Predicted NORMAL | Predicted ANOMALO')
print(f'True NORMAL       {tn:,} (TN)        | {fp:,} (FP)')
print(f'True ANOMALO      {fn:,} (FN)        | {tp:,} (TP)')
print()

# Calcular metricas
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f'=== METRICAS ===')
print(f'Precision: {precision:.4f} ({precision*100:.2f}%)')
print(f'Recall:    {recall:.4f} ({recall*100:.2f}%)')
print(f'F1-Score:  {f1:.4f}')
print()

print(f'=== DIAGNOSTICO ===')
print(f'Verdaderas anomalias en datos: {true_anomalies:,} ({true_anomalies/total*100:.2f}%)')
print(f'Anomalias predichas:           {pred_anomalies:,} ({pred_anomalies/total*100:.2f}%)')
print(f'Anomalias correctamente identificadas: {tp:,} de {true_anomalies:,} ({recall*100:.2f}%)')
print(f'Falsos positivos: {fp:,}')
print(f'Falsos negativos: {fn:,}')
print()

# Analizar ZSCORE de anomalias no detectadas
print(f'=== ANALISIS ZSCORE ===')
zscores_true_anomalo = [float(r['ZSCORE']) for r in rows if r['TRUE_LABEL'] == 'ANOMALO']
zscores_missed = [float(r['ZSCORE']) for r in rows if r['TRUE_LABEL'] == 'ANOMALO' and r['PRED_IF'] == 'NORMAL']

if zscores_true_anomalo:
    print(f'ZSCORE promedio de verdaderas anomalias: {sum(zscores_true_anomalo)/len(zscores_true_anomalo):.2f}')
    print(f'ZSCORE min/max de verdaderas anomalias: {min(zscores_true_anomalo):.2f} / {max(zscores_true_anomalo):.2f}')

if zscores_missed:
    print(f'ZSCORE promedio de anomalias NO detectadas: {sum(zscores_missed)/len(zscores_missed):.2f}')
    print(f'ZSCORE min/max de anomalias NO detectadas: {min(zscores_missed):.2f} / {max(zscores_missed):.2f}')

print()
print(f'=== PROBLEMA IDENTIFICADO ===')
if recall < 0.5:
    print(f'⚠️  RECALL MUY BAJO ({recall*100:.1f}%)')
    print(f'El modelo solo detecta {recall*100:.1f}% de las anomalias reales.')
    print(f'Está fallando en identificar {fn:,} anomalias verdaderas (falsos negativos).')
    print()
    print('POSIBLES CAUSAS:')
    print('1. Contamination parameter (0.1) no coincide con la proporción real de anomalías')
    print('2. Isolation Forest se basa en features numéricas que no capturan bien el patrón de anomalías')
    print('3. El ground truth (basado en ZSCORE) no es compatible con la detección de Isolation Forest')
    print('4. Las features necesitan normalización/escalado')
elif precision < 0.5:
    print(f'⚠️  PRECISION MUY BAJA ({precision*100:.1f}%)')
    print(f'El modelo genera muchos falsos positivos: {fp:,}')
else:
    print(f'✓ El modelo tiene métricas aceptables.')
