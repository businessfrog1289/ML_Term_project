"""
SECOM Semiconductor Manufacturing Defect Detection
Step 6: Final Model Comparison (앙상블 포함)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    f1_score, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score
)

BLUE   = '#2563EB'; RED    = '#DC2626'; GREEN  = '#10B981'
ORANGE = '#F59E0B'; PURPLE = '#7C3AED'; GRAY   = '#6B7280'
TEAL   = '#0891B2'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

# ── 결과 로드 ────────────────────────────────────────────────────────────────
print("결과 파일 로드 중...", flush=True)
with open('../results/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
y_test = data['y_test']

with open('../results/baseline_results.pkl', 'rb') as f:
    bl = pickle.load(f)['results']

with open('../results/glm_results.pkl', 'rb') as f:
    glm = pickle.load(f)

with open('../results/svm_results.pkl', 'rb') as f:
    svm = pickle.load(f)

with open('../results/dt_results.pkl', 'rb') as f:
    dt = pickle.load(f)

with open('../results/ensemble_results.pkl', 'rb') as f:
    ens = pickle.load(f)

print("로드 완료!", flush=True)

# ── 전체 모델 정보 통합 ───────────────────────────────────────────────────────
all_models = [
    # Baseline
    {'name': 'LR (Baseline)', 'color': GRAY,
     'y_prob': bl[0]['y_prob'], 'y_pred': bl[0]['y_pred'],
     'acc': bl[0]['acc'], 'f1_macro': bl[0]['f1_macro'],
     'f1_fail': bl[0]['f1_fail'], 'roc_auc': bl[0]['roc_auc'],
     'avg_prec': bl[0]['avg_prec']},
    {'name': 'LDA',           'color': TEAL,
     'y_prob': bl[1]['y_prob'], 'y_pred': bl[1]['y_pred'],
     'acc': bl[1]['acc'], 'f1_macro': bl[1]['f1_macro'],
     'f1_fail': bl[1]['f1_fail'], 'roc_auc': bl[1]['roc_auc'],
     'avg_prec': bl[1]['avg_prec']},
    {'name': 'Naive Bayes',   'color': GREEN,
     'y_prob': bl[2]['y_prob'], 'y_pred': bl[2]['y_pred'],
     'acc': bl[2]['acc'], 'f1_macro': bl[2]['f1_macro'],
     'f1_fail': bl[2]['f1_fail'], 'roc_auc': bl[2]['roc_auc'],
     'avg_prec': bl[2]['avg_prec']},
    # GLM
    {'name': 'Ridge (L2)',    'color': '#60A5FA',
     'y_prob': glm['ridge']['y_prob'], 'y_pred': glm['ridge']['y_pred'],
     'acc': glm['ridge']['acc'], 'f1_macro': glm['ridge']['f1_macro'],
     'f1_fail': glm['ridge']['f1_fail'], 'roc_auc': glm['ridge']['roc_auc'],
     'avg_prec': glm['ridge']['avg_prec']},
    {'name': 'Lasso (L1)',    'color': BLUE,
     'y_prob': glm['lasso']['y_prob'], 'y_pred': glm['lasso']['y_pred'],
     'acc': glm['lasso']['acc'], 'f1_macro': glm['lasso']['f1_macro'],
     'f1_fail': glm['lasso']['f1_fail'], 'roc_auc': glm['lasso']['roc_auc'],
     'avg_prec': glm['lasso']['avg_prec']},
    {'name': 'ElasticNet',    'color': '#818CF8',
     'y_prob': glm['enet']['y_prob'], 'y_pred': glm['enet']['y_pred'],
     'acc': glm['enet']['acc'], 'f1_macro': glm['enet']['f1_macro'],
     'f1_fail': glm['enet']['f1_fail'], 'roc_auc': glm['enet']['roc_auc'],
     'avg_prec': glm['enet']['avg_prec']},
    # SVM
    {'name': 'SVM (RBF)',     'color': RED,
     'y_prob': svm['y_prob'], 'y_pred': svm['y_pred'],
     'acc': svm['acc'], 'f1_macro': svm['f1_macro'],
     'f1_fail': svm['f1_fail'], 'roc_auc': svm['roc_auc'],
     'avg_prec': svm['avg_prec']},
    # Decision Tree
    {'name': 'Decision Tree', 'color': ORANGE,
     'y_prob': dt['y_prob'], 'y_pred': dt['y_pred'],
     'acc': dt['acc'], 'f1_macro': dt['f1_macro'],
     'f1_fail': dt['f1_fail'], 'roc_auc': dt['roc_auc'],
     'avg_prec': dt['avg_prec']},
    # Ensemble
    {'name': 'Random Forest', 'color': '#16A34A',
     'y_prob': ens['rf']['y_prob'], 'y_pred': ens['rf']['y_pred'],
     'acc': ens['rf']['acc'], 'f1_macro': ens['rf']['f1_macro'],
     'f1_fail': ens['rf']['f1_fail'], 'roc_auc': ens['rf']['roc_auc'],
     'avg_prec': ens['rf']['avg_prec']},
    {'name': 'Grad. Boosting','color': '#9333EA',
     'y_prob': ens['gb']['y_prob'], 'y_pred': ens['gb']['y_pred'],
     'acc': ens['gb']['acc'], 'f1_macro': ens['gb']['f1_macro'],
     'f1_fail': ens['gb']['f1_fail'], 'roc_auc': ens['gb']['roc_auc'],
     'avg_prec': ens['gb']['avg_prec']},
    {'name': 'Soft Voting',   'color': '#DB2777',
     'y_prob': ens['voting']['y_prob'], 'y_pred': ens['voting']['y_pred'],
     'acc': ens['voting']['acc'], 'f1_macro': ens['voting']['f1_macro'],
     'f1_fail': ens['voting']['f1_fail'], 'roc_auc': ens['voting']['roc_auc'],
     'avg_prec': ens['voting']['avg_prec']},
]

# ── Figure 15: 전체 ROC + PR ──────────────────────────────────────────────────
print("\n[그래프] Figure 15 생성 중...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Figure 15. All Models – ROC & Precision-Recall Curves',
             fontsize=14, fontweight='bold')

for m in all_models:
    fpr, tpr, _ = roc_curve(y_test, m['y_prob'])
    lw = 2.5 if m['name'] in ['Random Forest','Grad. Boosting','Soft Voting'] else 1.5
    ls = '-' if m['name'] in ['Random Forest','Grad. Boosting','Soft Voting',
                               'SVM (RBF)'] else '--'
    axes[0].plot(fpr, tpr, color=m['color'], lw=lw, ls=ls,
                 label=f"{m['name']} ({m['roc_auc']:.3f})")
    prec, rec, _ = precision_recall_curve(y_test, m['y_prob'])
    axes[1].plot(rec, prec, color=m['color'], lw=lw, ls=ls,
                 label=f"{m['name']} ({m['avg_prec']:.3f})")

axes[0].plot([0,1],[0,1],'k--', lw=1)
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('(a) ROC Curves (AUC in legend)', fontweight='bold')
axes[0].legend(fontsize=7, loc='lower right', ncol=2)

axes[1].axhline(y_test.mean(), color='k', linestyle='--', lw=1, label='Random')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('(b) PR Curves (AP in legend)', fontweight='bold')
axes[1].legend(fontsize=7, loc='upper right', ncol=2)

plt.tight_layout()
plt.savefig('../figures/fig15_all_roc_pr.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: fig15_all_roc_pr.png")

# ── Figure 16: ROC-AUC 순위 막대 ─────────────────────────────────────────────
print("[그래프] Figure 16 생성 중...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Figure 16. Final Model Comparison', fontsize=14, fontweight='bold')

names  = [m['name'] for m in all_models]
aucs   = [m['roc_auc'] for m in all_models]
colors_b = [m['color'] for m in all_models]
sorted_idx = np.argsort(aucs)

bars = axes[0].barh(
    [names[i] for i in sorted_idx],
    [aucs[i]  for i in sorted_idx],
    color=[colors_b[i] for i in sorted_idx], alpha=0.85
)
for bar, v in zip(bars, [aucs[i] for i in sorted_idx]):
    axes[0].text(bar.get_width() + 0.003,
                 bar.get_y() + bar.get_height()/2,
                 f'{v:.4f}', va='center', fontsize=9, fontweight='bold')
axes[0].set_xlim(0.4, 1.05)
axes[0].set_xlabel('ROC-AUC')
axes[0].set_title('(a) ROC-AUC Ranking', fontweight='bold')
axes[0].axvline(aucs[0], color=GRAY, linestyle='--', lw=1,
                alpha=0.6, label='LR Baseline')
axes[0].legend()

# (b) F1-Fail 순위
f1_fails = [m['f1_fail'] for m in all_models]
sorted_f1 = np.argsort(f1_fails)
bars2 = axes[1].barh(
    [names[i] for i in sorted_f1],
    [f1_fails[i] for i in sorted_f1],
    color=[colors_b[i] for i in sorted_f1], alpha=0.85
)
for bar, v in zip(bars2, [f1_fails[i] for i in sorted_f1]):
    axes[1].text(bar.get_width() + 0.003,
                 bar.get_y() + bar.get_height()/2,
                 f'{v:.4f}', va='center', fontsize=9, fontweight='bold')
axes[1].set_xlim(0, 0.75)
axes[1].set_xlabel('F1 Score (Fail Class)')
axes[1].set_title('(b) F1-Fail Ranking', fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/fig16_final_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: fig16_final_comparison.png")

# ── Figure 17: 클래스별 F1 ────────────────────────────────────────────────────
print("[그래프] Figure 17 생성 중...", flush=True)
fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle('Figure 17. Class-wise F1 Score – All Models',
             fontsize=14, fontweight='bold')

x = np.arange(len(all_models))
w = 0.35
f1_pass = [f1_score(y_test, m['y_pred'], pos_label=0) for m in all_models]
f1_fail = [f1_score(y_test, m['y_pred'], pos_label=1) for m in all_models]

ax.bar(x - w/2, f1_pass, w, color=BLUE,  alpha=0.8, label='F1 (Pass)', edgecolor='white')
ax.bar(x + w/2, f1_fail, w, color=RED,   alpha=0.8, label='F1 (Fail)', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels([m['name'] for m in all_models],
                   rotation=30, ha='right', fontsize=8)
ax.set_ylim(0, 1.1)
ax.set_ylabel('F1 Score')
ax.legend()
for i, (p, f) in enumerate(zip(f1_pass, f1_fail)):
    ax.text(i - w/2, p + 0.02, f'{p:.3f}', ha='center', fontsize=7)
    ax.text(i + w/2, f + 0.02, f'{f:.3f}', ha='center', fontsize=7)

# 앙상블 영역 표시
ax.axvspan(7.5, len(all_models)-0.5, alpha=0.05, color=GREEN, label='Ensemble')
ax.legend()
plt.tight_layout()
plt.savefig('../figures/fig17_classwise_f1.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: fig17_classwise_f1.png")

# ── Figure 18: 오분류 분석 (Best 모델) ───────────────────────────────────────
print("[그래프] Figure 18 생성 중...", flush=True)

# Best 모델 선택 (ROC-AUC 기준)
best_m = max(all_models, key=lambda x: x['roc_auc'])
print(f"  Best model: {best_m['name']} (AUC={best_m['roc_auc']:.4f})")

svm_pred = best_m['y_pred']
svm_prob = best_m['y_prob']
TP = (y_test==1) & (svm_pred==1)
FN = (y_test==1) & (svm_pred==0)
FP = (y_test==0) & (svm_pred==1)
TN = (y_test==0) & (svm_pred==0)

print(f"  TP={TP.sum()}  FN={FN.sum()}  FP={FP.sum()}  TN={TN.sum()}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'Figure 18. Failure Mode Analysis (Best: {best_m["name"]})',
             fontsize=14, fontweight='bold')

labels_m = ['TP\n(Fail→Fail)', 'FN\n(Fail→Pass)', 'FP\n(Pass→Fail)', 'TN\n(Pass→Pass)']
masks_m  = [TP, FN, FP, TN]
cols_m   = [GREEN, RED, ORANGE, BLUE]
for mask, lbl, col in zip(masks_m, labels_m, cols_m):
    if mask.sum() > 0:
        axes[0].hist(svm_prob[mask], bins=20, alpha=0.6, color=col,
                     label=f'{lbl} (n={mask.sum()})', density=True)
axes[0].set_xlabel('Predicted Probability (Fail)')
axes[0].set_ylabel('Density')
axes[0].set_title('(a) Probability by Outcome', fontweight='bold')
axes[0].legend(fontsize=8)
axes[0].axvline(0.5, color='black', linestyle='--', lw=1)

counts_m = [TP.sum(), FN.sum(), FP.sum(), TN.sum()]
explode  = [0, 0.1, 0.1, 0]
axes[1].pie(counts_m, labels=labels_m, colors=cols_m,
            autopct='%1.1f%%', explode=explode, startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2))
axes[1].set_title('(b) Confusion Outcome Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/fig18_failure_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: fig18_failure_analysis.png")

# ── 최종 요약 CSV 저장 ────────────────────────────────────────────────────────
final_df = pd.DataFrame([
    {k: m[k] for k in ['name','acc','f1_macro','f1_fail','roc_auc','avg_prec']}
    for m in all_models
])
final_df.to_csv('../results/final_comparison.csv', index=False)
print("✓ Saved: results/final_comparison.csv")

# ── 콘솔 출력 ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON SUMMARY")
print("=" * 70)
print(f"{'Model':<22} {'Acc':>7} {'F1-Mac':>8} {'F1-Fail':>8} {'AUC':>7} {'AP':>7}")
print("-" * 70)
for m in sorted(all_models, key=lambda x: x['roc_auc'], reverse=True):
    marker = " ◀ BEST" if m['name'] == best_m['name'] else ""
    print(f"{m['name']:<22} {m['acc']:>7.4f} {m['f1_macro']:>8.4f} "
          f"{m['f1_fail']:>8.4f} {m['roc_auc']:>7.4f} "
          f"{m['avg_prec']:>7.4f}{marker}")
print("=" * 70)
