"""
SECOM Semiconductor Manufacturing Defect Detection
Step 4: Support Vector Machine (속도 최적화 버전)
  - PCA 50개로 차원 축소 (95% 대신)
  - GridSearch 범위 축소
  - n_jobs=-1 병렬 처리
  - 실시간 진행도 출력
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, warnings, time
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE

BLUE  = '#2563EB'; RED = '#DC2626'; GREEN = '#10B981'
ORANGE = '#F59E0B'; PURPLE = '#7C3AED'; GRAY = '#6B7280'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
with open('../results/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

print("=" * 60)
print("Step 4: Support Vector Machine")
print("=" * 60)

# ── SMOTE ────────────────────────────────────────────────────────────────────
print("\n[SMOTE] 오버샘플링 중...", flush=True)
smote = SMOTE(random_state=42, k_neighbors=5)
X_sm, y_sm = smote.fit_resample(X_train, y_train)
print(f"[SMOTE] 완료: {np.bincount(y_train)} → {np.bincount(y_sm)}")

# ── PCA 차원 축소 ★ 핵심 속도 개선 ──────────────────────────────────────────
print("\n[PCA] 차원 축소 중...", flush=True)
pca = PCA(n_components=50, random_state=42)   # ★ 95%분산 대신 고정 50개
X_sm_pca   = pca.fit_transform(X_sm)
X_test_pca = pca.transform(X_test)
print(f"[PCA] 완료: {X_sm.shape[1]}차원 → 50차원 "
      f"(분산 설명률: {pca.explained_variance_ratio_.sum()*100:.1f}%)")

# ── Grid Search ★ 범위 축소 ──────────────────────────────────────────────────
# 기존: C=[0.1,1,10,100] x gamma=['scale','auto',0.001,0.01] = 16가지
# 변경: C=[0.1,1,10]     x gamma=['scale',0.01]              = 6가지
param_grid = {
    'C':     [0.1, 1, 10],
    'gamma': ['scale', 0.01]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 수동 그리드 서치 (진행도 출력용)
print("\n[GridSearch] RBF SVM 하이퍼파라미터 탐색 중...", flush=True)
combos = [(C, g) for C in param_grid['C'] for g in param_grid['gamma']]
total  = len(combos)
best_score  = -1
best_params = {}
all_scores  = {}

start_all = time.time()
for i, (C, gamma) in enumerate(combos):
    t0 = time.time()
    svc = SVC(kernel='rbf', C=C, gamma=gamma,
              class_weight='balanced', probability=True, random_state=42)
    score = cross_val_score(svc, X_sm_pca, y_sm,
                            cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    all_scores[(C, gamma)] = score
    elapsed   = time.time() - t0
    remaining = elapsed * (total - i - 1)
    bar = '█' * (i + 1) + '░' * (total - i - 1)
    print(f"  [{bar}] {i+1}/{total}  C={C}, gamma={gamma}  "
          f"AUC={score:.4f}  ({elapsed:.1f}s, 남은≈{remaining:.0f}s)", flush=True)
    if score > best_score:
        best_score  = score
        best_params = {'C': C, 'gamma': gamma}

print(f"\n[GridSearch] 완료! 총 소요={time.time()-start_all:.1f}s")
print(f"  Best params: {best_params}")
print(f"  Best CV AUC: {best_score:.4f}")

# ── Figure 9: GridSearch 히트맵 ──────────────────────────────────────────────
print("\n[그래프] Figure 9 생성 중...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Figure 9. SVM GridSearchCV Results', fontsize=14, fontweight='bold')

C_vals     = param_grid['C']
gamma_vals = [str(g) for g in param_grid['gamma']]
heat = np.array([[all_scores[(C, g)] for g in param_grid['gamma']]
                 for C in C_vals])

im = axes[0].imshow(heat, cmap='YlOrRd', aspect='auto',
                    vmin=heat.min(), vmax=heat.max())
axes[0].set_xticks(range(len(gamma_vals))); axes[0].set_xticklabels(gamma_vals)
axes[0].set_yticks(range(len(C_vals)));     axes[0].set_yticklabels(C_vals)
axes[0].set_xlabel('gamma'); axes[0].set_ylabel('C')
axes[0].set_title('(a) CV AUC Heatmap (RBF Kernel)', fontweight='bold')
plt.colorbar(im, ax=axes[0])
for i in range(len(C_vals)):
    for j in range(len(gamma_vals)):
        axes[0].text(j, i, f'{heat[i,j]:.3f}',
                     ha='center', va='center', fontsize=10)

# (b) 커널 비교
print("[그래프] 커널 비교 중...", flush=True)
bc = best_params['C']
bg = best_params['gamma']
kernels = ['linear', 'rbf', 'poly']
kernel_aucs = []
for k in kernels:
    svc = SVC(kernel=k, C=bc,
              gamma=bg if k != 'linear' else 'scale',
              class_weight='balanced', probability=True, random_state=42)
    auc = cross_val_score(svc, X_sm_pca, y_sm,
                          cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    kernel_aucs.append(auc)
    print(f"  Kernel={k}: CV AUC={auc:.4f}", flush=True)

axes[1].bar(kernels, kernel_aucs,
            color=[BLUE, RED, ORANGE], alpha=0.85, edgecolor='white')
for i, v in enumerate(kernel_aucs):
    axes[1].text(i, v + 0.003, f'{v:.4f}',
                 ha='center', fontsize=11, fontweight='bold')
axes[1].set_ylim(min(kernel_aucs) - 0.05, 1.02)
axes[1].set_ylabel('CV ROC-AUC')
axes[1].set_title(f'(b) Kernel Comparison\n(C={bc}, gamma={bg})', fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/fig09_svm_gridsearch.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: fig09_svm_gridsearch.png")

# ── 최적 SVM 최종 학습 & 평가 ────────────────────────────────────────────────
print(f"\n[평가] Best SVM 최종 학습 중... (C={bc}, gamma={bg})", flush=True)
best_svm = SVC(C=bc, gamma=bg, kernel='rbf',
               class_weight='balanced', probability=True, random_state=42)
best_svm.fit(X_sm_pca, y_sm)

y_pred = best_svm.predict(X_test_pca)
y_prob = best_svm.predict_proba(X_test_pca)[:, 1]
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average='macro')
f1b = f1_score(y_test, y_pred, pos_label=1)
auc = roc_auc_score(y_test, y_prob)
ap  = average_precision_score(y_test, y_prob)
cm  = confusion_matrix(y_test, y_pred)

print(f"\n[Best SVM]  Acc={acc:.4f}  F1-Mac={f1m:.4f}  "
      f"F1-Fail={f1b:.4f}  AUC={auc:.4f}  AP={ap:.4f}")
print(classification_report(y_test, y_pred, target_names=['Pass', 'Fail']))

# ── Figure 10: SVM 평가 ───────────────────────────────────────────────────────
print("\n[그래프] Figure 10 생성 중...", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Figure 10. Best SVM – Evaluation Results', fontsize=14, fontweight='bold')

# (a) Confusion Matrix
im = axes[0].imshow(cm, cmap='Blues')
axes[0].set_xticks([0,1]); axes[0].set_yticks([0,1])
axes[0].set_xticklabels(['Pass','Fail']); axes[0].set_yticklabels(['Pass','Fail'])
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
axes[0].set_title(f'(a) Confusion Matrix\n(C={bc}, γ={bg})', fontweight='bold')
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i,j]), ha='center', va='center',
                     fontsize=18, fontweight='bold',
                     color='white' if cm[i,j] > cm.max()/2 else 'black')

# (b) ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color=RED, lw=2.5, label=f'SVM RBF (AUC={auc:.3f})')
axes[1].plot([0,1],[0,1],'k--', lw=1)
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
axes[1].set_title('(b) ROC Curve', fontweight='bold'); axes[1].legend()

# (c) 2D 결정 경계 (PCA 2차원)
print("[그래프] 2D 결정 경계 계산 중...", flush=True)
pca2   = PCA(n_components=2, random_state=42)
X_2d   = pca2.fit_transform(X_sm)
X_te2d = pca2.transform(X_test)
svm2d  = SVC(C=bc, gamma=bg, kernel='rbf',
             class_weight='balanced', probability=True, random_state=42)
svm2d.fit(X_2d, y_sm)

h = 0.3
x_min, x_max = X_2d[:,0].min()-1, X_2d[:,0].max()+1
y_min, y_max = X_2d[:,1].min()-1, X_2d[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = svm2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[2].contourf(xx, yy, Z, alpha=0.2, cmap='RdBu')
axes[2].scatter(X_te2d[y_test==0,0], X_te2d[y_test==0,1],
                c=BLUE, s=12, alpha=0.5, label='Pass')
axes[2].scatter(X_te2d[y_test==1,0], X_te2d[y_test==1,1],
                c=RED,  s=40, alpha=0.9, label='Fail', marker='*')
axes[2].set_xlabel('PC1'); axes[2].set_ylabel('PC2')
axes[2].set_title('(c) Decision Boundary (2D PCA)', fontweight='bold')
axes[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig('../figures/fig10_svm_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: fig10_svm_evaluation.png")

# ── 저장 ─────────────────────────────────────────────────────────────────────
svm_results = {
    'best_params': best_params, 'best_cv_auc': best_score,
    'acc': acc, 'f1_macro': f1m, 'f1_fail': f1b,
    'roc_auc': auc, 'avg_prec': ap, 'cm': cm,
    'y_pred': y_pred, 'y_prob': y_prob,
    'pca': pca, 'model': best_svm,
    'kernel_aucs': dict(zip(kernels, kernel_aucs))
}
with open('../results/svm_results.pkl', 'wb') as f:
    pickle.dump(svm_results, f)
print("✓ Saved: results/svm_results.pkl")

print("\n" + "=" * 60)
print(f"Best SVM: C={bc}, gamma={bg}, AUC={auc:.4f}")
print("=" * 60)