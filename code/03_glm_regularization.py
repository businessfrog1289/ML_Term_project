"""
SECOM Semiconductor Manufacturing Defect Detection
Step 3: GLM + Regularization (속도 최적화 버전)
  - C_grid: 30개 → 10개
  - max_iter: 2000 → 500
  - cv: 5-fold 유지 (다른 모델과 통일)
  - solver: ridge=lbfgs, lasso=liblinear (saga보다 빠름)
  - n_jobs=-1: 병렬 처리
  - 실시간 진행도 출력
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle, warnings, time
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE

BLUE   = '#2563EB'; RED = '#DC2626'; GREEN = '#10B981'
ORANGE = '#F59E0B'; PURPLE = '#7C3AED'; GRAY = '#6B7280'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
with open('../results/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']
feat_names = data['feature_names']

print("=" * 60)
print("Step 3: GLM + Regularization")
print("=" * 60)

# ── SMOTE ────────────────────────────────────────────────────────────────────
print("\n[SMOTE] 오버샘플링 중...", flush=True)
smote = SMOTE(random_state=42, k_neighbors=5)
X_sm, y_sm = smote.fit_resample(X_train, y_train)
print(f"[SMOTE] 완료: {np.bincount(y_train)} → {np.bincount(y_sm)}")

# ── C 탐색 설정 ───────────────────────────────────────────────────────────────
C_grid = np.logspace(-3, 2, 10)           # ★ 30 → 10개
cv     = StratifiedKFold(n_splits=5,      # 5-fold 유지
                         shuffle=True, random_state=42)

def search_C(penalty, l1_ratio=None):
    aucs  = []
    total = len(C_grid)

    # penalty별 최적 solver (saga보다 훨씬 빠름)
    if penalty == 'l2':
        solver = 'lbfgs'
    elif penalty == 'l1':
        solver = 'liblinear'
    else:
        solver = 'saga'   # elasticnet은 saga만 지원

    start_all = time.time()
    for i, C in enumerate(C_grid):
        t0 = time.time()
        kw = dict(C=C, max_iter=500,      # ★ 2000 → 500
                  class_weight='balanced',
                  random_state=42, solver=solver)
        if penalty == 'elasticnet':
            kw['penalty'] = 'elasticnet'
            kw['l1_ratio'] = l1_ratio
        else:
            kw['penalty'] = penalty

        score = cross_val_score(
            LogisticRegression(**kw), X_sm, y_sm,
            cv=cv, scoring='roc_auc', n_jobs=-1   # ★ 병렬 처리
        ).mean()
        aucs.append(score)

        elapsed   = time.time() - t0
        remaining = elapsed * (total - i - 1)
        bar = '█' * (i + 1) + '░' * (total - i - 1)
        print(f"  [{bar}] {i+1}/{total}  C={C:.5f}  AUC={score:.4f}  "
              f"({elapsed:.1f}s/step, 남은시간≈{remaining:.0f}s)", flush=True)

    best_idx = np.argmax(aucs)
    print(f"  → 완료! Best C={C_grid[best_idx]:.5f}, "
          f"Best AUC={aucs[best_idx]:.4f}, "
          f"총 소요={time.time()-start_all:.1f}s")
    return C_grid[best_idx], aucs

print(f"\n[1/3] Ridge (L2) C 탐색 중... (C 후보 {len(C_grid)}개 × 5-fold)")
C_ridge, auc_ridge = search_C('l2')

print(f"\n[2/3] Lasso (L1) C 탐색 중... (C 후보 {len(C_grid)}개 × 5-fold)")
C_lasso, auc_lasso = search_C('l1')

print(f"\n[3/3] ElasticNet C 탐색 중... (C 후보 {len(C_grid)}개 × 5-fold)")
C_enet, auc_enet = search_C('elasticnet', l1_ratio=0.5)

# ── Figure 7: C 탐색 곡선 & 계수 경로 ───────────────────────────────────────
print("\n[그래프] Figure 7 생성 중...", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Figure 7. Regularization Hyperparameter Search', fontsize=14, fontweight='bold')

# (a) CV AUC vs C
axes[0].semilogx(C_grid, auc_ridge, color=BLUE,   lw=2, label='Ridge (L2)')
axes[0].semilogx(C_grid, auc_lasso, color=RED,    lw=2, label='Lasso (L1)')
axes[0].semilogx(C_grid, auc_enet,  color=ORANGE, lw=2, label='ElasticNet')
axes[0].axvline(C_ridge, color=BLUE,   linestyle='--', alpha=0.6)
axes[0].axvline(C_lasso, color=RED,    linestyle='--', alpha=0.6)
axes[0].axvline(C_enet,  color=ORANGE, linestyle='--', alpha=0.6)
axes[0].set_xlabel('C (Inverse Regularization Strength)')
axes[0].set_ylabel('CV ROC-AUC')
axes[0].set_title('(a) C Tuning Curve', fontweight='bold')
axes[0].legend()

# (b) Lasso 비영 계수 수 vs C
print("[그래프] Lasso 계수 경로 계산 중...", flush=True)
coef_paths = []
for i, C in enumerate(C_grid):
    m = LogisticRegression(penalty='l1', C=C, max_iter=500,
                            solver='liblinear', class_weight='balanced', random_state=42)
    m.fit(X_sm, y_sm)
    coef_paths.append(m.coef_[0])
    print(f"  계수 경로 [{i+1}/{len(C_grid)}] C={C:.5f}", flush=True)
coef_paths = np.array(coef_paths)
n_nonzero  = (coef_paths != 0).sum(axis=1)
axes[1].semilogx(C_grid, n_nonzero, color=RED, lw=2)
axes[1].axvline(C_lasso, color=RED, linestyle='--', alpha=0.7,
                label=f'Best C={C_lasso:.4f}')
axes[1].set_xlabel('C')
axes[1].set_ylabel('# Non-zero Coefficients')
axes[1].set_title('(b) Lasso: Non-zero Coeff vs C', fontweight='bold')
axes[1].legend()

# (c) 최적 Lasso 상위 20 계수
best_lasso_tmp = LogisticRegression(penalty='l1', C=C_lasso, max_iter=500,
                                     solver='liblinear', class_weight='balanced', random_state=42)
best_lasso_tmp.fit(X_sm, y_sm)
coef      = best_lasso_tmp.coef_[0]
top_idx   = np.argsort(np.abs(coef))[::-1][:20]
top_coefs = coef[top_idx]
top_names = [f'F{feat_names[i]}' for i in top_idx]
colors_bar = [RED if c > 0 else BLUE for c in top_coefs]
axes[2].barh(range(20), top_coefs[::-1], color=colors_bar[::-1], alpha=0.85)
axes[2].set_yticks(range(20))
axes[2].set_yticklabels(top_names[::-1], fontsize=8)
axes[2].axvline(0, color='black', lw=0.8)
axes[2].set_xlabel('Coefficient Value')
axes[2].set_title('(c) Lasso: Top-20 Coefficients', fontweight='bold')
axes[2].legend(handles=[
    mpatches.Patch(color=RED, label='→ Fail'),
    mpatches.Patch(color=BLUE, label='→ Pass')], fontsize=8)

plt.tight_layout()
plt.savefig('../figures/fig07_regularization.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: fig07_regularization.png")

# ── 최적 모델 최종 평가 ──────────────────────────────────────────────────────
def eval_model(name, model, X_tr, y_tr, X_te, y_te):
    print(f"\n[평가] {name} 학습 중...", flush=True)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    f1m = f1_score(y_te, y_pred, average='macro')
    f1b = f1_score(y_te, y_pred, pos_label=1)
    auc = roc_auc_score(y_te, y_prob)
    ap  = average_precision_score(y_te, y_prob)
    cm  = confusion_matrix(y_te, y_pred)
    print(f"[{name}]  Acc={acc:.4f}  F1-Mac={f1m:.4f}  F1-Fail={f1b:.4f}  AUC={auc:.4f}  AP={ap:.4f}")
    print(classification_report(y_te, y_pred, target_names=['Pass', 'Fail']))
    return dict(name=name, model=model, y_pred=y_pred, y_prob=y_prob,
                acc=acc, f1_macro=f1m, f1_fail=f1b, roc_auc=auc, avg_prec=ap, cm=cm)

ridge_res = eval_model('Ridge (L2)',
    LogisticRegression(penalty='l2', C=C_ridge, max_iter=500,
                       solver='lbfgs', class_weight='balanced', random_state=42),
    X_sm, y_sm, X_test, y_test)

lasso_res = eval_model('Lasso (L1)',
    LogisticRegression(penalty='l1', C=C_lasso, max_iter=500,
                       solver='liblinear', class_weight='balanced', random_state=42),
    X_sm, y_sm, X_test, y_test)

enet_res = eval_model('ElasticNet',
    LogisticRegression(penalty='elasticnet', C=C_enet, l1_ratio=0.5,
                       max_iter=500, solver='saga',
                       class_weight='balanced', random_state=42),
    X_sm, y_sm, X_test, y_test)

# ── Figure 8: ROC + PR ───────────────────────────────────────────────────────
print("\n[그래프] Figure 8 생성 중...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Figure 8. Regularized GLM – ROC & PR Curves', fontsize=14, fontweight='bold')

for res, c in zip([ridge_res, lasso_res, enet_res], [BLUE, RED, ORANGE]):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[0].plot(fpr, tpr, color=c, lw=2,
                 label=f"{res['name'].split()[0]} (AUC={res['roc_auc']:.3f})")
    prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
    axes[1].plot(rec, prec, color=c, lw=2,
                 label=f"{res['name'].split()[0]} (AP={res['avg_prec']:.3f})")

axes[0].plot([0,1],[0,1],'k--', lw=1)
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
axes[0].set_title('(a) ROC Curve', fontweight='bold'); axes[0].legend()
axes[1].axhline(y_test.mean(), color='k', linestyle='--', lw=1, label='Random')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('(b) Precision-Recall Curve', fontweight='bold'); axes[1].legend()

plt.tight_layout()
plt.savefig('../figures/fig08_glm_roc_pr.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: fig08_glm_roc_pr.png")

# ── 저장 ─────────────────────────────────────────────────────────────────────
glm_results = {'ridge': ridge_res, 'lasso': lasso_res, 'enet': enet_res,
               'best_C': {'ridge': C_ridge, 'lasso': C_lasso, 'enet': C_enet}}
with open('../results/glm_results.pkl', 'wb') as f:
    pickle.dump(glm_results, f)
print("✓ Saved: results/glm_results.pkl")

print("\n" + "=" * 60)
print("GLM Summary")
print("=" * 60)
print(f"{'Model':<20} {'Acc':>6} {'F1-Mac':>8} {'F1-Fail':>8} {'AUC':>7}")
print("-" * 55)
for r in [ridge_res, lasso_res, enet_res]:
    print(f"{r['name']:<20} {r['acc']:>6.4f} {r['f1_macro']:>8.4f} {r['f1_fail']:>8.4f} {r['roc_auc']:>7.4f}")