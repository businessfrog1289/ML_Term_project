"""
SECOM Semiconductor Manufacturing Defect Detection
Step 5: Decision Tree (속도 최적화 버전)
  - ccp_alpha 경로 대신 max_depth 기반 1-SE 규칙
  - n_jobs=-1 병렬 처리
  - 실시간 진행도 출력
  - 5-fold 유지
"""
import os
os.makedirs('../figures', exist_ok=True)
os.makedirs('../results', exist_ok=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, warnings, time
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier, plot_tree
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
feat_names = data['feature_names']

print("=" * 60)
print("Step 5: Decision Tree with 1-SE Rule")
print("=" * 60)

# ── SMOTE ────────────────────────────────────────────────────────────────────
print("\n[SMOTE] 오버샘플링 중...", flush=True)
smote = SMOTE(random_state=42, k_neighbors=5)
X_sm, y_sm = smote.fit_resample(X_train, y_train)
print(f"[SMOTE] 완료: {np.bincount(y_train)} → {np.bincount(y_sm)}")

# ── max_depth 기반 1-SE 탐색 ★ ccp_alpha 루프 제거 ──────────────────────────
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
depths = list(range(2, 16))   # depth 2~15 탐색

print(f"\n[Depth 탐색] max_depth {depths[0]}~{depths[-1]} × 5-fold CV")
cv_aucs, cv_stds, train_aucs = [], [], []
start_all = time.time()

for i, d in enumerate(depths):
    t0 = time.time()
    dt = DecisionTreeClassifier(max_depth=d,
                                class_weight='balanced', random_state=42)
    # CV AUC
    scores = cross_val_score(dt, X_sm, y_sm,
                             cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_aucs.append(scores.mean())
    cv_stds.append(scores.std())
    # Train AUC
    dt.fit(X_sm, y_sm)
    train_aucs.append(roc_auc_score(y_sm, dt.predict_proba(X_sm)[:,1]))

    elapsed   = time.time() - t0
    remaining = elapsed * (len(depths) - i - 1)
    bar = '█' * (i + 1) + '░' * (len(depths) - i - 1)
    print(f"  [{bar}] depth={d:2d}  CV AUC={scores.mean():.4f}±{scores.std():.4f}"
          f"  Train AUC={train_aucs[-1]:.4f}"
          f"  ({elapsed:.1f}s, 남은≈{remaining:.0f}s)", flush=True)

cv_aucs  = np.array(cv_aucs)
cv_stds  = np.array(cv_stds)

# 1-SE Rule
best_i   = np.argmax(cv_aucs)
threshold = cv_aucs[best_i] - cv_stds[best_i]
se_cands = [i for i in range(len(depths))
            if cv_aucs[i] >= threshold and i >= best_i]
se_i     = se_cands[-1] if se_cands else best_i

best_depth = depths[best_i]
se_depth   = depths[se_i]

print(f"\n[1-SE Rule]")
print(f"  Best depth  = {best_depth}  (CV AUC={cv_aucs[best_i]:.4f})")
print(f"  1-SE depth  = {se_depth}   (CV AUC={cv_aucs[se_i]:.4f})")
print(f"  threshold   = {threshold:.4f}")
print(f"  총 소요     = {time.time()-start_all:.1f}s")

# ── Figure 11: Depth 탐색 곡선 ────────────────────────────────────────────────
print("\n[그래프] Figure 11 생성 중...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Figure 11. Decision Tree – Depth Tuning', fontsize=14, fontweight='bold')

# (a) Train vs CV AUC
axes[0].plot(depths, train_aucs, color=BLUE, lw=2, label='Train AUC')
axes[0].plot(depths, cv_aucs,    color=RED,  lw=2, label='CV AUC')
axes[0].fill_between(depths,
                     cv_aucs - cv_stds,
                     cv_aucs + cv_stds,
                     alpha=0.15, color=RED)
axes[0].axvline(best_depth, color=ORANGE, linestyle='--', lw=2,
                label=f'Best depth={best_depth}')
axes[0].axvline(se_depth,   color=PURPLE, linestyle='--', lw=2,
                label=f'1-SE depth={se_depth}')
axes[0].set_xlabel('Max Depth'); axes[0].set_ylabel('ROC-AUC')
axes[0].set_title('(a) Train vs CV AUC by Depth', fontweight='bold')
axes[0].legend()

# (b) 1-SE Rule 시각화
axes[1].plot(depths, cv_aucs, 'o-', color=BLUE, lw=2, ms=6, label='CV AUC')
axes[1].fill_between(depths,
                     cv_aucs - cv_stds,
                     cv_aucs + cv_stds,
                     alpha=0.15, color=BLUE)
axes[1].axhline(threshold, color=GRAY, linestyle='--',
                label=f'1-SE threshold={threshold:.4f}')
axes[1].axvline(best_depth, color=ORANGE, linestyle='--', lw=2,
                label=f'Best depth={best_depth}')
axes[1].axvline(se_depth,   color=PURPLE, linestyle='--', lw=2,
                label=f'1-SE depth={se_depth}')
axes[1].set_xlabel('Max Depth'); axes[1].set_ylabel('CV ROC-AUC')
axes[1].set_title('(b) 1-SE Rule', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('../figures/fig11_dt_pruning.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: fig11_dt_pruning.png")

# ── Figure 12: 1-SE Rule 단독 ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(depths, cv_aucs, 'o-', color=BLUE, lw=2, ms=6, label='CV AUC (5-fold)')
ax.fill_between(depths,
                cv_aucs - cv_stds,
                cv_aucs + cv_stds,
                alpha=0.15, color=BLUE, label='±1 std')
ax.axhline(threshold, color=GRAY, linestyle='--',
           label=f'1-SE threshold ({threshold:.4f})')
ax.axvline(best_depth, color=ORANGE, linestyle='--', lw=2,
           label=f'Best depth={best_depth} (AUC={cv_aucs[best_i]:.4f})')
ax.axvline(se_depth,   color=PURPLE, linestyle='--', lw=2,
           label=f'1-SE depth={se_depth}  (AUC={cv_aucs[se_i]:.4f})')
ax.set_xlabel('Max Depth'); ax.set_ylabel('CV ROC-AUC')
ax.set_title('Figure 12. 1-SE Rule for Decision Tree Depth Selection',
             fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('../figures/fig12_dt_1se.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: fig12_dt_1se.png")

# ── 최적 모델 학습 (1-SE depth) ──────────────────────────────────────────────
print(f"\n[학습] 최적 트리 학습 중... (max_depth={se_depth})", flush=True)
dt_best = DecisionTreeClassifier(max_depth=se_depth,
                                  class_weight='balanced', random_state=42)
dt_best.fit(X_sm, y_sm)
print(f"  트리 깊이={dt_best.get_depth()}, 잎 노드={dt_best.get_n_leaves()}")

y_pred = dt_best.predict(X_test)
y_prob = dt_best.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average='macro')
f1b = f1_score(y_test, y_pred, pos_label=1)
auc = roc_auc_score(y_test, y_prob)
ap  = average_precision_score(y_test, y_prob)
cm  = confusion_matrix(y_test, y_pred)

print(f"\n[DT 1-SE]  Acc={acc:.4f}  F1-Mac={f1m:.4f}  "
      f"F1-Fail={f1b:.4f}  AUC={auc:.4f}  AP={ap:.4f}")
print(classification_report(y_test, y_pred, target_names=['Pass', 'Fail']))

# ── Figure 13: 트리 구조 + 특성 중요도 ───────────────────────────────────────
print("\n[그래프] Figure 13 생성 중...", flush=True)
importances = dt_best.feature_importances_
top20_idx   = np.argsort(importances)[::-1][:20]

fig = plt.figure(figsize=(20, 9))
fig.suptitle('Figure 13. Decision Tree – Structure & Feature Importance',
             fontsize=14, fontweight='bold')

ax_tree = plt.subplot2grid((1, 5), (0, 0), colspan=3)
plot_tree(dt_best, max_depth=3,
          feature_names=feat_names,
          class_names=['Pass', 'Fail'],
          filled=True, rounded=True, fontsize=7, ax=ax_tree)
ax_tree.set_title('(a) Pruned Tree (first 3 levels)', fontweight='bold')

ax_imp = plt.subplot2grid((1, 5), (0, 3), colspan=2)
colors_imp = plt.cm.YlOrRd(np.linspace(0.3, 0.9, 20))
ax_imp.barh(range(20), importances[top20_idx][::-1],
            color=colors_imp[::-1], alpha=0.9)
ax_imp.set_yticks(range(20))
ax_imp.set_yticklabels([f'F{feat_names[i]}' for i in top20_idx][::-1], fontsize=8)
ax_imp.set_xlabel('Feature Importance (Gini)')
ax_imp.set_title('(b) Top-20 Feature Importances', fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/fig13_dt_tree_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: fig13_dt_tree_importance.png")

# ── Figure 14: 평가 결과 ──────────────────────────────────────────────────────
print("[그래프] Figure 14 생성 중...", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle('Figure 14. Decision Tree – Evaluation Results',
             fontsize=14, fontweight='bold')

# (a) Confusion Matrix
axes[0].imshow(cm, cmap='Blues')
axes[0].set_xticks([0,1]); axes[0].set_yticks([0,1])
axes[0].set_xticklabels(['Pass','Fail']); axes[0].set_yticklabels(['Pass','Fail'])
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
axes[0].set_title('(a) Confusion Matrix', fontweight='bold')
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i,j]), ha='center', va='center',
                     fontsize=18, fontweight='bold',
                     color='white' if cm[i,j] > cm.max()/2 else 'black')

# (b) ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color=ORANGE, lw=2.5, label=f'DT 1-SE (AUC={auc:.3f})')
axes[1].plot([0,1],[0,1],'k--', lw=1)
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
axes[1].set_title('(b) ROC Curve', fontweight='bold'); axes[1].legend()

# (c) PR
prec, rec, _ = precision_recall_curve(y_test, y_prob)
axes[2].plot(rec, prec, color=ORANGE, lw=2.5, label=f'DT (AP={ap:.3f})')
axes[2].axhline(y_test.mean(), color='k', linestyle='--', lw=1, label='Random')
axes[2].set_xlabel('Recall'); axes[2].set_ylabel('Precision')
axes[2].set_title('(c) Precision-Recall Curve', fontweight='bold'); axes[2].legend()

plt.tight_layout()
plt.savefig('../figures/fig14_dt_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: fig14_dt_evaluation.png")

# ── 저장 ─────────────────────────────────────────────────────────────────────
dt_results = {
    'best_depth': best_depth, 'se1_depth': se_depth,
    'depth': dt_best.get_depth(), 'leaves': dt_best.get_n_leaves(),
    'acc': acc, 'f1_macro': f1m, 'f1_fail': f1b,
    'roc_auc': auc, 'avg_prec': ap, 'cm': cm,
    'y_pred': y_pred, 'y_prob': y_prob,
    'importances': importances
}
with open('../results/dt_results.pkl', 'wb') as f:
    pickle.dump(dt_results, f)
print("✓ Saved: results/dt_results.pkl")

print("\n" + "=" * 60)
print(f"Best depth={best_depth}, 1-SE depth={se_depth}, AUC={auc:.4f}")
print("=" * 60)
