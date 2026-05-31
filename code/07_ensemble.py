"""
SECOM Semiconductor Manufacturing Defect Detection
Step 7: Ensemble Models (속도 최적화 버전)
  - Random Forest
  - XGBoost (GradientBoosting)
  - Voting Classifier (Soft Voting)
  - 5-fold CV 유지
  - n_jobs=-1 병렬 처리
  - 실시간 진행도 출력
"""
import os
os.makedirs('../figures', exist_ok=True)
os.makedirs('../results', exist_ok=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle, warnings, time
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE

BLUE   = '#2563EB'; RED    = '#DC2626'; GREEN  = '#10B981'
ORANGE = '#F59E0B'; PURPLE = '#7C3AED'; GRAY   = '#6B7280'
TEAL   = '#0891B2'
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
print("Step 7: Ensemble Models")
print("=" * 60)

# ── SMOTE ────────────────────────────────────────────────────────────────────
print("\n[SMOTE] 오버샘플링 중...", flush=True)
smote = SMOTE(random_state=42, k_neighbors=5)
X_sm, y_sm = smote.fit_resample(X_train, y_train)
print(f"[SMOTE] 완료: {np.bincount(y_train)} → {np.bincount(y_sm)}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── 평가 헬퍼 ────────────────────────────────────────────────────────────────
def eval_model(name, model, X_tr, y_tr, X_te, y_te):
    print(f"\n[학습] {name} ...", flush=True)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    print(f"  학습 완료 ({time.time()-t0:.1f}s)", flush=True)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    f1m = f1_score(y_te, y_pred, average='macro')
    f1b = f1_score(y_te, y_pred, pos_label=1)
    auc = roc_auc_score(y_te, y_prob)
    ap  = average_precision_score(y_te, y_prob)
    cm  = confusion_matrix(y_te, y_pred)

    print(f"  Acc={acc:.4f}  F1-Mac={f1m:.4f}  "
          f"F1-Fail={f1b:.4f}  AUC={auc:.4f}  AP={ap:.4f}")
    print(classification_report(y_te, y_pred, target_names=['Pass', 'Fail']))
    return dict(name=name, model=model, y_pred=y_pred, y_prob=y_prob,
                acc=acc, f1_macro=f1m, f1_fail=f1b, roc_auc=auc, avg_prec=ap, cm=cm)

# ════════════════════════════════════════════════════════════════
# 1. Random Forest
# ════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[1/3] Random Forest n_estimators 탐색 중...")
print("─" * 50)

n_list = [50, 100, 200]
rf_aucs = []
for i, n in enumerate(n_list):
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=n, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    score = cross_val_score(rf, X_sm, y_sm,
                            cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    rf_aucs.append(score)
    bar = '█' * (i+1) + '░' * (len(n_list)-i-1)
    print(f"  [{bar}] {i+1}/{len(n_list)}  "
          f"n_estimators={n}  AUC={score:.4f}  ({time.time()-t0:.1f}s)", flush=True)

best_n = n_list[np.argmax(rf_aucs)]
print(f"  → Best n_estimators={best_n}, AUC={max(rf_aucs):.4f}")

rf_res = eval_model(
    'Random Forest',
    RandomForestClassifier(
        n_estimators=best_n, class_weight='balanced',
        max_features='sqrt', random_state=42, n_jobs=-1
    ),
    X_sm, y_sm, X_test, y_test
)

# ════════════════════════════════════════════════════════════════
# 2. Gradient Boosting (XGBoost 대체)
# ════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[2/3] Gradient Boosting 탐색 중...")
print("─" * 50)

# 속도를 위해 n_estimators 작게, subsample로 속도 개선
gb_params = [
    {'n_estimators': 100, 'learning_rate': 0.1,  'max_depth': 3},
    {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 4},
    {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 3},
]
gb_aucs = []
for i, p in enumerate(gb_params):
    t0 = time.time()
    gb = GradientBoostingClassifier(
        subsample=0.8, random_state=42, **p
    )
    score = cross_val_score(gb, X_sm, y_sm,
                            cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    gb_aucs.append(score)
    bar = '█' * (i+1) + '░' * (len(gb_params)-i-1)
    print(f"  [{bar}] {i+1}/{len(gb_params)}  "
          f"n={p['n_estimators']}, lr={p['learning_rate']}, "
          f"depth={p['max_depth']}  AUC={score:.4f}  ({time.time()-t0:.1f}s)", flush=True)

best_p = gb_params[np.argmax(gb_aucs)]
print(f"  → Best params={best_p}, AUC={max(gb_aucs):.4f}")

gb_res = eval_model(
    'Gradient Boosting',
    GradientBoostingClassifier(
        subsample=0.8, random_state=42, **best_p
    ),
    X_sm, y_sm, X_test, y_test
)

# ════════════════════════════════════════════════════════════════
# 3. Soft Voting Ensemble (RF + GB + LR)
# ════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[3/3] Soft Voting Ensemble 학습 중...")
print("─" * 50)

voting_res = eval_model(
    'Soft Voting (RF+GB+LR)',
    VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(
                n_estimators=best_n, class_weight='balanced',
                random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(
                subsample=0.8, random_state=42, **best_p)),
            ('lr', LogisticRegression(
                C=0.1, max_iter=500, solver='lbfgs',
                class_weight='balanced', random_state=42))
        ],
        voting='soft', n_jobs=-1
    ),
    X_sm, y_sm, X_test, y_test
)

# ── Figure 19: 앙상블 모델 평가 ──────────────────────────────────────────────
print("\n[그래프] Figure 19 생성 중...", flush=True)
ensemble_results = [rf_res, gb_res, voting_res]
colors_e = [GREEN, PURPLE, ORANGE]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Figure 19. Ensemble Models – Evaluation', fontsize=14, fontweight='bold')

# (a) ROC Curve
for res, c in zip(ensemble_results, colors_e):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[0].plot(fpr, tpr, color=c, lw=2,
                 label=f"{res['name']} (AUC={res['roc_auc']:.3f})")
axes[0].plot([0,1],[0,1],'k--', lw=1)
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
axes[0].set_title('(a) ROC Curve', fontweight='bold')
axes[0].legend(fontsize=8)

# (b) PR Curve
for res, c in zip(ensemble_results, colors_e):
    prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
    axes[1].plot(rec, prec, color=c, lw=2,
                 label=f"{res['name']} (AP={res['avg_prec']:.3f})")
axes[1].axhline(y_test.mean(), color='k', linestyle='--', lw=1, label='Random')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('(b) Precision-Recall Curve', fontweight='bold')
axes[1].legend(fontsize=8)

# (c) 지표 비교
metrics      = ['acc', 'f1_macro', 'f1_fail', 'roc_auc']
metric_labels = ['Accuracy', 'F1 Macro', 'F1 (Fail)', 'ROC-AUC']
x = np.arange(len(metrics))
w = 0.25
for i, (res, c) in enumerate(zip(ensemble_results, colors_e)):
    vals = [res[m] for m in metrics]
    axes[2].bar(x + i*w, vals, w, color=c, alpha=0.85,
                label=res['name'].split()[0])
axes[2].set_xticks(x + w)
axes[2].set_xticklabels(metric_labels, fontsize=9)
axes[2].set_ylim(0, 1.1)
axes[2].set_ylabel('Score')
axes[2].set_title('(c) Metric Comparison', fontweight='bold')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('../figures/fig19_ensemble_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: fig19_ensemble_evaluation.png")

# ── Figure 20: RF 특성 중요도 ─────────────────────────────────────────────────
print("[그래프] Figure 20 생성 중...", flush=True)
rf_importances = rf_res['model'].feature_importances_
top20_idx = np.argsort(rf_importances)[::-1][:20]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Figure 20. Random Forest – Feature Importance & Confusion Matrices',
             fontsize=14, fontweight='bold')

colors_imp = plt.cm.YlGn(np.linspace(0.3, 0.9, 20))
axes[0].barh(range(20), rf_importances[top20_idx][::-1],
             color=colors_imp[::-1], alpha=0.9)
axes[0].set_yticks(range(20))
axes[0].set_yticklabels([f'F{feat_names[i]}' for i in top20_idx][::-1], fontsize=8)
axes[0].set_xlabel('Feature Importance (Mean Decrease Impurity)')
axes[0].set_title('(a) RF Top-20 Feature Importances', fontweight='bold')

# Voting 혼동행렬
cm_v = voting_res['cm']
axes[1].imshow(cm_v, cmap='Purples')
axes[1].set_xticks([0,1]); axes[1].set_yticks([0,1])
axes[1].set_xticklabels(['Pass','Fail']); axes[1].set_yticklabels(['Pass','Fail'])
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
axes[1].set_title('(b) Soft Voting Confusion Matrix', fontweight='bold')
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm_v[i,j]), ha='center', va='center',
                     fontsize=18, fontweight='bold',
                     color='white' if cm_v[i,j] > cm_v.max()/2 else 'black')

plt.tight_layout()
plt.savefig('../figures/fig20_ensemble_detail.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: fig20_ensemble_detail.png")

# ── 저장 ─────────────────────────────────────────────────────────────────────
ensemble_save = {
    'rf':     rf_res,
    'gb':     gb_res,
    'voting': voting_res,
    'best_rf_n': best_n,
    'best_gb_params': best_p
}
with open('../results/ensemble_results.pkl', 'wb') as f:
    pickle.dump(ensemble_save, f)
print("✓ Saved: results/ensemble_results.pkl")

print("\n" + "=" * 60)
print("Ensemble Summary")
print("=" * 60)
print(f"{'Model':<25} {'Acc':>6} {'F1-Mac':>8} {'F1-Fail':>8} {'AUC':>7}")
print("-" * 60)
for r in ensemble_results:
    print(f"{r['name']:<25} {r['acc']:>6.4f} {r['f1_macro']:>8.4f} "
          f"{r['f1_fail']:>8.4f} {r['roc_auc']:>7.4f}")
