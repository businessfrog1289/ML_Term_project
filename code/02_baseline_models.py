"""
SECOM Semiconductor Manufacturing Defect Detection
Step 2: Baseline Models
  - Logistic Regression (main baseline)
  - LDA
  - Naive Bayes
모두 클래스 불균형 처리(class_weight='balanced' / SMOTE)를 적용
"""
import os
os.makedirs('../figures', exist_ok=True)
os.makedirs('../results', exist_ok=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle, warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

BLUE  = '#2563EB'; RED = '#DC2626'; GREEN = '#10B981'; ORANGE = '#F59E0B'; GRAY='#6B7280'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
with open('../results/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
X_train, X_test   = data['X_train'], data['X_test']
y_train, y_test   = data['y_train'], data['y_test']

print("=" * 60)
print("Step 2: Baseline Models")
print("=" * 60)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── 평가 헬퍼 ────────────────────────────────────────────────────────────────
def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred  = model.predict(X_te)
    y_prob  = model.predict_proba(X_te)[:, 1]

    acc  = accuracy_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred, average='macro')
    f1b  = f1_score(y_te, y_pred, pos_label=1)     # binary (Fail class)
    auc  = roc_auc_score(y_te, y_prob)
    ap   = average_precision_score(y_te, y_prob)
    cm   = confusion_matrix(y_te, y_pred)

    # 5-fold CV AUC
    cv_auc = cross_val_score(model, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                             scoring='roc_auc').mean()

    print(f"\n[{name}]")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Macro : {f1:.4f}")
    print(f"  F1 Fail  : {f1b:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  Avg Prec : {ap:.4f}")
    print(f"  CV AUC   : {cv_auc:.4f}")
    print(classification_report(y_te, y_pred, target_names=['Pass', 'Fail']))
    return {
        'name': name, 'model': model,
        'y_pred': y_pred, 'y_prob': y_prob,
        'acc': acc, 'f1_macro': f1, 'f1_fail': f1b,
        'roc_auc': auc, 'avg_prec': ap, 'cv_auc': cv_auc, 'cm': cm
    }

# ── 모델 학습 ────────────────────────────────────────────────────────────────
results = []

# 1. Logistic Regression (Baseline)
lr = LogisticRegression(max_iter=1000, class_weight='balanced',
                        solver='lbfgs', random_state=42)
results.append(evaluate('Logistic Regression (Baseline)', lr,
                         X_train, y_train, X_test, y_test))

# 2. LDA
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
results.append(evaluate('LDA', lda, X_train, y_train, X_test, y_test))

# 3. Naive Bayes (Gaussian)
nb = GaussianNB(var_smoothing=1e-9)
results.append(evaluate('Naive Bayes', nb, X_train, y_train, X_test, y_test))

# ── Figure 5: ROC & PR 곡선 ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Figure 5. Baseline Model Evaluation', fontsize=14, fontweight='bold')

colors = [BLUE, GREEN, ORANGE]
names_short = ['LR', 'LDA', 'NB']

# (a) ROC Curve
for res, c, ns in zip(results, colors, names_short):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[0].plot(fpr, tpr, color=c, lw=2,
                 label=f"{ns} (AUC={res['roc_auc']:.3f})")
axes[0].plot([0,1],[0,1],'k--', lw=1)
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('(a) ROC Curve', fontweight='bold')
axes[0].legend(fontsize=9)

# (b) Precision-Recall Curve
for res, c, ns in zip(results, colors, names_short):
    prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
    axes[1].plot(rec, prec, color=c, lw=2,
                 label=f"{ns} (AP={res['avg_prec']:.3f})")
axes[1].axhline(y_test.mean(), color='k', linestyle='--', lw=1, label='Random')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('(b) Precision-Recall Curve', fontweight='bold')
axes[1].legend(fontsize=9)

# (c) 지표 비교 막대
metrics = ['acc', 'f1_macro', 'f1_fail', 'roc_auc']
metric_labels = ['Accuracy', 'F1 Macro', 'F1 (Fail)', 'ROC-AUC']
x = np.arange(len(metrics))
w = 0.25
for i, (res, c, ns) in enumerate(zip(results, colors, names_short)):
    vals = [res[m] for m in metrics]
    axes[2].bar(x + i*w, vals, w, label=ns, color=c, alpha=0.85)
axes[2].set_xticks(x + w)
axes[2].set_xticklabels(metric_labels, fontsize=9)
axes[2].set_ylim(0, 1.05)
axes[2].set_ylabel('Score')
axes[2].set_title('(c) Metric Comparison', fontweight='bold')
axes[2].legend()

plt.tight_layout()
plt.savefig('../figures/fig05_baseline_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Saved: fig05_baseline_evaluation.png")

# ── Figure 6: Confusion Matrices ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle('Figure 6. Confusion Matrices – Baseline Models', fontsize=14, fontweight='bold')

for ax, res, ns in zip(axes, results, names_short):
    cm = res['cm']
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pass','Fail']); ax.set_yticklabels(['Pass','Fail'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'({ns})', fontweight='bold')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color='white' if cm[i,j] > cm.max()/2 else 'black')

plt.tight_layout()
plt.savefig('../figures/fig06_baseline_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: fig06_baseline_confusion.png")

# ── 결과 저장 ─────────────────────────────────────────────────────────────────
baseline_summary = [{k: v for k, v in r.items() if k != 'model'} for r in results]
with open('../results/baseline_results.pkl', 'wb') as f:
    pickle.dump({'results': results, 'summary': baseline_summary}, f)

print("\n" + "=" * 60)
print("Baseline Summary")
print("=" * 60)
print(f"{'Model':<30} {'Acc':>6} {'F1-Mac':>8} {'F1-Fail':>8} {'AUC':>7}")
print("-" * 65)
for r in results:
    print(f"{r['name']:<30} {r['acc']:>6.4f} {r['f1_macro']:>8.4f} {r['f1_fail']:>8.4f} {r['roc_auc']:>7.4f}")
