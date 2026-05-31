"""
SECOM Semiconductor Manufacturing Defect Detection
Step 1: Exploratory Data Analysis & Preprocessing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ── 스타일 설정 ──────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
BLUE   = '#2563EB'
RED    = '#DC2626'
GRAY   = '#6B7280'
ORANGE = '#F59E0B'
GREEN  = '#10B981'
PALETTE = [BLUE, RED]

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
print("=" * 60)
print("SECOM Dataset Loading & Basic Info")
print("=" * 60)

df = pd.read_csv('secom.csv')  # 경로 조정 필요)          # 경로는 실행 위치에 따라 조정
print(f"Shape: {df.shape}")
print(f"Columns: Time(1) + Features(590) + Label(1) = {df.shape[1]}")

# 타겟 분리
df['Time'] = pd.to_datetime(df['Time'])
df['label'] = df['Pass/Fail'].map({-1: 0, 1: 1})   # 0=Pass, 1=Fail
feature_cols = [c for c in df.columns if c not in ['Time', 'Pass/Fail', 'label']]

print(f"\nClass distribution:")
vc = df['label'].value_counts().sort_index()
print(f"  Pass (0): {vc[0]} ({vc[0]/len(df)*100:.1f}%)")
print(f"  Fail (1): {vc[1]} ({vc[1]/len(df)*100:.1f}%)")
print(f"  Imbalance ratio: 1:{vc[0]/vc[1]:.1f}")

# ── Figure 1: 클래스 불균형 & 시간 분포 ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Figure 1. Dataset Overview', fontsize=14, fontweight='bold', y=1.01)

# (a) 클래스 비율 파이차트
sizes = [vc[0], vc[1]]
labels_pie = [f'Pass\n(n={vc[0]})', f'Fail\n(n={vc[1]})']
wedges, texts, autotexts = axes[0].pie(
    sizes, labels=labels_pie, colors=[BLUE, RED],
    autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2))
for at in autotexts:
    at.set_fontsize(11)
axes[0].set_title('(a) Class Distribution', fontweight='bold')

# (b) 월별 불량 추이
df['month'] = df['Time'].dt.to_period('M')
monthly = df.groupby('month')['label'].agg(['sum', 'count'])
monthly['fail_rate'] = monthly['sum'] / monthly['count'] * 100
months_str = [str(m) for m in monthly.index]
axes[1].bar(range(len(months_str)), monthly['count'], color=BLUE, alpha=0.6, label='Total')
axes[1].bar(range(len(months_str)), monthly['sum'], color=RED, alpha=0.9, label='Fail')
axes[1].set_xticks(range(len(months_str)))
axes[1].set_xticklabels(months_str, rotation=45, ha='right', fontsize=8)
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Sample Count')
axes[1].set_title('(b) Monthly Sample & Fail Count', fontweight='bold')
axes[1].legend()

# (c) 결측치 분포
X = df[feature_cols]
miss_ratio = X.isnull().mean().values
bins = [0, 0.01, 0.1, 0.3, 0.5, 1.0]
labels_b = ['0%', '0~10%', '10~30%', '30~50%', '>50%']
counts_b = [
    (miss_ratio == 0).sum(),
    ((miss_ratio > 0) & (miss_ratio <= 0.1)).sum(),
    ((miss_ratio > 0.1) & (miss_ratio <= 0.3)).sum(),
    ((miss_ratio > 0.3) & (miss_ratio <= 0.5)).sum(),
    (miss_ratio > 0.5).sum()
]
colors_b = [GREEN, BLUE, ORANGE, RED, '#7C3AED']
bars = axes[2].bar(labels_b, counts_b, color=colors_b, edgecolor='white', linewidth=1.5)
for bar, cnt in zip(bars, counts_b):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 str(cnt), ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[2].set_xlabel('Missing Rate')
axes[2].set_ylabel('Number of Features')
axes[2].set_title('(c) Missing Value Distribution per Feature', fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/fig01_dataset_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: fig01_dataset_overview.png")

# ── Figure 2: 특성 통계 분포 ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Figure 2. Feature Statistics', fontsize=14, fontweight='bold')

# (a) 분산 분포 (log scale)
var_vals = X.var(ddof=0).dropna()
var_vals = var_vals[var_vals > 0]
axes[0].hist(np.log10(var_vals + 1e-10), bins=50, color=BLUE, edgecolor='white', alpha=0.85)
axes[0].set_xlabel('log10(Variance)')
axes[0].set_ylabel('Count')
axes[0].set_title('(a) Feature Variance Distribution', fontweight='bold')
axes[0].axvline(np.log10(var_vals.quantile(0.25)), color=RED, linestyle='--', label='Q1')
axes[0].legend()

# (b) 특성별 결측치 Top 20
miss_sorted = X.isnull().mean().sort_values(ascending=False)[:20]
axes[1].barh(range(20), miss_sorted.values[::-1], color=ORANGE, alpha=0.85)
axes[1].set_yticks(range(20))
axes[1].set_yticklabels([f'F{c}' for c in miss_sorted.index[::-1]], fontsize=7)
axes[1].set_xlabel('Missing Rate')
axes[1].set_title('(b) Top 20 Features with Most Missing Values', fontweight='bold')
axes[1].axvline(0.5, color=RED, linestyle='--', linewidth=1.5, label='50% threshold')
axes[1].legend()

# (c) Pass vs Fail: 몇 가지 대표 특성 분포
# 클래스별 평균 차이가 큰 특성 Top 5 선택
X_labeled = X.copy()
X_labeled['label'] = df['label'].values
means_diff = abs(X_labeled.groupby('label').mean().diff().iloc[-1]).sort_values(ascending=False)
top5_feats = means_diff.dropna().head(5).index.tolist()

x_pos = np.arange(len(top5_feats))
w = 0.35
pass_means = X_labeled[X_labeled['label']==0][top5_feats].mean()
fail_means = X_labeled[X_labeled['label']==1][top5_feats].mean()
# 정규화해서 비교
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
pm_norm = sc.fit_transform(pass_means.values.reshape(-1,1)).flatten()
fm_norm = sc.fit_transform(fail_means.values.reshape(-1,1)).flatten()
axes[2].bar(x_pos - w/2, pm_norm, w, label='Pass', color=BLUE, alpha=0.85)
axes[2].bar(x_pos + w/2, fm_norm, w, label='Fail', color=RED, alpha=0.85)
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels([f'F{c}' for c in top5_feats], rotation=30, fontsize=9)
axes[2].set_ylabel('Normalized Mean')
axes[2].set_title('(c) Top-5 Discriminative Features\n(Pass vs Fail Mean)', fontweight='bold')
axes[2].legend()

plt.tight_layout()
plt.savefig('../figures/fig02_feature_stats.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: fig02_feature_stats.png")

# ── Figure 3: 상관관계 히트맵 (결측치 제거 후 상위 특성) ─────────────────────
from sklearn.feature_selection import f_classif

X_clean = X.dropna(axis=1, thresh=len(X)*0.5)  # 50% 이상 결측 컬럼 제거
# 상수 컬럼 제거
X_clean = X_clean.loc[:, X_clean.var() > 0]
# 결측치 중앙값 대체 (임시)
X_fill = X_clean.fillna(X_clean.median())
y = df['label'].values

# ANOVA F-score로 상위 30개 선택
f_scores, _ = f_classif(X_fill, y)
top30_idx = np.argsort(f_scores)[::-1][:30]
top30_cols = X_fill.columns[top30_idx]

corr_mat = X_fill[top30_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask=mask, ax=ax,
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            xticklabels=[f'F{c}' for c in top30_cols],
            yticklabels=[f'F{c}' for c in top30_cols],
            linewidths=0.3, cbar_kws={'shrink': 0.8})
ax.set_title('Figure 3. Correlation Heatmap of Top-30 Discriminative Features',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/fig03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: fig03_correlation_heatmap.png")

# ── 전처리 파이프라인 ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Preprocessing Pipeline")
print("=" * 60)

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Step 1: 50% 이상 결측 컬럼 제거
miss_frac = X.isnull().mean()
cols_drop_miss = miss_frac[miss_frac > 0.5].index.tolist()
print(f"Step1 - Drop >50% missing columns: {len(cols_drop_miss)} removed")
X1 = X.drop(columns=cols_drop_miss)

# Step 2: 분산 0(상수) 컬럼 제거
const_cols = [c for c in X1.columns if X1[c].nunique() <= 1]
print(f"Step2 - Drop zero-variance columns: {len(const_cols)} removed")
X2 = X1.drop(columns=const_cols)

print(f"Remaining features: {X2.shape[1]}")

# Step 3: 중앙값 대치
imputer = SimpleImputer(strategy='median')
X3 = pd.DataFrame(imputer.fit_transform(X2), columns=X2.columns)

# Step 4: Near-zero variance 추가 제거 (분산 하위 5%)
var_threshold = X3.var().quantile(0.05)
low_var_cols = X3.var()[X3.var() < var_threshold].index.tolist()
print(f"Step4 - Drop near-zero variance (bottom 5%): {len(low_var_cols)} removed")
X4 = X3.drop(columns=low_var_cols)
print(f"Final feature count: {X4.shape[1]}")

# Step 5: Train/Test split (stratified, 80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X4.values, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain: {X_train.shape[0]} (Fail: {y_train.sum()})")
print(f"Test:  {X_test.shape[0]}  (Fail: {y_test.sum()})")

# Step 6: StandardScaler
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# 저장
import pickle, os
os.makedirs('../results', exist_ok=True)
save_dict = {
    'X_train': X_train_sc, 'X_test': X_test_sc,
    'y_train': y_train, 'y_test': y_test,
    'feature_names': list(X4.columns),
    'scaler': scaler
}
with open('../results/preprocessed_data.pkl', 'wb') as f:
    pickle.dump(save_dict, f)
print("\n✓ Saved: results/preprocessed_data.pkl")

# ── Figure 4: 전처리 요약 플로우 ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Figure 4. Preprocessing Results', fontsize=14, fontweight='bold')

# (a) 전처리 단계별 특성 수 변화
steps = ['Raw\n(590)', 'Remove\n>50% NaN\n(−28)', 'Remove\nConst.\n(−116)', 
         'Remove\nLow Var.\n(−28)', 'Final\nFeatures']
counts = [590, 562, 446, 418, X4.shape[1]]
colors_s = [GRAY, ORANGE, ORANGE, ORANGE, GREEN]
bars = axes[0].bar(steps, counts, color=colors_s, edgecolor='white', linewidth=1.5, width=0.6)
for bar, cnt in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
                 str(cnt), ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Number of Features')
axes[0].set_title('(a) Feature Reduction Steps', fontweight='bold')
axes[0].set_ylim(0, 650)

# (b) Train/Test 클래스 분포
train_vc = pd.Series(y_train).value_counts().sort_index()
test_vc  = pd.Series(y_test).value_counts().sort_index()
x_pos = np.array([0, 1])
w = 0.35
axes[1].bar(x_pos - w/2, [train_vc[0], train_vc[1]], w, color=[BLUE, RED], 
            label='Train', alpha=0.85)
axes[1].bar(x_pos + w/2, [test_vc[0], test_vc[1]], w, color=[BLUE, RED],
            label='Test', alpha=0.5, hatch='//')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(['Pass (0)', 'Fail (1)'])
axes[1].set_ylabel('Count')
axes[1].set_title('(b) Train/Test Class Distribution\n(Stratified 80:20 Split)', fontweight='bold')
axes[1].legend()
for i, (tr, te) in enumerate(zip([train_vc[0], train_vc[1]], [test_vc[0], test_vc[1]])):
    axes[1].text(i - w/2, tr + 2, str(tr), ha='center', fontsize=10)
    axes[1].text(i + w/2, te + 2, str(te), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('../figures/fig04_preprocessing_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: fig04_preprocessing_summary.png")

print("\n" + "=" * 60)
print("Step 1 (EDA & Preprocessing) Complete!")
print("=" * 60)
print(f"  Original features : 590")
print(f"  Final features    : {X4.shape[1]}")
print(f"  Train samples     : {X_train.shape[0]}")
print(f"  Test samples      : {X_test.shape[0]}")
print(f"  Class imbalance   : 1:{y_train.sum()/max((y_train==0).sum(),1):.2f} (Pass:Fail)")
