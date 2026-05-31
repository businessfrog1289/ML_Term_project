# SECOM 반도체 제조 공정 불량 탐지
**[2026-1] Machine Learning Term Project**  

---

## 문제 정의

반도체 제조 공정에서 수집된 590개의 익명 센서 신호를 바탕으로 **불량 웨이퍼(Fail)를 사전에 탐지**하는 이진 분류 모델을 구축한다.

- **핵심 과제**: 심각한 클래스 불균형 (Pass 93.4% vs Fail 6.6%, 약 14:1)
- **평가 지표**: ROC-AUC, F1-Fail (불량 클래스 F1), Accuracy

---

## 데이터셋

| 항목 | 내용 |
|------|------|
| 출처 | https://www.kaggle.com/datasets/paresh2047/uci-semcom |
| 기간 | 2008.01 ~ 2008.12 |
| 샘플 수 | 1,567 |
| 특성 수 | 590 (연속형 익명 센서 신호) |
| 타겟 | -1=Pass / 1=Fail |
| 결측치 | 약 4.5% |

---

## 프로젝트 구조

```
ML_Term_project/
├── secom.csv                       ← 원본 데이터
├── requirements.txt
├── README.md
├── code/
│   ├── 01_eda_preprocessing.py     # EDA & 전처리
│   ├── 02_baseline_models.py       # LR(Baseline) · LDA · NB
│   ├── 03_glm_regularization.py    # Ridge · Lasso · ElasticNet
│   ├── 04_svm.py                   # SVM + GridSearch
│   ├── 05_decision_tree.py         # DT + 1-SE Pruning
│   ├── 06_final_comparison.py      # 전체 모델 비교 (앙상블 포함)
│   └── 07_ensemble.py              # Random Forest · GradBoost · Soft Voting
├── figures/                        ← 생성된 그래프 (실행 시 자동 생성)
└── results/                        ← pkl / csv 결과 (실행 시 자동 생성)
```

> `figures/`와 `results/` 폴더는 코드 실행 시 자동으로 생성되므로 저장소에 포함되지 않는다.

---

## 환경 설정 및 실행 방법

### 1. Python 버전 확인

```bash
python --version   # Python 3.9 이상 권장
```

### 2. 가상환경 생성 및 활성화

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 실행 순서

코드는 `code/` 폴더 안에서 실행한다. `secom.csv`는 `code/`의 **상위 폴더(프로젝트 루트)** 에 위치해야 한다.

```
ML_Term_project/   ← 여기에 secom.csv가 있어야 함
└── code/          ← 여기서 아래 명령 실행
```

```bash
cd code

python 01_eda_preprocessing.py    # EDA & 전처리    → ../figures/fig01~04
python 02_baseline_models.py      # Baseline        → ../figures/fig05~06
python 03_glm_regularization.py   # GLM             → ../figures/fig07~08
python 04_svm.py                  # SVM             → ../figures/fig09~10
python 05_decision_tree.py        # Decision Tree   → ../figures/fig11~14
python 07_ensemble.py             # Ensemble        → ../figures/fig19~20
python 06_final_comparison.py     # 최종 비교       → ../figures/fig15~18
```

> **주의**: 반드시 위 순서대로 실행해야 한다. 각 스크립트는 이전 단계에서 `../results/`에 저장된 `.pkl` 파일을 읽는다.  
> `06_final_comparison.py`는 `07_ensemble.py` 이후에 실행한다.

---

## 모델 및 방법론 요약

| 단계 | 모델 | 핵심 선택 이유 |
|------|------|---------------|
| Baseline | Logistic Regression | 선형 기준선, 최고 해석 가능성 |
| Baseline | LDA, Naive Bayes | 분포 가정 기반 비교 |
| GLM | Ridge / Lasso / ElasticNet | 정규화, 변수 선택, 다중공선성 처리 |
| Kernel | SVM (RBF) | 비선형 결정 경계, 마진 최대화 |
| Tree | Decision Tree + 1-SE | 해석 가능성, 과적합 제어 |
| Ensemble | Random Forest | 배깅, 분산 감소 |
| Ensemble | Gradient Boosting | 부스팅, 편향 감소 |
| Ensemble | Soft Voting | 다양성 결합, 예측 안정성 |

**공통 처리**: SMOTE (클래스 불균형), StandardScaler, 5-Fold Stratified CV

---

## 주요 결과

| 모델 | Accuracy | F1 (Fail) | ROC-AUC |
|------|----------|-----------|---------|
| LR (Baseline) | 0.8408 | 0.1071 | 0.6200 |
| SVM (RBF) | 0.9140 | 0.1818 | 0.6938 |
| Random Forest | 0.9331 | 0.0000 | 0.7529 |
| Gradient Boosting | 0.9268 | 0.1481 | 0.7144 |

- ROC-AUC 최고: **Random Forest (0.7529)** — 단, F1-Fail = 0.000으로 실질적 탐지 불가
- 균형 성능 최고: **SVM RBF (AUC 0.6938, F1-Fail 0.1818)**

전체 결과는 실행 후 생성되는 `results/final_comparison.csv` 참고.

---

## 생성형 AI 활용 내역

본 프로젝트에서 **Claude (Anthropic)** 를 다음 목적으로 활용하였다:
- 코드 구조 설계 및 디버깅 보조
- 보고서 문단 초안 작성

모든 분석 로직, 모델 선택 근거, 결과 해석은 직접 작성·검토하였다.

---

## 참고문헌

- Chawla et al. (2002). SMOTE. *JAIR*, 16, 321–357.
- Cortes & Vapnik (1995). Support-vector networks. *Machine Learning*, 20(3).
- Tibshirani (1996). Lasso. *JRSS-B*, 58(1).
- Breiman (2001). Random Forests. *Machine Learning*, 45(1).
- Friedman (2001). Gradient Boosting Machine. *Annals of Statistics*, 29(5).
- Pedregosa et al. (2011). Scikit-learn. *JMLR*, 12.
