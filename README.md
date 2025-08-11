## Home-Credit-Default-Risk 파이프라인 가이드

### 목적
- 이 프로젝트는 Kaggle Home Credit Default Risk 대회에서 대출 연체 가능성(TARGET)을 예측하기 위한 피처 엔지니어링 + LightGBM 파이프라인입니다.
- 데이터셋별(Previous/POS/Installments/Credit/Bureau/BB)로 많은 파생 피처를 생성하고, 중요도 기반으로 모델을 최적화합니다.

### 디렉터리 개요
- `py/`: 메인 파이프라인 스크립트(피처 생성, 학습, 평가, 예측)와 유틸리티
- `py_prev/`, `py_bureau/`: 서브 데이터셋 전용 파이프라인(분리 실험)
- `jn/`: EDA, Leak Hunt, Adversarial Validation 등 분석 노트북/스크립트
- `feature/`: 생성된 피처 저장 경로(feather, train_/test_ 접두사). .gitignore로 제외됨
- `data/`: 전처리된 데이터(pickle 분할 등). .gitignore로 제외됨
- `LOG/`: 실행 로그 및 중요도 CSV 저장 경로

### 개발 환경
- Python 3.6+
- pandas, numpy, scikit-learn, lightgbm, tqdm, pyarrow(feather), requests
- 설치: `pip install pandas numpy scikit-learn lightgbm tqdm pyarrow requests`

### 전체 흐름(High-level)
1) 데이터 준비 → 2) 피처 생성 → 3) 피처 검증 → 4) 중요도 산출/베이스 학습 → 5) 상위 N 피처 CV 스윕 → 6) 최종 학습 → 7) 예측/제출 → 8) 블렌딩(선택)

---

## 0. 데이터 준비
- `init.txt` 참고(원본 Kaggle 데이터 다운로드 → `input/` 이동)
- 이 저장소에서는 전처리된 pickles를 `data/` 아래(예: `../data/train`, `../data/test`)로 분할 저장하여 사용합니다.
- 샘플 추출 예시는 `SAMPLE CSV.ipynb` 참고

## 1. 피처 생성(도메인별 스크립트)
- 위치: `py/` 폴더의 번호대별 스크립트
  - `100~199`: Previous Application 집계/파생 (예: `101_aggregate.py`, `108_latest.py`, `109_first.py`)
  - `200~299`: POS CASH (예: `201_aggregate.py`, `205_aggregate-6m.py`)
  - `300~399`: Installments (예: `301_aggregate.py`, `306_days_payment.py`, `311_month.py`)
  - `400~499`: Credit Card (예: `401_aggregate.py`, `405_aggregate-6m.py`)
  - `500~599`: Bureau (예: `501_aggregate.py`, `509_latest.py`, `510_first.py`)
  - `600~699`: Bureau Balance (예: `601_aggregate.py`)
  - `700~799`: 기타/사전처리 (예: `702_prev_imputation.py`)
- 공용 집계 정의: `py/utils_agg.py`
- 공용 유틸: `py/utils.py`, `py/utils_cat.py`
- 각 스크립트는 보통 `../feature/train_*.f`, `../feature/test_*.f` 형태의 feather 파일을 생성합니다.

### 실행 예시(리눅스/WSL)
```bash
python py/201_aggregate.py
python py/301_aggregate.py
# 필요 도메인/윈도우/전략 조합을 순차 실행
```

### 실행 예시(Windows PowerShell)
```powershell
python py/201_aggregate.py
python py/301_aggregate.py
```

## 2. 피처 파일 검증
- 목적: train/test 쌍 피처가 모두 존재하는지 확인
- 함수: `utils.check_feature()`

예시(Python 셸):
```python
import py.utils as u
u.check_feature()
```

## 3. 베이스 학습 및 피처 중요도 산출
- 스크립트: `py/801_imp_lgb.py`
- 동작:
  - `LOG/imp_***.csv`(여러 실험의 중요도)로부터 사용할 피처 목록 취합
  - `../feature/train_*.f`를 모두 로드하여 `X` 구성, `../data/label`에서 `y` 로드
  - LightGBM `lgb.cv`로 베이스 성능 측정 및 모델 수집
  - `lgbextension.getImp`로 중요도 산출(split/gain 정규화, total=합산)
  - 저장: `LOG/imp_801_imp_lgb.py*.csv`

### 실행 예시
```bash
python py/801_imp_lgb.py
```

## 4. 상위 N 피처 CV 스윕
- 스크립트: `py/802_cv_lgb.py`
- 동작:
  - `LOG/imp_801_imp_lgb.py-2.csv` 기준으로 total 상위 N개(500~2300, 100단위)를 순회
  - 각 N에 대해 `../feature/train_*.f` 로드 → LightGBM CV → AUC 로깅/알림
  - 선택적으로 `../data/drop_ids.csv`를 통해 특정 ID 제외

### 실행 예시
```bash
python py/802_cv_lgb.py
```

## 5. 최종 학습(옵션 스크립트)
- 변형 예시: `py/804_cv_lgb.py`, `py/803_adv_imp_lgb.py` 등
  - 교차검증 전략/파라미터 변경, Adversarial Validation 등 고급 실험

## 6. 예측/제출
- 스크립트: `py/931_predict_906-1.py` ~ `py/935_predict_908-2.py` 등
- 동작: 학습된 모델로 test 예측 → 제출 파일 생성
- 제출: `utils.submit(file_path, comment)` 또는 Kaggle CLI 직접 사용

### 실행 예시
```bash
python py/933_predict_907-2.py
```

## 7. 블렌딩(선택)
- 노트북/스크립트: `jn/` 내 `LB.ipynb`, `stochastic_blending_*.csv` 등
- 목적: 서로 다른 모델/피처 조합의 예측을 가중 평균/스태킹

---

## 공용 스크립트 요점
- `py/utils.py`
  - 데이터 로딩(`read_pickles`, `load_train/test`), 피처 저장(`to_feature`, `to_pickles`), 메모리 최적화(`reduce_mem_usage`)
  - 제출/알림(`submit`, `send_line`)
- `py/utils_agg.py`
  - 도메인별 집계 사전 정의(prev/pos/ins/cre/bure/bb). 차분/증감율/비율/윈도우 집계 포함
- `py/utils_cat.py`
  - 범주형 피처 목록 등 카테고리 관련 정의
- `py/run.py`
  - 긴 작업을 백그라운드로 실행(리눅스/WSL). Windows는 직접 `python ...` 실행 권장

---

## 실행 팁(Windows / WSL)
- `py/run.py`는 `nohup ... &`를 사용하므로 Windows PowerShell에서는 동작하지 않습니다.
  - 대안(Windows): `python py/801_imp_lgb.py > LOG\log_801.txt 2>&1`
  - 또는 WSL/리눅스 환경에서 `python py/run.py 801_imp_lgb.py` 사용
- feather 파일 사용을 위해 `pyarrow` 설치 필요

## 최소 실행 순서(Quick Start)
1) 피처 생성 스크립트 실행(필요 도메인만 선별 가능)
2) `python py/801_imp_lgb.py`로 중요도/베이스 성능 산출
3) `python py/802_cv_lgb.py`로 상위 N 피처 스윕
4) 최종 학습 스크립트(옵션) → 예측 스크립트 실행 → 제출

## 문제 해결(FAQ)
- 피처 누락 에러: `utils.check_feature()`로 train/test 쌍 확인
- 메모리 부족: `utils.reduce_mem_usage` 적용, 피처 수 축소(802에서 N 조정)
- Windows에서 백그라운드 실행 문제: PowerShell 리디렉션 사용 또는 WSL 이용

---

### 부록: 로그/중요도 관리
- 실행 로그: `LOG/` 폴더의 `log_*.txt`
- 중요도 CSV: `LOG/imp_*.csv`
- 중요도 병합/필터링 예시: `py/imp_concat.py` 