#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 12:39:16 2018

@author: kazuki.onodera
"""

import gc, os
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count, Pool
from collections import defaultdict
from glob import glob
import count
import utils, utils_cat
utils.start(__file__)
#==============================================================================

# 랜덤 시드 설정
SEED = np.random.randint(9999)
print('SEED:', SEED)

# 교차검증 폴드 수
NFOLD = 4

# 반복 학습 횟수 (LOOP)
LOOP = 2

# 피처 파일 재생성 여부
RESET = False

# 본인만의 실험 모드 여부
ONLY_ME = False

# 802_cv_lgb.py 추가 실행 여부
EXE_802 = False

# LightGBM 하이퍼파라미터 설정
defaults = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.01,
    'max_depth': 6,
    'num_leaves': 63,
    'max_bin': 255,
    'min_child_weight': 10,
    'min_data_in_leaf': 150,
    'reg_lambda': 0.5,  # L2 정규화
    'reg_alpha': 0.5,   # L1 정규화
    'colsample_bytree': 0.5,
    'subsample': 0.5,
    'nthread': cpu_count(),
    'bagging_freq': 1,
    'verbose':-1,
    'seed': SEED
}
param = defaults

# 피처 중요도 파일 리스트 (여러 실험 결과를 합침)
imp_files = ['LOG/imp_181_imp.py.csv', 
             'LOG/imp_182_imp.py.csv', 
             'LOG/imp_183_imp.py.csv', 
             'LOG/imp_184_imp.py.csv', 
             'LOG/imp_185_imp.py.csv', 
             'LOG/imp_790_imp.py.csv', ]

# =============================================================================
# 전체 데이터에서 사용할 피처 파일 리스트 생성
# =============================================================================
files_tr = []

for p in imp_files:
    imp = pd.read_csv(p)
    imp = imp[imp.split>2]  # split 값이 2 초과인 중요 피처만 사용
    files_tr += ('../feature/train_' + imp.feature + '.f').tolist()

files_tr = sorted(set(files_tr))  # 중복 제거 및 정렬

print('features:', len(files_tr))

# 피처 파일을 모두 읽어와서 하나의 데이터프레임으로 합침
X = pd.concat([
                pd.read_feather(f) for f in tqdm(files_tr, mininterval=60)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET  # 타겟값 로드

#X['nejumi'] = np.load('../feature_someone/train_nejumi.npy')  # (주석 처리됨)

# 컬럼 중복 체크
if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

# 범주형 피처 리스트 추출
CAT = list( set(X.columns)&set(utils_cat.ALL))
print(f'CAT: {CAT}')

# =============================================================================
# 교차검증(CV) LightGBM 학습
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=CAT, free_raw_data=False)
gc.collect()

model_all = []
for i in range(LOOP):
    gc.collect()
    # LightGBM 교차검증 수행, 모델 저장
    ret, models = lgb.cv(param, dtrain, 9999, nfold=NFOLD, 
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    model_all += models

# 최종 CV 결과 출력
result = f"CV auc-mean: {ret['auc-mean'][-1]} + {ret['auc-stdv'][-1]}"
print(result)

# LINE 알림 전송(외부 함수)
utils.send_line(result)
# 피처 중요도 산출 및 정규화
imp = ex.getImp(model_all)
imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']

imp.sort_values('total', ascending=False, inplace=True)
imp.reset_index(drop=True, inplace=True)

# 중요도 결과 저장
imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)

# (아래는 사용하지 않는 코드, 필요시 주석 해제)
#def multi_touch(arg):
#    os.system(f'touch "../feature_unused/{arg}.f"')
#col = imp[imp['split']==0]['feature'].tolist()
#pool = Pool(cpu_count())
#pool.map(multi_touch, col)
#pool.close()

# 802_cv_lgb.py 추가 실행 (옵션)
if EXE_802:
    os.system(f'nohup python -u 802_cv_lgb.py > LOG/log_802_cv_lgb.py.txt &')

#==============================================================================
utils.end(__file__)
#utils.stop_instance()

