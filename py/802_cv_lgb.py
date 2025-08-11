#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 01:29:16 2018

@author: Kazuki

nohup python run2.py 801_imp_lgb.py 802_cv_lgb.py &

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
from glob import glob
from sklearn.model_selection import GroupKFold
import count
import utils, utils_cat
utils.start(__file__)
#==============================================================================

# 랜덤 시드 설정
SEED = np.random.randint(9999)

# 교차검증 폴드 수
NFOLD = 7

# 실험할 피처 개수 리스트(500~2300, 100 단위)
HEADS = list(range(500, 2300, 100))

# LightGBM 하이퍼파라미터 설정
param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.01,
         'max_depth': 6,
         'num_leaves': 63,
         'max_bin': 255,
         'min_child_weight': 10,
         'min_data_in_leaf': 150,
         'reg_lambda': 0.5,  # L2 정규화
         'reg_alpha': 0.5,  # L1 정규화
         'colsample_bytree': 0.9,
         'subsample': 0.9,
         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         'seed': SEED
         }

# =============================================================================
# 데이터 및 피처 중요도 로드
# =============================================================================
imp = pd.read_csv('LOG/imp_801_imp_lgb.py-2.csv')  # 피처 중요도 파일
imp.sort_values('total', ascending=False, inplace=True)

y = utils.read_pickles('../data/label').TARGET  # 타겟값 로드

drop_ids = pd.read_csv('../data/drop_ids.csv')['SK_ID_CURR']  # 제외할 ID
SK_ID_CURR = utils.load_train(['SK_ID_CURR'])  # 학습 데이터 ID

# =============================================================================
# (주석) GroupKFold 등 다양한 폴드 전략 실험 코드 존재
# =============================================================================

# =============================================================================
# 피처 개수별로 반복 실험
# =============================================================================
for HEAD in HEADS:
    files = ('../feature/train_' + imp.head(HEAD).feature + '.f').tolist()  # 상위 N개 피처만 사용
    
    # 피처 파일 로드 및 합치기
    X = pd.concat([
                    pd.read_feather(f) for f in tqdm(files, mininterval=60)
                   ], axis=1)
    
    # =============================================================================
    # 제외할 ID 제거
    # =============================================================================
    X['SK_ID_CURR'] = SK_ID_CURR
    y = y[~X.SK_ID_CURR.isin(drop_ids)]
    X = X[~X.SK_ID_CURR.isin(drop_ids)].drop('SK_ID_CURR', axis=1)
    
    # 컬럼 중복 체크
    if X.columns.duplicated().sum()>0:
        raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
    print('no dup :) ')
    print(f'X.shape {X.shape}')
    
    gc.collect()
    
    # 범주형 피처 추출
    CAT = list( set(X.columns)&set(utils_cat.ALL))
    
    # =============================================================================
    # 교차검증(CV) LightGBM 학습
    # =============================================================================
    dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
    gc.collect()
    
    ret, models = lgb.cv(param, dtrain, 9999, nfold=NFOLD,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    
    # 결과 출력 및 알림
    result = f"CV auc-mean({SEED}:{HEAD}): {ret['auc-mean'][-1]} + {ret['auc-stdv'][-1]}"
    print(result)
    
    utils.send_line(result)

#==============================================================================
utils.end(__file__)
#utils.stop_instance()


