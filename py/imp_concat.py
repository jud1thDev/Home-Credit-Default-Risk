#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 20:00:33 2018

@author: kazuki.onodera
"""

import pandas as pd

# 여러 피처 중요도 파일 경로 리스트 (로컬 경로)
li = ['/Users/Kazuki/Downloads/imp_191_imp.py.csv',
      '/Users/Kazuki/Downloads/imp_192_imp.py.csv',
      '/Users/Kazuki/Downloads/imp_193_imp.py.csv',
      '/Users/Kazuki/Downloads/imp_194_imp.py.csv',
      '/Users/Kazuki/Downloads/imp_195_imp.py.csv',
      ]

# 피처명 수집용 리스트
tmp = []
for p in li:
    imp = pd.read_csv(p)
    # split 값이 0보다 크고, f15/f025로 시작하는 피처만 추출하여 리스트에 추가
    tmp += imp[imp['split']>0][imp.feature.str.startswith('f15')].feature.tolist()
    tmp += imp[imp['split']>0][imp.feature.str.startswith('f025')].feature.tolist()
    # (아래는 주석 처리된 대안 로직)
    # imp['split'] /= imp['split'].max()
    # imp['gain'] /= imp['gain'].max()
    # imp['total'] = imp['split'] + imp['gain']
    # tmp += imp[imp['total']>0.005].feature.tolist()

# (주석) 추가 피처 리스트 합치기 예시
#tmp += pd.read_csv('/Users/kazuki.onodera/Downloads/imp_801_imp_lgb.py-2.csv').head(1000).feature.tolist()

# 중복 제거 및 정렬 후 DataFrame으로 변환
tmp = sorted(set(tmp))
df = pd.DataFrame(tmp, columns=['feature'])

# 최소 1회 이상 사용된 피처 리스트를 CSV로 저장
df.to_csv('imp_atleast_use.csv', index=False)

# 두 개의 중요도 파일을 읽어와서 feature를 인덱스로 설정
imp1 = pd.read_csv('/Users/Kazuki/Downloads/imp_801_imp_lgb.py-2-1.csv').set_index('feature')
imp2 = pd.read_csv('/Users/Kazuki/Downloads/imp_801_imp_lgb.py-2-2.csv').set_index('feature')

# 두 중요도 파일의 total 값을 합산하여 내림차순 정렬 후 DataFrame으로 변환
tmp = (imp1.total + imp2.total).sort_values(ascending=False).to_frame().reset_index()

# 합산 결과를 CSV로 저장
tmp.to_csv('/Users/Kazuki/Downloads/imp_801_imp_lgb.py-2.csv', index=False)

