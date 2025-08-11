#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 19:37:49 2018

@author: Kazuki
"""

import os
from time import sleep
import sys

# 명령행 인자(argv)로 실행할 파일명과(필수) 대기 시간(선택)을 받음
argv = sys.argv

file = argv[1]  # 실행할 파이썬 파일명 (예: 801_imp_lgb.py)
if len(argv)>2:
    sec = 60 * int(argv[2])  # 두 번째 인자가 있으면 분 단위로 변환하여 대기 시간 설정
    print(f'wait {sec} sec')
else:
    sec = 0  # 대기 시간 없으면 0초

sleep(sec)  # 지정된 시간만큼 대기
# nohup으로 백그라운드에서 파일 실행, 로그는 LOG/log_{file}.txt에 저장
os.system(f'nohup python -u {file} > LOG/log_{file}.txt &')

