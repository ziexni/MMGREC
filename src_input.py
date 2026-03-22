"""
make_src_mtx.py
===============
SASRec 방식: timestamp 순서 기반 시퀀스 → src_mtx (인코더 입력)
- 마지막 2개 (val/test)를 제외한 train 인터랙션만 사용
- 최근 src_len 개 아이템을 오른쪽 정렬, 왼쪽 0 패딩
"""

import numpy as np
import pickle

f_para = open('load.para', 'rb')
para = pickle.load(f_para)
user_num = para['user_num']
item_num = para['item_num']
train_matrix = para['train_matrix']

train_matrix.data = np.ones_like(train_matrix.data, dtype=np.int8)
train_arr = train_matrix.toarray()   # (user_num, item_num)

src_len = 50   # SASRec 기본 시퀀스 길이 (원본 32 → 50으로 확대)
# train_ui는 이미 timestamp 순 정렬되어 있으므로
# 각 유저 행에서 1인 item idx를 순서대로 추출
src_mtx = np.zeros((user_num, src_len), dtype=np.int32)
index   = np.arange(item_num)

for u, row in enumerate(train_arr):
    itr = index[row == 1] + 1   # 1-indexed (0 = pad)
    if len(itr) >= src_len:
        u_src = itr[-src_len:]  # 가장 최근 src_len개
    else:
        pad   = np.zeros(src_len - len(itr), dtype=np.int32)
        u_src = np.concatenate([pad, itr])  # 왼쪽 패딩
    src_mtx[u] = u_src

np.save('src_mtx.npy', src_mtx)
print(f'src_mtx saved: shape={src_mtx.shape}')