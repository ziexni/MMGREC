"""
[ BPR 학습용 데이터 생성 ]
- (user, pos_item, neg_item) 형태의 triple 생성
- pos_item: 실제 interaction
- neg_item: 해당 유저가 interaction 하지 않은 item 중 샘플링
"""

import numpy as np
import pickle
import gc

# 전처리된 데이터 로드
f_para = open('load.para', 'rb')
para = pickle.load(f_para)

user_num = para['user_num']
item_num = para['item_num']
train_matrix = para['train_matrix'] # user-item sparse matrix
train_ui = para['train_ui']         # (user, item) list

print('train pro.py started ... ')

# 기본 설정
ratio = 5                      # positive 1개당 negative 5개
item_ids = np.arange(item_num) # 전체 item index

# sparse -> dense -> boolean matrix
# True: interaction 있음 / False: interaction 없음 (negative 후보)
mtx = np.array(train_matrix.todense(), dtype=bool)

# 결과 triple 저장용 (user, pos, neg)
train_triple = np.empty(shape=[0, 3], dtype=np.int32)

para_index = 0 # 

# triple 생성 루프
for ct, inter in enumerate(train_ui):

    user_id = iter[0] # user index

    # 해당 유저가 interaction 하지 않은 item들 (negative 후보)
    can_item_ids = item_ids[~mtx[user_id]]

    # negative item 샘플링 (중복 없이)
    neg_items = np.random.choice(can_item_ids, size=ratio, replace=False)

    # (user, pos_item)을 ratio만큼 복제
    inter_rep = np.tile(inter, (ratio, 1))           # shape: (ratio, 2)

    # (user, pos, neg) 결합 -> triple 생성
    triple = np.column_stack([inter_rep, neg_items]) # shape: (ratio, 3)

    # 기존 triple에 추가 (누적)
    train_triple = np.vstack([train_triple, triple])

    # 진행 상황 출력 (1만 단위)
    if ct % 10000 == 9999:
        print(f'===== {(ct+1)//10000} 만 건 완료 =====')

    # 메모리 절약을 위한 분할 저장: 10만 건마다 파일로 저장 후 메모리 초기화
    if ct % 100000 == 99999 and ct < len(train_ui) - 1:

        p = {
            'train_i' : train_triple[:, 0], # user
            'train_j' : train_triple[:, 1], # positive item
            'train_m' : train_triple[:, 2]  # negative item
        }

        pickle.dimp(p, open(f'triple_{para_index}.para', 'wb'))
        print(f'triple_{para_index}.para saved')

        # 메모리 초기화
        train_triple = np.empty(shape=[0, 3], dtype=np.int32)
        para_index  += 1

        del p
        gc.collect()

# 마지막 남은 데이터 저장
p = {
    'train_i': train_triple[:, 0],
    'train_j': train_triple[:, 1],
    'train_m': train_triple[:, 2]
}

pickle.dump(p, open(f'triple_{para_index}.para', 'wb'))

print(f'triple_{para_index}.para saved')
print('data_triple finished.')