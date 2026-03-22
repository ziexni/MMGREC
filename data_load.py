import numpy as np
import pandas as pd
import scipy.sparse as sparse
import pickle
import random

# 데이터 로드
interaction = pd.read_parquet('interaction.parquet') 
item_df = pd.read_parquet('item_used.parquet')

unique_users = sorted(interaction['user_id'].unique())
unique_items = sorted(interaction['item_id'].unique())

user2idx = {u: i for i, u in enumerate(unique_users)}
item2idx = {it: i for i, it in enumerate(unique_items)}

interaction['user_idx'] = interaction['user_id'].map(user2idx)
interaction['item_idx'] = interaction['item_id'].map(item2idx)

USER_NUM = len(unique_users)
ITEM_NUM = len(unique_items)
print(f'USER_NUM = {USER_NUM}, ITEM_NUM = {ITEM_NUM}')

interaction = interaction.sort_values(['user_id', 'timestamp'])

train_ui = [] # (user, item) pair list
val_ui = []
test_ui = []

for uid, grp in interaction.groupby('user_id'):
    items = grp['item_id'].tolist() # 해당 유저의 item 시퀀스

    # 최소 3개 이상이어야 train/val/test 분할 가능
    if len(items) < 3:
        continue
    
    # train: 마지막 2개 제외
    train_ui.extend([[uid, it] for it in items[:-2]])
    
    # validation: 마지막에서 두 번째
    val_ui.append([uid, items[-2]])

    # test: 마지막
    test_ui.append([uid, items[-1]])

# numpy array로 변환 (모델 입력용)
train_ui = np.array(train_ui, dtype=np.int32)
val_ui = np.array(val_ui, dtype=np.int32)
test_ui = np.array(test_ui, dtype=np.int32)

# sparse matrix 생성: user-item interaction을 sparse 형태로 표현 (메모리 절약)
def build_sparse(ui, n_user, n_item):
    row = ui[:, 0] # user index
    col = ui[:, 1] # item index
    data = np.ones(len(row), dtype=np.int8) # interaction 존재 여부 (1)

    return sparse.coo_matrix((data, (row, col)), shape=(n_user, n_item))

train_matrix = build_sparse(train_ui, USER_NUM, ITEM_NUM)
val_matrix = build_sparse(val_ui, USER_NUM, ITEM_NUM)
test_matrix = build_sparse(test_ui, USER_NUM, ITEM_NUM)

# item feature 구성 (GCN 입력용)
item_df = item_df.copy()

# item_id -> item_idx 매핑
item_df['item_idx'] = item_df['item_id'].map(item2idx)

# item_idx 기준 정렬
item_df = item_df.sort_values('item_idx').reset_index(drop=True)

# (1) video feature
video_features = np.vstack(item_df['video_feature'].values).astype(np.float32)
np.save('v_feat.npy', video_features)

# (2) title feature
title_feat = np.load('title.npy').astype(np.float32)
np.save('t_feat.npy', title_feat)

# (3) category feature (multi-hot encoding)
all_cats = set()

# 전체 카테고리 집합 수집
for cats in item_df['category_id']:
    if isinstance(cats, (list, np.ndarray)):
        all_cats.update(cats)
    else:
        all_cats.add(cats)

CAT_NUM = max(all_cats) + 1 

# multi-hot matrix 생성 (item_num x cat_num)
cat_feat = np.zeros((ITEM_NUM, CAT_NUM), dtype=np.float32)

for _, row in item_df.iterrows():
    idx = int(row['item_id'])
    cats = row['category_id']

    # 단일 값이면 리스트로 변환
    if not isinstance(cats, (list, np.ndarray)):
        cats = [cats]
    # 해당 카테고리 위치에 1 할당
    for c in cats:
        cat_feat[idx, int(c)] = 1.0

np.save('c_feat.npy', cat_feat)

# 최종 저장
para = {
    'user_num' : USER_NUM,
    'item_num' : ITEM_NUM,
    'train_matrix' : train_matrix,
    'val_matrix': val_matrix,
    'test_matrix' : test_matrix,
    'train_ui' : train_ui,
    'user2idx' : user2idx,
    'item2idx' : item2idx,
}

# pickle로 직렬화 저장 (모델 학습 시 로드)
pickle.dump(para, open('load.para', 'wb'))

print('data_load finished')