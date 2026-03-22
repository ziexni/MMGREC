"""
evaluate.py
===========
평가 지표: HR@10, NDCG@10, MRR@10

SASRec 방식:
  - 각 유저의 test 아이템 1개 + random negative 99개 = 100개 후보
  - beam 결과 중 100개 후보 안에 속하는 것만 유효 hit로 인정
  - beam 생성 순서 = 순위 (생성형 추천 모델 관행)
"""

import math
import numpy as np
import pickle

# ─────────────────────────────────────────────────────────────
# 1. 로드
# ─────────────────────────────────────────────────────────────
result_dict = pickle.load(open('result.pkl', 'rb'))

para      = pickle.load(open('load.para', 'rb'))
user_num  = para['user_num']
item_num  = para['item_num']

train_matrix = para['train_matrix']
train_matrix.data = np.ones_like(train_matrix.data, dtype=np.int8)
train_arr = train_matrix.toarray()

test_matrix = para['test_matrix']
test_matrix.data = np.ones_like(test_matrix.data, dtype=np.int8)
test_arr = test_matrix.toarray()

item_matrix = np.load('tgt_mtx.npy')   # (item_num, 4): [c1, c2, c3, col]
K = 10

# ─────────────────────────────────────────────────────────────
# 2. 지표 함수
# ─────────────────────────────────────────────────────────────
def hit_at_k(rank, k):
    return 1 if 0 < rank <= k else 0

def ndcg_at_k(rank, k):
    # IDCG = 1 (test 아이템 1개이므로 이상적 순위 = 1위)
    if rank == 0 or rank > k:
        return 0.0
    return (1.0 / math.log2(rank + 1)) / (1.0 / math.log2(2))

def mrr_at_k(rank, k):
    if rank == 0 or rank > k:
        return 0.0
    return 1.0 / rank

# ─────────────────────────────────────────────────────────────
# 3. 평가
# ─────────────────────────────────────────────────────────────
HR = NDCG = MRR = 0.0
valid_users = 0
item_ids    = np.arange(item_num)

for user_id in range(user_num):
    test_row = test_arr[user_id]
    if test_row.sum() == 0:
        continue

    # ── positive test 아이템 코드
    pos_item_idx = int(np.where(test_row == 1)[0][0])
    pos_code     = item_matrix[pos_item_idx]   # [c1, c2, c3, col]

    # ── random negative 99개 (train + test에 없는 아이템)
    seen_mask  = (train_arr[user_id] + test_row) > 0
    unseen_ids = item_ids[~seen_mask]
    neg_items  = np.random.choice(unseen_ids, size=min(99, len(unseen_ids)), replace=False)
    neg_codes  = item_matrix[neg_items]   # (99, 4)

    # ── 100개 후보 집합 (tuple set으로 O(1) 조회)
    candidate_codes = np.vstack([pos_code.reshape(1, -1), neg_codes])   # (100, 4)
    candidate_set   = set(map(tuple, candidate_codes.tolist()))

    # ── beam 결과에서 rank 계산
    beams = result_dict[user_id]   # (num_beams, 4)

    rank = 0
    valid_rank = 0   # 후보 100개 안에 있는 beam 순위 카운터
    for beam in beams:
        beam_tuple = tuple(beam.tolist())
        if beam_tuple not in candidate_set:
            # ✅ 후보 밖 beam은 순위 계산에서 제외 (SASRec 방식)
            continue
        valid_rank += 1
        if (beam == pos_code).all():
            rank = valid_rank
            break

    HR   += hit_at_k(rank, K)
    NDCG += ndcg_at_k(rank, K)
    MRR  += mrr_at_k(rank, K)
    valid_users += 1

HR   /= valid_users
NDCG /= valid_users
MRR  /= valid_users

print(f'Evaluated on {valid_users} users  (1 positive + 99 negatives)')
print(f'HR@{K}:   {HR:.4f}')
print(f'NDCG@{K}: {NDCG:.4f}')
print(f'MRR@{K}:  {MRR:.4f}')