"""
evaluate.py (수정버전)
=======================
평가 지표: HR@10, NDCG@10, MRR@10

수정 사항:
  - beam search 방식 → 101개 후보 likelihood scoring 방식
  - negative 99개 → 100개 (우리 베이스라인 조건 통일)
  - 각 후보 아이템 코드를 Transformer decoder에 넣어 likelihood 계산 후 ranking
"""

import math
import numpy as np
import pickle
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────
# 1. 로드
# ─────────────────────────────────────────────────────────────
para      = pickle.load(open('load.para', 'rb'))
user_num  = para['user_num']
item_num  = para['item_num']

train_matrix = para['train_matrix']
train_matrix.data = np.ones_like(train_matrix.data, dtype=np.int8)
train_arr = train_matrix.toarray()

test_matrix = para['test_matrix']
test_matrix.data = np.ones_like(test_matrix.data, dtype=np.int8)
test_arr = test_matrix.toarray()

val_matrix = para['val_matrix']
val_matrix.data = np.ones_like(val_matrix.data, dtype=np.int8)
val_arr = val_matrix.toarray()

item_matrix = np.load('tgt_mtx.npy')   # (item_num, 4): [c1, c2, c3, col]
src_mtx     = np.load('src_mtx.npy')   # (user_num, src_len)

K = 10

# ─────────────────────────────────────────────────────────────
# 2. 모델 로드 (train_transformer.py에서 저장한 모델)
# ─────────────────────────────────────────────────────────────
# train_transformer.py의 하이퍼파라미터와 동일하게 맞춰야 함
import sys
sys.path.append('.')
from train_transformer import Transformer, src_emb_weight, user_emb_weight, \
    S, E, P, tgt_vocab_size, tgt_len, d_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Transformer(src_emb_weight, user_emb_weight).to(device)
model.load_state_dict(
    {k: v.clone() for k, v in pickle.load(open('best_model_state.pkl', 'rb')).items()}
)
model.eval()


# ─────────────────────────────────────────────────────────────
# 3. Likelihood 계산 함수
# ─────────────────────────────────────────────────────────────
def compute_likelihood(model, enc_out, enc_input, candidate_codes):
    """
    candidate_codes: (N, tgt_len) numpy array - 후보 아이템 코드들
    반환: (N,) likelihood scores
    """
    N = len(candidate_codes)

    # decoder 입력: [SOS, c1, c2, c3] (마지막 col 제외)
    dec_inputs = np.concatenate(
        [np.full((N, 1), S, dtype=np.int32), candidate_codes[:, :-1]], axis=1
    )  # (N, tgt_len)

    # decoder 출력 정답: [c1, c2, c3, col]
    dec_targets = candidate_codes  # (N, tgt_len)

    dec_inputs_t  = torch.LongTensor(dec_inputs).to(device)   # (N, tgt_len)
    dec_targets_t = torch.LongTensor(dec_targets).to(device)  # (N, tgt_len)

    # enc_out을 N개로 복제
    enc_out_exp   = enc_out.expand(N, -1, -1)          # (N, src_len, d_model)
    enc_input_exp = enc_input.expand(N, -1)            # (N, src_len)

    with torch.no_grad():
        dec_out, _, _ = model.decoder(dec_inputs_t, enc_input_exp, enc_out_exp)
        logits = model.projection(dec_out)  # (N, tgt_len, vocab_size)

        # 각 위치의 log probability 합산 → likelihood
        log_probs = torch.log_softmax(logits, dim=-1)  # (N, tgt_len, vocab_size)
        scores = torch.zeros(N, device=device)
        for t in range(tgt_len):
            target_ids = dec_targets_t[:, t]  # (N,)
            scores += log_probs[:, t, :].gather(1, target_ids.unsqueeze(1)).squeeze(1)

    return scores.cpu().numpy()  # (N,)


# ─────────────────────────────────────────────────────────────
# 4. 지표 함수
# ─────────────────────────────────────────────────────────────
def hit_at_k(rank, k):
    return 1 if 0 < rank <= k else 0

def ndcg_at_k(rank, k):
    if rank == 0 or rank > k:
        return 0.0
    return (1.0 / math.log2(rank + 1)) / (1.0 / math.log2(2))

def mrr_at_k(rank, k):
    if rank == 0 or rank > k:
        return 0.0
    return 1.0 / rank


# ─────────────────────────────────────────────────────────────
# 5. 평가 함수 (101개 후보 방식)
# ─────────────────────────────────────────────────────────────
def evaluate(mode='test'):
    """
    mode: 'test' or 'valid'
    """
    if mode == 'test':
        target_arr = test_arr
        # test 시: val 아이템을 시퀀스에 포함
        use_val = True
    else:
        target_arr = val_arr
        use_val = False

    HR = NDCG = MRR = 0.0
    valid_users = 0
    item_ids = np.arange(item_num)

    for user_id in range(user_num):
        target_row = target_arr[user_id]
        if target_row.sum() == 0:
            continue

        # 정답 아이템
        pos_item_idx = int(np.where(target_row == 1)[0][0])
        pos_code     = item_matrix[pos_item_idx]  # [c1, c2, c3, col]

        # negative 100개 (train + target에 없는 아이템)
        seen_mask  = (train_arr[user_id] + target_row) > 0
        unseen_ids = item_ids[~seen_mask]
        neg_items  = np.random.choice(
            unseen_ids, size=min(100, len(unseen_ids)), replace=False
        )

        # 101개 후보 코드 (정답 1 + negative 100)
        candidate_items = np.concatenate([[pos_item_idx], neg_items])
        candidate_codes = item_matrix[candidate_items]  # (101, 4)

        # encoder 입력 구성
        if use_val:
            # test: val 아이템을 시퀀스 끝에 추가
            val_row      = val_arr[user_id]
            val_item_idx = int(np.where(val_row == 1)[0][0]) if val_row.sum() > 0 else 0
            seq = src_mtx[user_id].copy()
            seq = np.roll(seq, -1)
            seq[-1] = val_item_idx + 1  # 1-indexed
        else:
            seq = src_mtx[user_id].copy()

        enc_input = torch.LongTensor(seq).unsqueeze(0).to(device)     # (1, src_len)
        u_input   = torch.LongTensor([[user_id]]).to(device)           # (1, 1)

        with torch.no_grad():
            enc_out, _ = model.encoder(enc_input, u_input)  # (1, src_len, d_model)

        # 101개 후보 likelihood 계산
        scores = compute_likelihood(model, enc_out, enc_input, candidate_codes)

        # 정답(index 0)의 rank 계산 (높은 score = 좋은 rank)
        rank = int((scores > scores[0]).sum()) + 1  # 1-based rank

        HR   += hit_at_k(rank, K)
        NDCG += ndcg_at_k(rank, K)
        MRR  += mrr_at_k(rank, K)
        valid_users += 1

        if valid_users % 500 == 0:
            print(f'{valid_users} / {user_num} done')

    HR   /= valid_users
    NDCG /= valid_users
    MRR  /= valid_users

    print(f'Evaluated on {valid_users} users  (1 positive + 100 negatives)')
    print(f'HR@{K}:   {HR:.4f}')
    print(f'NDCG@{K}: {NDCG:.4f}')
    print(f'MRR@{K}:  {MRR:.4f}')
    return HR, NDCG, MRR


# ─────────────────────────────────────────────────────────────
# 6. 실행
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=== Valid ===')
    evaluate(mode='valid')
    print('\n=== Test ===')
    evaluate(mode='test')
