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
# 2. 모델 로드
# ─────────────────────────────────────────────────────────────
import sys
sys.path.append('.')

# model_train.py 전체 재실행 방지: 필요한 것만 가져오기
import importlib.util, types

spec = importlib.util.spec_from_file_location("model_train", "./model_train.py")
mod  = types.ModuleType("model_train")

# __name__ 을 __main__ 이 아니게 해서 if __name__=="__main__" 블록 스킵
mod.__name__ = "model_train"
spec.loader.exec_module(mod)

Transformer     = mod.Transformer
src_emb_weight  = mod.src_emb_weight
user_emb_weight = mod.user_emb_weight
S               = mod.S
E               = mod.E
P               = mod.P
tgt_vocab_size  = mod.tgt_vocab_size
tgt_len         = mod.tgt_len
d_model         = mod.d_model

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
    N = len(candidate_codes)
    dec_inputs = np.concatenate(
        [np.full((N, 1), S, dtype=np.int32), candidate_codes[:, :-1]], axis=1
    )
    dec_targets = candidate_codes

    dec_inputs_t  = torch.LongTensor(dec_inputs).to(device)
    dec_targets_t = torch.LongTensor(dec_targets).to(device)

    enc_out_exp   = enc_out.expand(N, -1, -1)
    enc_input_exp = enc_input.expand(N, -1)

    with torch.no_grad():
        dec_out, _, _ = model.decoder(dec_inputs_t, enc_input_exp, enc_out_exp)
        logits = model.projection(dec_out)

        log_probs = torch.log_softmax(logits, dim=-1)
        scores = torch.zeros(N, device=device)
        for t in range(tgt_len):
            target_ids = dec_targets_t[:, t]
            scores += log_probs[:, t, :].gather(1, target_ids.unsqueeze(1)).squeeze(1)

    return scores.cpu().numpy()


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
    if mode == 'test':
        target_arr = test_arr
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

        pos_item_idx = int(np.where(target_row == 1)[0][0])
        pos_code     = item_matrix[pos_item_idx]

        seen_mask  = (train_arr[user_id] + target_row) > 0
        unseen_ids = item_ids[~seen_mask]
        neg_items  = np.random.choice(
            unseen_ids, size=min(100, len(unseen_ids)), replace=False
        )

        candidate_items = np.concatenate([[pos_item_idx], neg_items])
        candidate_codes = item_matrix[candidate_items]

        if use_val:
            val_row      = val_arr[user_id]
            val_item_idx = int(np.where(val_row == 1)[0][0]) if val_row.sum() > 0 else 0
            seq = src_mtx[user_id].copy()
            seq = np.roll(seq, -1)
            seq[-1] = val_item_idx + 1
        else:
            seq = src_mtx[user_id].copy()

        enc_input = torch.LongTensor(seq).unsqueeze(0).to(device)
        u_input   = torch.LongTensor([[user_id]]).to(device)

        with torch.no_grad():
            enc_out, _ = model.encoder(enc_input, u_input)

        scores = compute_likelihood(model, enc_out, enc_input, candidate_codes)
        rank = int((scores > scores[0]).sum()) + 1

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
