"""
train_transformer.py
====================
User-aware Transformer (Encoder-Decoder) 학습 및 beam search 추론.

수정 사항:
  - EncoderLayer d_model/2 → d_model//2 (정수 나눗셈 버그 수정)
  - LayerNorm / PoswiseFeedForwardNet 모듈 수준으로 등록 (매 forward 생성 버그 수정)
  - epoch_max 파라미터화
  - result.pkl 구조를 evaluate.py 형식에 맞게 통일
    { user_id: beams(np.array, shape=(num_beams, tgt_len)) }
"""

import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

# ─────────────────────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────────────────────
src_mtx = np.load('src_mtx.npy')   # (user_num, src_len)
tgt_mtx = np.load('tgt_mtx.npy')   # (item_num, 4): [c1, c2, c3, col]

para     = pickle.load(open('load.para', 'rb'))
train_ui = para['train_ui']
user_num = para['user_num']
item_num = para['item_num']

rep         = np.load('representation.npy')
user_matrix = rep[:user_num, :]
item_matrix = rep[user_num: user_num + item_num, :]

# pad(0) 행 추가 → index 1부터 실제 아이템
src_emb_weight  = torch.FloatTensor(
    np.concatenate([np.zeros((1, item_matrix.shape[1])), item_matrix], axis=0))
user_emb_weight = torch.FloatTensor(user_matrix)

# ─────────────────────────────────────────────────────────────
# 2. 특수 토큰
# ─────────────────────────────────────────────────────────────
P = 0
S = int(np.max(tgt_mtx)) + 1   # <SOS>
E = S + 1                       # <EOS>
tgt_vocab_size = E + 1

# ─────────────────────────────────────────────────────────────
# 3. 하이퍼파라미터
# ─────────────────────────────────────────────────────────────
d_model   = 64
d_ff      = 64
d_k = d_v = 64
n_layers  = 2
n_heads   = 4
epoch_max = 100
patience  = 10      # early stopping patience
batch_size = 3000
num_beams = 10
tgt_len   = 4   # RQ-VAE code 길이 (c1, c2, c3, col)

# ─────────────────────────────────────────────────────────────
# 4. Dataset
# ─────────────────────────────────────────────────────────────
def make_data():
    enc_inputs, dec_inputs, dec_outputs, u_inputs = [], [], [], []
    for pair in train_ui:
        enc_inputs.append(src_mtx[pair[0]])
        dec_inputs.append(np.insert(tgt_mtx[pair[1]], 0, S))   # <SOS> + code
        dec_outputs.append(np.append(tgt_mtx[pair[1]], E))      # code + <EOS>
        u_inputs.append([pair[0]])
    return (torch.LongTensor(enc_inputs),
            torch.LongTensor(dec_inputs),
            torch.LongTensor(dec_outputs),
            torch.LongTensor(u_inputs))

enc_inputs, dec_inputs, dec_outputs, u_inputs = make_data()

class MyDataSet(Data.Dataset):
    def __init__(self, ei, di, do_, ui):
        self.ei, self.di, self.do_, self.ui = ei, di, do_, ui
    def __len__(self): return self.ei.shape[0]
    def __getitem__(self, idx):
        return self.ei[idx], self.di[idx], self.do_[idx], self.ui[idx]

loader = Data.DataLoader(
    MyDataSet(enc_inputs, dec_inputs, dec_outputs, u_inputs),
    batch_size=batch_size, shuffle=True)

# ─────────────────────────────────────────────────────────────
# 5. 모델 컴포넌트
# ─────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0)])


def get_attn_pad_mask(seq_q, seq_k):
    B, Lq = seq_q.size()
    _, Lk = seq_k.size()
    return seq_k.eq(0).unsqueeze(1).expand(B, Lq, Lk)


def get_attn_subsequence_mask(seq):
    shape = [seq.size(0), seq.size(1), seq.size(1)]
    return torch.from_numpy(np.triu(np.ones(shape), k=1)).byte()


class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        return torch.matmul(torch.softmax(scores, dim=-1), V), torch.softmax(scores, dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q    = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K    = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V    = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc     = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.norm   = nn.LayerNorm(d_model)   # ✅ 모듈로 등록
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        B = input_Q.size(0)
        Q = self.W_Q(input_Q).view(B, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(B, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(B, -1, n_heads, d_v).transpose(1, 2)
        mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        ctx, attn = ScaledDotProductAttention()(Q, K, V, mask)
        ctx = ctx.transpose(1, 2).reshape(B, -1, n_heads * d_v)
        out = self.dropout(self.fc(ctx))
        return self.norm(out + input_Q), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc   = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False), nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))
        self.norm = nn.LayerNorm(d_model)   # ✅ 모듈로 등록

    def forward(self, x):
        return self.norm(self.fc(x) + x)


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.U_Q = nn.Linear(d_model, d_k, bias=False)
        self.U_K = nn.Linear(d_model, d_k, bias=False)
        self.U_V = nn.Linear(d_model, d_v, bias=False)
        self.sca_q = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False), nn.ReLU(),  # ✅ // 정수 나눗셈
            nn.Linear(d_model // 2, 1, bias=False))
        self.sca_k = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False), nn.ReLU(),
            nn.Linear(d_model // 2, 1, bias=False))
        self.sca_v = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False), nn.ReLU(),
            nn.Linear(d_model // 2, 1, bias=False))
        self.fc      = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.norm    = nn.LayerNorm(d_model)   # ✅ 모듈로 등록
        self.pos_ffn = PoswiseFeedForwardNet()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, attn_mask, u):
        B = x.size(0)
        Q = self.W_Q(x).view(B, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(x).view(B, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(x).view(B, -1, n_heads, d_v).transpose(1, 2)

        S_q = self.sca_q(u).expand(-1, x.size(1), d_k)
        S_k = self.sca_k(u).expand(-1, x.size(1), d_k)
        S_v = self.sca_v(u).expand(-1, x.size(1), d_k)

        uQ = torch.mul(self.U_Q(x), S_q).view(B, -1, 1, d_k).transpose(1, 2).expand(-1, n_heads, -1, -1)
        uK = torch.mul(self.U_K(x), S_k).view(B, -1, 1, d_k).transpose(1, 2).expand(-1, n_heads, -1, -1)
        uV = torch.mul(self.U_V(x), S_v).view(B, -1, 1, d_v).transpose(1, 2).expand(-1, n_heads, -1, -1)

        mask   = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        scores = (torch.matmul(Q, K.transpose(-1, -2)) +
                  torch.matmul(uQ, uK.transpose(-1, -2))) / math.sqrt(d_k)
        scores.masked_fill_(mask, -1e9)
        attn    = torch.softmax(scores, dim=-1)
        ctx     = torch.matmul(attn, V + uV)
        ctx     = ctx.transpose(1, 2).reshape(B, -1, n_heads * d_v)
        out     = self.dropout(self.fc(ctx))
        enc_out = self.norm(out + x)
        return self.pos_ffn(enc_out), attn


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn  = MultiHeadAttention()
        self.pos_ffn       = PoswiseFeedForwardNet()

    def forward(self, dec_in, enc_out, self_mask, enc_mask):
        dec_out, sa = self.dec_self_attn(dec_in, dec_in, dec_in, self_mask)
        dec_out, ea = self.dec_enc_attn(dec_out, enc_out, enc_out, enc_mask)
        return self.pos_ffn(dec_out), sa, ea


class Encoder(nn.Module):
    def __init__(self, src_emb_weight, user_emb_weight):
        super().__init__()
        self.src_emb  = nn.Embedding.from_pretrained(src_emb_weight,  freeze=True)
        self.user_emb = nn.Embedding.from_pretrained(user_emb_weight, freeze=True)
        self.pos_emb  = PositionalEncoding(d_model)
        self.dropout  = nn.Dropout(p=0.2)
        self.layers   = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs, u_inputs):
        enc_out  = self.dropout(self.src_emb(enc_inputs))
        pad_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        u_out    = self.user_emb(u_inputs)   # (B, 1, d_model)
        attns    = []
        for layer in self.layers:
            enc_out, attn = layer(enc_out, pad_mask, u_out)
            attns.append(attn)
        return enc_out, attns


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers  = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_out   = self.tgt_emb(dec_inputs).cuda()
        dec_out   = self.pos_emb(dec_out.transpose(0, 1)).transpose(0, 1).cuda()
        pad_mask  = get_attn_pad_mask(dec_inputs, dec_inputs).cuda()
        sub_mask  = get_attn_subsequence_mask(dec_inputs).cuda()
        self_mask = torch.gt(pad_mask + sub_mask, 0).cuda()
        enc_mask  = get_attn_pad_mask(dec_inputs, enc_inputs)
        sa_list, ea_list = [], []
        for layer in self.layers:
            dec_out, sa, ea = layer(dec_out, enc_outputs, self_mask, enc_mask)
            sa_list.append(sa); ea_list.append(ea)
        return dec_out, sa_list, ea_list


class Transformer(nn.Module):
    def __init__(self, src_emb_weight, user_emb_weight):
        super().__init__()
        self.encoder    = Encoder(src_emb_weight, user_emb_weight).cuda()
        self.decoder    = Decoder().cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()

    def forward(self, enc_inputs, dec_inputs, u_inputs):
        enc_out, enc_attns             = self.encoder(enc_inputs, u_inputs)
        dec_out, dec_sa, dec_ea        = self.decoder(dec_inputs, enc_inputs, enc_out)
        logits = self.projection(dec_out)
        return logits.view(-1, logits.size(-1)), enc_attns, dec_sa, dec_ea

# ─────────────────────────────────────────────────────────────
# 6. 학습 (early stopping + best model 저장)
# ─────────────────────────────────────────────────────────────
model     = Transformer(src_emb_weight, user_emb_weight).cuda()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=1e-5)

best_loss       = float('inf')
patience_count  = 0
best_model_state = None

model.train()
for epoch in range(epoch_max):
    total_loss = 0.0
    for step, (ei, di, do_, ui) in enumerate(loader):
        ei, di, do_, ui = ei.cuda(), di.cuda(), do_.cuda(), ui.cuda()
        outputs, *_     = model(ei, di, ui)
        loss            = criterion(outputs, do_.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (step + 1) % 20 == 0:
            print(f'Epoch {epoch+1:04d} | Step {step+1} | loss: {loss.item():.6f}')

    avg_loss = total_loss / len(loader)
    print(f'=== Epoch {epoch+1} avg loss: {avg_loss:.6f} ===')

    # early stopping
    if avg_loss < best_loss:
        best_loss        = avg_loss
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_count   = 0
        print(f'  ✓ best model updated (loss={best_loss:.6f})')
    else:
        patience_count += 1
        print(f'  patience {patience_count}/{patience}')
        if patience_count >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

# best 모델로 복원
model.load_state_dict(best_model_state)
print(f'Transformer best loss: {best_loss:.6f}')

# ─────────────────────────────────────────────────────────────
# 7. Beam Search 추론
# ─────────────────────────────────────────────────────────────
def beam_search(model, enc_input, u_input, num_beams, max_len):
    enc_out, _ = model.encoder(enc_input, u_input)
    beams      = [[S]]
    scores     = [1.0]
    sfm        = nn.Softmax(dim=-1)

    for _ in range(max_len):
        cands, c_scores = [], []
        for beam, score in zip(beams, scores):
            dec_in          = torch.LongTensor([beam]).cuda()
            dec_out, _, _   = model.decoder(dec_in, enc_input, enc_out)
            proj            = model.projection(dec_out).squeeze(0)[-1]
            probs           = sfm(proj)
            topk_p, topk_i  = torch.topk(probs, k=num_beams)
            for p, idx in zip(topk_p, topk_i):
                t = idx.item()
                if t in (E, S, P):
                    continue
                cands.append(beam + [t])
                c_scores.append(score * p.item())

        if not cands:
            break
        beams  = np.array(cands)
        scores = np.array(c_scores)
        if len(beams) > num_beams:
            top_idx = (-scores).argsort()[:num_beams]
            beams   = beams[top_idx]
            scores  = scores[top_idx]

    return beams, scores


model.eval()
result = {}

with torch.no_grad():
    enc_inputs_all = torch.LongTensor(src_mtx).cuda()
    u_inputs_all   = torch.LongTensor(np.arange(user_num).reshape(-1, 1)).cuda()

    for i in range(user_num):
        beams, scores = beam_search(
            model,
            enc_inputs_all[i].view(1, -1),
            u_inputs_all[i].view(1, -1),
            num_beams, tgt_len)

        # ✅ SOS 토큰 제거: beams = [SOS, c1, c2, c3, col] → [c1, c2, c3, col]
        result[i] = beams[:, 1:] if beams.ndim == 2 and beams.shape[1] > 1 else beams

        if (i + 1) % 1000 == 0:
            print(f'{i+1} / {user_num} done')

# ✅ evaluate.py 호환 형식: { user_id: beams(np.array) }
with open('result.pkl', 'wb') as f:
    pickle.dump(result, f)
print('result.pkl saved. Done!')
