"""
train_gcn_rqvae.py
==================
1) GCN (GraphSAGE): user-item bipartite graph에서 representation 학습 (BPR loss)
2) RQ-VAE: item embedding → 3단계 residual quantization → semantic ID

멀티모달 feature 구성:
  - video   : v_feat.npy  (item_num, video_dim)
  - title   : t_feat.npy  (item_num, title_dim)
  - category: c_feat.npy  (item_num, cat_num) multi-hot
              → GCN 내부 EmbeddingBag(mean)으로 cat_embed_dim 차원 변환 후 concat

출력:
  - representation.npy : GCN embedding (user_num + item_num, gcn_out)
  - tgt_mtx.npy        : semantic ID + collision rank (item_num, 4)
"""

import pickle, time, glob
import torch
import numpy as np
import torch.utils.data as D
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

# ─────────────────────────────────────────────────────────────
# 1. 학습 설정
# ─────────────────────────────────────────────────────────────
batch_size     = 3000
step_threshold = 500
epoch_max      = 60
gcn_hidden     = 128
gcn_out        = 64
cat_embed_dim  = 32

rqvae_hidden   = 16
rqvae_out      = 4
n_embed        = 128
rqvae_epochs   = 60
rqvae_lr       = 1e-3
l_w_embedding  = 1.0
l_w_commitment = 0.25

# ─────────────────────────────────────────────────────────────
# 2. 데이터 로드
# ─────────────────────────────────────────────────────────────
para_load = pickle.load(open('load.para', 'rb'))
user_num  = para_load['user_num']
item_num  = para_load['item_num']
train_ui  = para_load['train_ui']
print(f'users={user_num}, items={item_num}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

v_feat = np.load('v_feat.npy')   # (item_num, video_dim)
t_feat = np.load('t_feat.npy')   # (item_num, title_dim)
c_feat = np.load('c_feat.npy')   # (item_num, cat_num)
cat_num = c_feat.shape[1]

# ─────────────────────────────────────────────────────────────
# 3. category multi-hot → EmbeddingBag 입력 형식으로 변환
# ─────────────────────────────────────────────────────────────
cat_indices_list   = [np.where(row > 0)[0].tolist() for row in c_feat]
offsets            = np.cumsum([0] + [len(x) for x in cat_indices_list[:-1]], dtype=np.int64)
cat_indices_tensor = torch.tensor(
    np.concatenate(cat_indices_list).astype(np.int64), dtype=torch.long).to(device)
cat_offsets_tensor = torch.tensor(offsets, dtype=torch.long).to(device)

# ─────────────────────────────────────────────────────────────
# 4. video + title → GPU tensor
# ─────────────────────────────────────────────────────────────
vt_feat   = np.concatenate([v_feat, t_feat], axis=1).astype(np.float32)
vt_lookup = torch.from_numpy(vt_feat).float().to(device)

# ─────────────────────────────────────────────────────────────
# 5. BPR triple 로드
# ─────────────────────────────────────────────────────────────
triple_files = sorted(glob.glob('triple_*.para'))
train_i = torch.empty(0).long()
train_j = torch.empty(0).long()
train_m = torch.empty(0).long()

for fp in triple_files:
    tp      = pickle.load(open(fp, 'rb'))
    train_i = torch.cat([train_i, torch.tensor(tp['train_i']).long()])
    train_j = torch.cat([train_j, torch.tensor(tp['train_j']).long()])
    train_m = torch.cat([train_m, torch.tensor(tp['train_m']).long()])

train_loader = D.DataLoader(
    D.TensorDataset(train_i, train_j, train_m),
    batch_size=batch_size, shuffle=True)

# ─────────────────────────────────────────────────────────────
# 6. Graph 구성 (user-item bipartite, undirected)
# ─────────────────────────────────────────────────────────────
_ui        = train_ui + [0, user_num]
edge       = np.concatenate([_ui, _ui[:, [1, 0]]], axis=0)
edge_index = torch.tensor(edge, dtype=torch.long).t().contiguous().to(device)

# ─────────────────────────────────────────────────────────────
# 7. GCN 모델
# ─────────────────────────────────────────────────────────────
class GCN(nn.Module):
    def __init__(self, vt_dim, cat_num, cat_embed_dim, user_num, item_num,
                 hidden=128, out=64):
        super().__init__()
        self.user_num     = user_num
        self.cat_embedder = nn.EmbeddingBag(cat_num, cat_embed_dim, mode='mean')
        self.fuse         = nn.Linear(vt_dim + cat_embed_dim, hidden, bias=False)
        self.conv1        = SAGEConv(hidden, hidden)
        self.conv2        = SAGEConv(hidden, out)
        self.user         = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(user_num, hidden)))

    def forward(self, vt_feature, cat_indices, cat_offsets, edge_index):
        cat_emb   = self.cat_embedder(cat_indices, cat_offsets)
        item_feat = self.fuse(torch.cat([vt_feature, cat_emb], dim=1))
        x0 = torch.cat([self.user, item_feat], dim=0)
        x1 = F.leaky_relu(self.conv1(x0, edge_index))
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x2 = F.dropout(self.conv2(x1, edge_index), p=0.2, training=self.training)
        return x2

gcn = GCN(vt_feat.shape[1], cat_num, cat_embed_dim,
          user_num, item_num, hidden=gcn_hidden, out=gcn_out).to(device)
optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=1e-3, weight_decay=1e-6)

# ─────────────────────────────────────────────────────────────
# 8. GCN 학습 (BPR loss)
# ─────────────────────────────────────────────────────────────
gcn.train()
for epoch in range(epoch_max):
    running_loss = 0.0
    for step, (bi, bj, bm) in enumerate(train_loader):
        out = gcn(vt_lookup, cat_indices_tensor, cat_offsets_tensor, edge_index)

        ei  = out[bi.numpy()]
        ej  = out[bj.numpy() + user_num]
        em  = out[bm.numpy() + user_num]

        pij  = torch.sum(ei * ej, dim=1)
        pim  = torch.sum(ei * em, dim=1)
        loss = -torch.mean(torch.log(torch.sigmoid(pij - pim) + 1e-10))

        optimizer_gcn.zero_grad()
        loss.backward()
        optimizer_gcn.step()

        running_loss += loss.item()
        if (step + 1) % step_threshold == 0:
            print(f'[epoch {epoch+1}, step {step+1}] loss: {running_loss/step_threshold:.5f}')
            running_loss = 0.0

gcn.eval()
with torch.no_grad():
    out = gcn(vt_lookup, cat_indices_tensor, cat_offsets_tensor, edge_index).cpu().numpy()
np.save('representation.npy', out)
print('representation.npy saved.')

# ─────────────────────────────────────────────────────────────
# 9. RQ-VAE
# ─────────────────────────────────────────────────────────────
class MyDataSet(D.Dataset):
    def __init__(self, x): self.x = x
    def __len__(self):     return len(self.x)
    def __getitem__(self, i): return self.x[i]

class RQVAE(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, k1, k2, k3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, hidden), nn.Sigmoid(), nn.Linear(hidden, in_dim))
        for i, k in enumerate([k1, k2, k3], 1):
            emb = nn.Embedding(k, out_dim)
            emb.weight.data.uniform_(-1/k, 1/k)
            setattr(self, f'vq_{i}', emb)

    def _quantize(self, ze, emb):
        N, C = ze.shape
        K    = emb.shape[0]
        d    = torch.sum((emb.reshape(1, K, C) - ze.reshape(N, 1, C)) ** 2, 2)
        idx  = torch.argmin(d, 1)
        return emb[idx], idx

    def forward(self, x):
        ze1       = self.encoder(x)
        zq1, n1   = self._quantize(ze1, self.vq_1.weight.data)
        ze2       = ze1 - zq1
        zq2, n2   = self._quantize(ze2, self.vq_2.weight.data)
        ze3       = ze2 - zq2
        zq3, n3   = self._quantize(ze3, self.vq_3.weight.data)
        dec_in    = ze1 + ((zq1 + zq2 + zq3) - ze1).detach()
        return self.decoder(dec_in), ze1, ze2, ze3, zq1, zq2, zq3, n1, n2, n3

content = np.load('representation.npy')[user_num: user_num + item_num, :]
loader  = D.DataLoader(MyDataSet(torch.Tensor(content)), batch_size=1024, shuffle=True)
model   = RQVAE(gcn_out, rqvae_hidden, rqvae_out, n_embed, n_embed, n_embed).cuda()
opt     = torch.optim.Adam(model.parameters(), rqvae_lr, weight_decay=1e-4)
mse     = nn.MSELoss()

model.train()
tic = time.time()
for e in range(rqvae_epochs):
    total = 0
    for x in loader:
        x                                        = x.cuda()
        xh, ze1, ze2, ze3, zq1, zq2, zq3, *_   = model(x)
        l_r  = mse(x, xh)
        l_e  = mse(ze1.detach(), zq1) + mse(ze2.detach(), zq2) + mse(ze3.detach(), zq3)
        l_c  = mse(ze1, zq1.detach()) + mse(ze2, zq2.detach()) + mse(ze3, zq3.detach())
        loss = l_r + l_w_embedding * l_e + l_w_commitment * l_c
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * x.size(0)
    total /= len(loader.dataset)
    print(f'RQ-VAE epoch {e} loss: {total:.5f} elapsed {time.time()-tic:.1f}s')

# ─────────────────────────────────────────────────────────────
# 10. Semantic ID 추출 + collision 해결
# ─────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    id1 = id2 = id3 = np.empty(0)
    for batch in np.array_split(np.arange(item_num), 4):
        x        = torch.Tensor(content[batch]).cuda()
        *_, n1, n2, n3 = model(x)
        id1 = np.append(id1, n1.cpu().numpy())
        id2 = np.append(id2, n2.cpu().numpy())
        id3 = np.append(id3, n3.cpu().numpy())

r_id = (np.stack([id1, id2, id3], axis=1) + 1).astype(np.int16)

result, inv = np.unique(r_id, axis=0, return_inverse=True)
col         = np.zeros(item_num, dtype=np.int32)

train_matrix = para_load['train_matrix']
train_matrix.data = np.ones_like(train_matrix.data, dtype=np.int8)
popularity   = np.asarray(train_matrix.sum(axis=0)).flatten()

for i in range(result.shape[0]):
    loc = np.where(inv == i)[0]
    loc = loc[np.argsort(-popularity[loc])]
    for j, pos in enumerate(loc):
        col[pos] = j + 1

tgt_mtx = np.concatenate([r_id, col.reshape(-1, 1)], axis=1)
np.save('tgt_mtx.npy', tgt_mtx)
print('tgt_mtx.npy saved. All done!')