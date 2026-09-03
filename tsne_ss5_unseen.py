import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import csv

sys.path.insert(0, '.')
from model import Encoder
from data_cnn60 import NTUDataLoaders

device = torch.device('cuda')
ntu, ss = 60, 5
latent_size = 100
vis_emb_input_size = 256
le, tm = 'qwen512', 'lb_dirv2_mdv3'
wdir = 'results/5_r'
epoch = 15299
dataset_path = 'sk_feats/shift_5_r/'

unseen_inds = np.sort(np.load('label_splits/ru5.npy'))
labels_map = {}
with open(f'sem_info/lb_{ntu}.csv') as f:
    r = csv.reader(f); next(r)
    for idx, lb in r: labels_map[int(idx) - 1] = lb

sequence_encoder = Encoder([vis_emb_input_size, latent_size]).to(device)
sequence_encoder.load_state_dict(torch.load(f'{wdir}/{le}/{tm}/se_{epoch}.pth.tar', map_location=device)['state_dict'])
sequence_encoder.eval()

tml = tm.split('_')
tfl = [torch.from_numpy(np.load(f'./text_feats/{le}/{m}_{ntu}.npy')) for m in tml]
text_feat = torch.cat(tfl, dim=-1)
text_emb = text_feat / torch.norm(text_feat, dim=1, keepdim=True).repeat([1, text_feat.size(-1)])
text_encoder = Encoder([text_feat.size(-1), latent_size]).to(device)
text_encoder.load_state_dict(torch.load(f'{wdir}/{le}/{tm}/te_{epoch}.pth.tar', map_location=device)['state_dict'])
text_encoder.eval()

with torch.no_grad():
    text_mu_all, _ = text_encoder(text_emb.to(device).float())
text_mu_all = text_mu_all.cpu().numpy()
text_mu_all = text_mu_all / np.linalg.norm(text_mu_all, axis=1, keepdims=True)
text_mu_unseen = text_mu_all[unseen_inds]

ntu_loaders = NTUDataLoaders(dataset_path, 'max', 1)
val_loader = ntu_loaders.get_test_loader(64, 0)

sk_mu_list, sk_labels_list = [], []
with torch.no_grad():
    for inp, target in val_loader:
        mask = np.isin(target.numpy(), unseen_inds)
        if mask.sum() == 0: continue
        mu, _ = sequence_encoder(inp[mask].to(device))
        sk_mu_list.append(mu.cpu().numpy())
        sk_labels_list.append(target.numpy()[mask])
sk_mu = np.concatenate(sk_mu_list, axis=0)
sk_labels = np.concatenate(sk_labels_list, axis=0)
sk_mu = sk_mu / np.linalg.norm(sk_mu, axis=1, keepdims=True)

# sous-echantillonne a 120/classe pour rester lisible
MAX_PER_CLASS = 120
rng = np.random.RandomState(0)
keep = []
for c in unseen_inds:
    idx = np.where(sk_labels == c)[0]
    if len(idx) > MAX_PER_CLASS:
        idx = rng.choice(idx, MAX_PER_CLASS, replace=False)
    keep.append(idx)
keep = np.concatenate(keep)
sk_mu_sub, sk_labels_sub = sk_mu[keep], sk_labels[keep]

combined = np.concatenate([sk_mu_sub, text_mu_unseen], axis=0)
n_sk = sk_mu_sub.shape[0]
tsne = TSNE(n_components=2, perplexity=30, random_state=0, init='pca')
proj = tsne.fit_transform(combined)
sk_proj, text_proj = proj[:n_sk], proj[n_sk:]

fig, ax = plt.subplots(1, 1, figsize=(14, 11))
colors = {10: 'tab:blue', 11: 'tab:orange', 19: 'tab:green', 26: 'tab:red', 56: 'tab:purple'}

for c in unseen_inds:
    mask = sk_labels_sub == c
    ax.scatter(sk_proj[mask, 0], sk_proj[mask, 1], color=colors[c], s=20, alpha=0.5,
               label=f'{labels_map[c]} (squelette)', edgecolors='none')

for i, c in enumerate(unseen_inds):
    ax.scatter(*text_proj[i], color=colors[c], marker='*', s=700,
               edgecolors='black', linewidths=2, zorder=5)
    ax.annotate(labels_map[c], text_proj[i], fontsize=12, fontweight='bold',
                xytext=(8, 8), textcoords='offset points', zorder=6)

ax.set_title(f'NTU-60 ss=5 — les 5 classes unseen (esterel29)\n'
             f'reading (20.6% acc, absorbee dans writing 79.4%) vs writing (85.9% acc) — les autres quasi parfaites',
             fontsize=13)
ax.set_xticks([]); ax.set_yticks([])
ax.legend(loc='upper right', fontsize=10, markerscale=2)
plt.tight_layout()
plt.savefig('tsne_ss5_5classes.png', dpi=150)
print("Sauvegarde : tsne_ss5_5classes.png")
