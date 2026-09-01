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
ntu, ss = 60, 12
latent_size = 100
vis_emb_input_size = 256
le, tm = 'qwen512', 'lb_dirv2_mdv3'
wdir = 'results/12_r'
epoch = 5099
dataset_path = 'sk_feats/shift_12_r/'

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
    text_mu, _ = text_encoder(text_emb.to(device).float())
text_mu = text_mu.cpu().numpy()
text_mu = text_mu / np.linalg.norm(text_mu, axis=1, keepdims=True)

ntu_loaders = NTUDataLoaders(dataset_path, 'max', 1)
val_loader = ntu_loaders.get_test_loader(64, 0)
unseen_inds = set(np.load('label_splits/ru12.npy').tolist())

sk_mu_list, sk_labels_list = [], []
with torch.no_grad():
    for inp, target in val_loader:
        mu, _ = sequence_encoder(inp.to(device))
        sk_mu_list.append(mu.cpu().numpy())
        sk_labels_list.append(target.numpy())
sk_mu = np.concatenate(sk_mu_list, axis=0)
sk_labels = np.concatenate(sk_labels_list, axis=0)

n_classes = 60
sk_centroids = np.array([sk_mu[sk_labels == c].mean(axis=0) for c in range(n_classes)])
sk_centroids = sk_centroids / np.linalg.norm(sk_centroids, axis=1, keepdims=True)

labels_map = {}
with open(f'sem_info/lb_{ntu}.csv') as f:
    r = csv.reader(f); next(r)
    for idx, lb in r:
        labels_map[int(idx) - 1] = lb

dist = np.linalg.norm(sk_centroids[:, None, :] - text_mu[None, :, :], axis=-1)
nearest = np.argmin(dist, axis=1)
correct = nearest == np.arange(n_classes)

combined = np.concatenate([sk_centroids, text_mu], axis=0)
tsne = TSNE(n_components=2, perplexity=15, random_state=0, init='pca')
proj = tsne.fit_transform(combined)
sk_proj, text_proj = proj[:n_classes], proj[n_classes:]

fig, ax = plt.subplots(1, 1, figsize=(16, 14))
cmap = plt.colormaps.get_cmap('tab20')

for c in range(n_classes):
    is_unseen = c in unseen_inds
    color = cmap(c % 20)
    line_color = 'green' if correct[c] else 'red'
    ax.plot([sk_proj[c, 0], text_proj[c, 0]], [sk_proj[c, 1], text_proj[c, 1]],
            color=line_color, alpha=0.4, linewidth=1, zorder=1)
    ax.scatter(*sk_proj[c], color=color, marker='^' if is_unseen else 'o', s=140,
               edgecolors='black', linewidths=1, zorder=3)
    ax.scatter(*text_proj[c], color=color, marker='*', s=350,
               edgecolors='red' if is_unseen else 'black', linewidths=1.5, zorder=3)
    mid_x, mid_y = (sk_proj[c,0]+text_proj[c,0])/2, (sk_proj[c,1]+text_proj[c,1])/2
    ax.annotate(labels_map[c], (mid_x, mid_y), fontsize=6.5,
                color=line_color, fontweight='bold' if not correct[c] else 'normal')

ax.set_title(f'NTU-60 ss=12 — Centroides squelette (rond/triangle) <-> texte (etoile), par classe\n'
             f'Vert = plus-proche-voisin correct ({correct.sum()}/60), Rouge = incorrect. '
             f'Triangle/contour rouge = classe Unseen.', fontsize=12)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig('tsne_centroids_ss12.png', dpi=150)
print("Sauvegarde : tsne_centroids_ss12.png")

print("\n=== TEST MODALITY GAP ===")
intra_sk = sk_pairwise_dist[~np.isnan(sk_pairwise_dist)].mean() if 'sk_pairwise_dist' in dir() else np.nanmean(np.linalg.norm(sk_centroids[:,None,:]-sk_centroids[None,:,:],axis=-1))
intra_text = np.nanmean(np.linalg.norm(text_mu[:,None,:]-text_mu[None,:,:],axis=-1))
cross_correct = dist[np.arange(n_classes), np.arange(n_classes)].mean()  # distance texte<->squelette POUR LA BONNE CLASSE
cross_all = dist.mean()
print(f"Distance intra-squelette (moyenne): {intra_sk:.4f}")
print(f"Distance intra-texte (moyenne): {intra_text:.4f}")
print(f"Distance texte<->squelette, MEME classe (moyenne): {cross_correct:.4f}")
print(f"Distance texte<->squelette, TOUTES paires (moyenne): {cross_all:.4f}")
print(f"\nSi cross_correct << cross_all : l'alignement fonctionne malgre le modality gap visuel")
print(f"Si cross_correct proche de cross_all : la classe correcte n'est pas privilegiee, vrai probleme")
