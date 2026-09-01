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
se_ckpt = torch.load(f'{wdir}/{le}/{tm}/se_{epoch}.pth.tar', map_location=device)
sequence_encoder.load_state_dict(se_ckpt['state_dict'])
sequence_encoder.eval()

tml = tm.split('_')
tfl = [torch.from_numpy(np.load(f'./text_feats/{le}/{m}_{ntu}.npy')) for m in tml]
text_feat = torch.cat(tfl, dim=-1)
text_emb_input_size = text_feat.size(-1)
text_emb = text_feat / torch.norm(text_feat, dim=1, keepdim=True).repeat([1, text_emb_input_size])

text_encoder = Encoder([text_emb_input_size, latent_size]).to(device)
te_ckpt = torch.load(f'{wdir}/{le}/{tm}/te_{epoch}.pth.tar', map_location=device)
text_encoder.load_state_dict(te_ckpt['state_dict'])
text_encoder.eval()

with torch.no_grad():
    text_mu, _ = text_encoder(text_emb.to(device).float())
text_mu = text_mu.cpu().numpy()

ntu_loaders = NTUDataLoaders(dataset_path, 'max', 1)
val_loader = ntu_loaders.get_test_loader(64, 0)
unseen_inds = set(np.load('label_splits/ru12.npy').tolist())

sk_mu_list, sk_labels_list = [], []
with torch.no_grad():
    for inp, target in val_loader:
        t_s = inp.to(device)
        mu, _ = sequence_encoder(t_s)
        sk_mu_list.append(mu.cpu().numpy())
        sk_labels_list.append(target.numpy())

sk_mu = np.concatenate(sk_mu_list, axis=0)
sk_labels = np.concatenate(sk_labels_list, axis=0)
print(f"Skeleton: {sk_mu.shape}, Text: {text_mu.shape}")

MAX_PER_CLASS = 40
rng = np.random.RandomState(0)
keep_idx = []
for c in np.unique(sk_labels):
    idx = np.where(sk_labels == c)[0]
    if len(idx) > MAX_PER_CLASS:
        idx = rng.choice(idx, MAX_PER_CLASS, replace=False)
    keep_idx.append(idx)
keep_idx = np.concatenate(keep_idx)
sk_mu_sub, sk_labels_sub = sk_mu[keep_idx], sk_labels[keep_idx]

sk_norm = np.linalg.norm(sk_mu_sub, axis=1).mean()
text_norm = np.linalg.norm(text_mu, axis=1).mean()
print(f"Norme moyenne squelette: {sk_norm:.3f}, texte: {text_norm:.3f} (ratio: {sk_norm/text_norm:.2f}x)")

sk_mu_sub = sk_mu_sub / np.linalg.norm(sk_mu_sub, axis=1, keepdims=True)
text_mu = text_mu / np.linalg.norm(text_mu, axis=1, keepdims=True)

combined = np.concatenate([sk_mu_sub, text_mu], axis=0)
n_sk = sk_mu_sub.shape[0]

print("t-SNE en cours (peut prendre 1-2 min)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=0, init='pca')
proj = tsne.fit_transform(combined)
sk_proj, text_proj = proj[:n_sk], proj[n_sk:]

labels_map = {}
with open(f'sem_info/lb_{ntu}.csv') as f:
    r = csv.reader(f); next(r)
    for idx, lb in r:
        labels_map[int(idx) - 1] = lb

fig, ax = plt.subplots(1, 1, figsize=(20, 16))
n_classes = len(labels_map)
cmap = plt.cm.get_cmap('tab20', n_classes)

for c in np.unique(sk_labels_sub):
    mask = sk_labels_sub == c
    marker = '^' if c in unseen_inds else 'o'
    ax.scatter(sk_proj[mask, 0], sk_proj[mask, 1], color=cmap(c % 20),
               marker=marker, s=18, alpha=0.55, edgecolors='none')

for c in range(n_classes):
    is_unseen = c in unseen_inds
    ax.scatter(text_proj[c, 0], text_proj[c, 1], color=cmap(c % 20), marker='*', s=400,
               edgecolors='red' if is_unseen else 'black', linewidths=1.5, zorder=5)
    ax.annotate(labels_map[c], (text_proj[c, 0], text_proj[c, 1]), fontsize=7,
                fontweight='bold', zorder=6, xytext=(3, 3), textcoords='offset points')

ax.set_title('NTU-60 ss=12 (esterel29) — lb_dirv2_mdv3\n'
              'rond=squelette seen, triangle=squelette unseen, etoile=texte (contour rouge=unseen classe)',
              fontsize=13)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig('tsne_ss12_champion.png', dpi=150)
print("Sauvegarde : tsne_ss12_champion.png")

print("\n=== VERIFICATION QUANTITATIVE (espace reel 100D, pas le t-SNE) ===")

# 1. Le texte est-il vraiment tasse en variance absolue, ou juste relativement au squelette ?
text_mu_raw = text_mu  # deja normalise plus haut, mais on regarde la dispersion inter-classe
text_pairwise_dist = np.linalg.norm(text_mu[:, None, :] - text_mu[None, :, :], axis=-1)
np.fill_diagonal(text_pairwise_dist, np.nan)
print(f"Distance texte-texte : min={np.nanmin(text_pairwise_dist):.4f}, "
      f"moyenne={np.nanmean(text_pairwise_dist):.4f}, max={np.nanmax(text_pairwise_dist):.4f}")

sk_centroids = np.array([sk_mu_sub[sk_labels_sub == c].mean(axis=0) for c in range(n_classes)])
sk_centroids = sk_centroids / np.linalg.norm(sk_centroids, axis=1, keepdims=True)
sk_pairwise_dist = np.linalg.norm(sk_centroids[:, None, :] - sk_centroids[None, :, :], axis=-1)
np.fill_diagonal(sk_pairwise_dist, np.nan)
print(f"Distance squelette-squelette (centroides) : min={np.nanmin(sk_pairwise_dist):.4f}, "
      f"moyenne={np.nanmean(sk_pairwise_dist):.4f}, max={np.nanmax(sk_pairwise_dist):.4f}")

# 2. Le VRAI test d'alignement : pour chaque classe, son propre texte est-il le plus proche
#    voisin parmi les 60 embeddings texte, en partant du centroide squelette de cette classe ?
dist_sk_to_text = np.linalg.norm(sk_centroids[:, None, :] - text_mu[None, :, :], axis=-1)
nearest_text = np.argmin(dist_sk_to_text, axis=1)
correct = (nearest_text == np.arange(n_classes))
print(f"\nNearest-neighbor squelette-centroide -> texte : {correct.sum()}/{n_classes} classes correctes ({correct.mean():.1%})")
print("Classes ou le nearest-neighbor texte est FAUX :")
for c in range(n_classes):
    if not correct[c]:
        print(f"  {labels_map[c]:35s} -> plus proche de: {labels_map[nearest_text[c]]}")
