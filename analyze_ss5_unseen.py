import sys
import numpy as np
import torch
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
print(f"Classes unseen ss=5 (indices 0-based): {unseen_inds.tolist()}")

labels_map, dirv2_map, mdv3_map = {}, {}, {}
with open(f'sem_info/lb_{ntu}.csv') as f:
    r = csv.reader(f); next(r)
    for idx, lb in r: labels_map[int(idx) - 1] = lb
with open(f'sem_info/dirv2_{ntu}.csv') as f:
    r = csv.reader(f); next(r)
    for idx, desc in r: dirv2_map[int(idx) - 1] = desc
with open(f'sem_info/mdv3_{ntu}.csv') as f:
    r = csv.reader(f); next(r)
    for idx, desc in r: mdv3_map[int(idx) - 1] = desc

print("\n=== Les 5 classes unseen : label + description dir + description md ===")
for c in unseen_inds:
    print(f"\n[{c}] {labels_map[c]}")
    print(f"  dirv2: {dirv2_map[c]}")
    print(f"  mdv3:  {mdv3_map[c]}")

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
print(f"\nEchantillons squelette unseen (gtest, ss=5): {sk_mu.shape[0]}")

# Prediction restreinte aux 5 classes unseen (= exactement le calcul ZSL pur)
dist_to_unseen_text = np.linalg.norm(sk_mu[:, None, :] - text_mu_unseen[None, :, :], axis=-1)
pred_local = np.argmin(dist_to_unseen_text, axis=1)
pred_class = unseen_inds[pred_local]
acc = (pred_class == sk_labels).mean()
print(f"Accuracy ZSL (5 classes, nearest-neighbor sur cet embedding): {acc:.1%}  (reference train.py: 80.31%)")

print("\n=== Matrice de confusion (5x5), restreinte aux 5 classes unseen ===")
print(f"{'vrai_vs_predit':30s}" + "".join(f"{labels_map[c][:12]:>14s}" for c in unseen_inds))
for c_true in unseen_inds:
    row = f"{labels_map[c_true][:28]:30s}"
    mask = sk_labels == c_true
    for c_pred in unseen_inds:
        pct = (pred_class[mask] == c_pred).mean() if mask.sum() > 0 else 0
        row += f"{pct:14.1%}"
    print(row)

print("\n=== Distances latentes texte<->texte entre les 5 classes (plus petit = plus confondable) ===")
text_dist = np.linalg.norm(text_mu_unseen[:, None, :] - text_mu_unseen[None, :, :], axis=-1)
print(f"{'':30s}" + "".join(f"{labels_map[c][:12]:>14s}" for c in unseen_inds))
for i, c1 in enumerate(unseen_inds):
    row = f"{labels_map[c1][:28]:30s}"
    for j, c2 in enumerate(unseen_inds):
        row += f"{text_dist[i,j]:14.3f}" if i != j else f"{'--':>14s}"
    print(row)
