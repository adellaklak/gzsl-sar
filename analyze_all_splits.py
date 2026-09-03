import numpy as np
import csv
from pathlib import Path
from collections import Counter

TWO_PERSON_60 = set(range(50, 61))          # idx 50-60
TWO_PERSON_120_EXTRA = set(range(106, 121)) # idx 106-120

def load_labels(ntu):
    path = f"sem_info/lb_{ntu}.csv"
    labels = {}
    with open(path) as f:
        r = csv.reader(f); next(r)
        for idx, lb in r:
            labels[int(idx)] = lb
    return labels

def is_two_person(idx, ntu):
    if idx in TWO_PERSON_60:
        return True
    if ntu == 120 and idx in TWO_PERSON_120_EXTRA:
        return True
    return False

SPLITS = [
    (60, 5,  "results/5_r/qwen512/lb_dirv2_mdv3", "label_splits/ru5.npy"),
    (60, 12, "results/12_r/qwen512/lb_dirv2_mdv3", "label_splits/ru12.npy"),
    (120, 10, "results/10_r/qwen512/lb_dirv2_mdv3", "label_splits/ru10.npy"),
    (120, 24, "results/24_r/qwen512/lb_dirv2_mdv3", "label_splits/ru24.npy"),
]

for ntu, ss, wdir, unseen_path in SPLITS:
    print(f"\n{'='*90}\nNTU-{ntu} ss={ss}\n{'='*90}")
    p = Path(wdir)
    try:
        preds = np.load(p / "final_preds.npy")
        true = np.load(p / "true_labels.npy")
    except FileNotFoundError:
        print(f"  [MANQUANT] {wdir}/final_preds.npy — pas encore genere pour ce split")
        continue

    labels = load_labels(ntu)
    unseen_inds = set(np.load(unseen_path).tolist())  # 0-indexe

    rows = []
    for cl in sorted(set(true.tolist())):
        mask = true == cl
        n = mask.sum()
        acc = (preds[mask] == true[mask]).mean()
        idx1 = cl + 1  # csv est 1-indexe
        seen_flag = "Unseen" if cl in unseen_inds else "Seen"
        tp_flag = "2P" if is_two_person(idx1, ntu) else "1P"
        wrong = preds[mask][preds[mask] != cl]
        top_wrong = Counter(wrong).most_common(2)
        confused_str = ", ".join(f"{labels.get(c+1,c)}({n2}x)" for c, n2 in top_wrong) if top_wrong else "-"
        rows.append((acc, labels.get(idx1, idx1), seen_flag, tp_flag, n, confused_str))

    rows.sort(key=lambda r: r[0])

    print(f"\n--- BLOQUEES (<10%) ---")
    for acc, name, sf, tp, n, conf in rows:
        if acc < 0.10:
            print(f"  {acc:5.1%}  [{sf:7s}][{tp}] {name:35s} (n={n:3d})  confondu avec: {conf}")

    print(f"\n--- FAIBLES (10-40%) ---")
    for acc, name, sf, tp, n, conf in rows:
        if 0.10 <= acc < 0.40:
            print(f"  {acc:5.1%}  [{sf:7s}][{tp}] {name:35s} (n={n:3d})  confondu avec: {conf}")

    print(f"\n--- BONNES (70-100%) ---")
    for acc, name, sf, tp, n, conf in rows:
        if acc >= 0.70:
            print(f"  {acc:5.1%}  [{sf:7s}][{tp}] {name:35s} (n={n:3d})  confondu avec: {conf}")
    print(f"\n--- MOYENNES (40-70%) ---")
    for acc, name, sf, tp, n, conf in rows:
        if 0.40 <= acc < 0.70:
            print(f"  {acc:5.1%}  [{sf:7s}][{tp}] {name:35s} (n={n:3d})  confondu avec: {conf}")

    n_2p_weak = sum(1 for acc, _, _, tp, _, _ in rows if acc < 0.40 and tp == "2P")
    n_1p_weak = sum(1 for acc, _, _, tp, _, _ in rows if acc < 0.40 and tp == "1P")
    n_2p_total = sum(1 for _, _, _, tp, _, _ in rows if tp == "2P")
    n_1p_total = sum(1 for _, _, _, tp, _, _ in rows if tp == "1P")
    print(f"\n  Resume 2P: {n_2p_weak}/{n_2p_total} classes 2-personnes sous 40%")
    print(f"  Resume 1P: {n_1p_weak}/{n_1p_total} classes 1-personne sous 40%")
