import numpy as np
import csv
import sys
from pathlib import Path

# Charge les noms de classes
labels = {}
with open('sem_info/lb_60.csv') as f:
    r = csv.reader(f)
    next(r)
    for idx, lb in r:
        labels[int(idx) - 1] = lb  # -1 : idx CSV commence a 1, indices modele a 0

configs = sys.argv[1:] if len(sys.argv) > 1 else [
    "md", "ad_md", "lb_md", "ac_md", "dirv2_md",
    "lbac_md_bdavg", "bd_md", "lb_bdavg_md", "lbac_dirv2_bdavg", "dirv2_bdavg_md"
]

wdir_base = "results/12_r"
le = "qwen512"

results = {}
for tm in configs:
    p = Path(f"{wdir_base}/{le}/{tm}")
    preds_f = p / "final_preds.npy"
    true_f = p / "true_labels.npy"
    if not preds_f.exists():
        print(f"[MANQUANT] {tm}")
        continue
    preds = np.load(preds_f)
    true = np.load(true_f)
    acc_per_class = {}
    for c in np.unique(true):
        mask = true == c
        acc_per_class[c] = (preds[mask] == true[mask]).mean()
    results[tm] = acc_per_class

if not results:
    print("Aucun resultat trouve — verifie que les jobs sont bien termines.")
    sys.exit(1)

all_classes = sorted(set().union(*[r.keys() for r in results.values()]))
configs_found = list(results.keys())

print(f"{'classe':35s}" + "".join(f"{c:>16s}" for c in configs_found))
for cl in all_classes:
    name = labels.get(cl, f"idx{cl}")[:33]
    row = f"{name:35s}"
    for tm in configs_found:
        v = results[tm].get(cl)
        row += f"{v:16.1%}" if v is not None else f"{'--':>16s}"
    print(row)

print("\n=== Moyenne globale par config ===")
for tm in configs_found:
    vals = list(results[tm].values())
    print(f"{tm:30s} {np.mean(vals):.2%}")
