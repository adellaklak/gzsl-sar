import numpy as np
import csv
from pathlib import Path

labels = {}
with open('sem_info/lb_60.csv') as f:
    r = csv.reader(f); next(r)
    for idx, lb in r:
        labels[int(idx) - 1] = lb

patched_classes = [5, 10, 11, 12, 15, 58, 59]  # pickup, reading, writing, tear_up_paper, wear_shoe, walk_towards, walk_apart

configs = ["dirv2_bdavg_md", "dirv3_bdavg_md"]
wdir_base = "results/12_r/qwen512"

print(f"{'classe':30s} {'dirv2_bdavg_md':>16s} {'dirv3_bdavg_md':>16s}")
for cl in patched_classes:
    row = f"{labels.get(cl, cl)[:28]:30s}"
    for tm in configs:
        p = Path(f"{wdir_base}/{tm}")
        try:
            preds = np.load(p / "final_preds.npy")
            true = np.load(p / "true_labels.npy")
            mask = true == cl
            acc = (preds[mask] == true[mask]).mean() if mask.sum() > 0 else float('nan')
            row += f"{acc:16.1%}"
        except FileNotFoundError:
            row += f"{'--':>16s}"
    print(row)
