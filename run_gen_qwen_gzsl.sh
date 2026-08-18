#!/bin/bash
set -e
source /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/miniconda3/etc/profile.d/conda.sh
conda activate qwen-embed
cd /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/FSVAE-clean-repro

for m in lb ad md; do
  for n in 60 120; do
    echo "=== Encodage ${m}_${n}.csv ==="
    python gen_text_feat_qwen.py --csv sem_info/${m}_${n}.csv --out text_feats/qwen512/${m}_${n}.npy
  done
done

echo "=== Terminé ==="
find text_feats/qwen512 -type f
