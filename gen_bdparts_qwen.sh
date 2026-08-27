#!/bin/bash
set -e
source /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/miniconda3/etc/profile.d/conda.sh
conda activate qwen-embed
cd /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/FSVAE-clean-repro

for p in head hand arm hip leg foot; do
  echo "=== bp_${p} ==="
  python gen_text_feat_qwen.py --csv sem_info/bp_${p}_60.csv --out text_feats/qwen512/bp_${p}_60.npy --text_col description
done
