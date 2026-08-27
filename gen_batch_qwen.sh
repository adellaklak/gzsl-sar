#!/bin/bash
set -e
source /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/miniconda3/etc/profile.d/conda.sh
conda activate qwen-embed
cd /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/FSVAE-clean-repro

for f in ac mc bd; do
  echo "=== ${f}_60 ==="
  python gen_text_feat_qwen.py --csv sem_info/${f}_60.csv --out text_feats/qwen512/${f}_60.npy --text_col description || \
  python gen_text_feat_qwen.py --csv sem_info/${f}_60.csv --out text_feats/qwen512/${f}_60.npy --text_col label
done

echo "=== Verification ==="
python3 -c "
import numpy as np
for f in ['ac','mc','bd']:
    a = np.load(f'text_feats/qwen512/{f}_60.npy')
    print(f, a.shape)
"
