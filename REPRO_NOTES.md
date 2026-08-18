# Reproduction FS-VAE — notes de validation

## Résultats obtenus (lb_ad_md, seed=5 codé en dur, epochs pleins)

| Split | H obtenu | H papier (Table 1, †) | Écart |
|---|---|---|---|
| NTU-60 ss=5  | 72.55 | 75.7 | -3.15 |
| NTU-60 ss=12 | 50.40 | 52.1 | -1.70 |
| NTU-120 ss=10| 59.39 | 63.3 | -3.91 |
| NTU-120 ss=24| 52.98 | 54.7 | -1.72 |

## Constats
- Écart systématique, toujours dans le même sens (jamais au-dessus).
- `train.py` original n'avait pas `cudnn.deterministic=True`/`benchmark=False` malgré un seed fixé en dur (`seed=5`, ligne ~68) — fix ajouté (commit `0ac7ff5`), mais **résultat rigoureusement identique** (H=50.40 avant/après sur ss=12) → ce n'était pas la source de l'écart, contrairement à ce qu'on avait vu sur DescVAE.
- Hypothèse non testée : le papier rapporte peut-être le meilleur seed sur plusieurs essais plutôt que seed=5 codé en dur. Pas creusé plus loin pour l'instant (décision du 18/08).
- `fsvae_120.sh` référence `train_neg_t.py`, absent du repo officiel publié — remplacé par `train.py` (validé smoke test + run complet, résultats sensés).

## Fichiers de référence
- `repro_60.sh` / `repro_120.sh` : run complet lb_ad_md seul (utilisés pour le tableau ci-dessus)
- `smoke_60.sh` / `smoke_120.sh` : validation rapide du pipeline (nc=2, nepc=50)
