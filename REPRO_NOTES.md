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

## Qwen3-VL-Embedding-8B vs CLIP ViT-B/32 (lb_ad_md, mrl_dim=512, seed=5, epochs pleins)

| Split | ZSL CLIP | ZSL Qwen | H CLIP | H Qwen | H papier (CLIP, †) |
|---|---|---|---|---|---|
| NTU-60 ss=5   | 85.32 | 85.18 | 72.55 | 74.30 | 75.7 |
| NTU-60 ss=12  | 54.04 | 63.30 | 50.40 | 56.99 | 52.1 |
| NTU-120 ss=10 | 74.45 | 77.86 | 59.39 | 59.46 | 63.3 |
| NTU-120 ss=24 | 61.99 | 63.26 | 52.98 | 54.01 | 54.7 |

Gain net et cohérent avec les essais DescVAE précédents, concentré sur ss=12 (+9.26 ZSL, +6.59 H).
Un seul seed testé (seed=5, en dur dans train.py) — pas encore de variance multi-seed sur ce repo.

## Axe `dir` (source->destination du mouvement, genere par Claude)

Premier test lance par erreur : dir concatene A lb_ad_md (4 composantes), pas le protocole voulu.
Resultat conserve pour info seulement (question dilution vs redondance), PAS representatif du potentiel de dir seul :

| Config | ZSL | H | Delta H vs lb_ad_md (56.99) |
|---|---|---|---|
| lb_ad_md_dirv1 | 59.97 | 55.52 | -1.47 |
| lb_ad_md_dirv2 | 62.45 | 56.60 | -0.39 |

Sous-resultat retenu malgre tout : divergence structurelle entre phrases-paires reduit la casse (+1.08 H entre v1/v2),
pertinent pour la redaction future de rel/phase/neg.

Vrai protocole (dir seul, et dir substitue a ad) : voir tableau ci-dessous une fois lance.

## Protocole correct : dir seul et dir en substitution d'ad

| Config | ZSL | H | Delta H vs lb_ad_md (56.99) |
|---|---|---|---|
| dirv2 seul | 47.77 | 48.01 | (bien au-dessus du hasard ~8.3%, signal reel) |
| lb_dirv2_md (dir remplace ad) | 65.32 | 59.00 | **+2.01** |

Premier resultat positif net du chantier descriptions. Confirme l'hypothese dilution (echec en ajout ligne precedente)
vs substitution (succes ici) : dir porte une info non-redondante avec md/lb mais partiellement redondante ou
moins complementaire quand empilee sur ad plutot que substituee.
Seed=5 uniquement, ss=12 uniquement pour l'instant. A verifier : autres seeds, autres splits (ss=5, NTU-120),
et si dir tient aussi en substitution dans une config plus riche (ex: lbac_md_bdavg_cvg).

## md patch final : mdv3 — SUCCES (nouveau meilleur all-time)

Apres deux echecs (mdn: reecriture complete, -5.52 H ; mdv2: patch 5 lignes sans contrainte negative, -0.83 H),
mdv3 applique un principe decouvert dans un chantier separe (generation de squelettes MotionGPT3, sans lien
technique direct) : une contrainte NEGATIVE explicite ("never lowering", "without touching it") compte autant
que la description positive pour separer deux classes confondues.

lb_dirv2_mdv3 (ss=12, Qwen) : ZSL=66.33, S=60.67, U=60.09, H=60.38
vs lb_dirv2_md (H=59.00) : +1.38 H, +3.33 ZSL

Diagnostic par classe : pickup bouge pour la premiere fois du chantier (0.0% -> 12.1%, toujours faible mais signal reel),
tear_up_paper recupere la regression causee par mdv2, walking_apart continue de s'ameliorer (83.3%).
walking_towards reste bloque (0.4%) — confirme le diagnostic structurel, pas descriptif.

Nouveau meilleur all-time du chantier descriptions.

## Comparaison multi-split du champion (lb_dirv2_mdv3) — premiers résultats bruts

| Split | ZSL | S_Acc | U_Acc | H |
|---|---|---|---|---|
| NTU-60 ss=5 | 80.38 | 74.75 | 68.29 | 71.37 |
| NTU-60 ss=12 | 66.45 | 57.29 | 54.98 | 56.11 |
| NTU-120 ss=10 | 79.91 | 64.28 | 65.71 | 64.99 |
| NTU-120 ss=24 | 65.31 | 57.49 | 51.97 | 54.59 |

## Point de reproductibilité découvert (important)

ss=12 donne H=56.11 ici (nœud esterel19) vs H=60.38 la semaine derniere (nœud esterel29), meme config exacte,
meme seed=5. Stage 1 (ZSL brut) quasi identique (66.45 vs 66.33) — la divergence vient du gating (stages 2-4),
pas du VAE. Hypothese : cudnn.deterministic garantit la reproductibilite sur le MEME GPU physique, pas forcement
entre deux modeles de GPU differents. A verifier : relancer plusieurs fois sur le meme noeud pour confirmer,
et si confirme, toujours reporter le noeud utilise pour tout resultat final cite dans le papier.

## Signal de sur-ajustement a ss=12

lb_dirv2_mdv3 (H=71.37) est INFERIEUR a la baseline simple lb_ad_md (H=74.30) sur ss=5 — jamais teste sur ce
split avant aujourd'hui, toute la construction (dir, mdv3) optimisee exclusivement sur ss=12. A investiguer.

## Reproductibilite inter-noeuds : CONFIRMEE dependante du GPU physique

Rerun exact de lb_dirv2_mdv3 ss=12 sur esterel29 : ZSL=66.33, S=60.67, U=60.09, H=60.38 —
identique au centime pres au run de la semaine derniere sur le meme noeud. cudnn.deterministic
garantit donc une reproductibilite bit-a-bit sur un GPU physique donne, mais PAS entre noeuds
de modeles differents (esterel19 donnait H=56.11 sur la config strictement identique).

REGLE ADOPTEE : tout chiffre final destine au papier doit systematiquement citer le noeud esterel
utilise, et idealement etre confirme par un rerun sur ce meme noeud avant publication.

## Decouverte critique : les noms esterelN sont des pools multi-machines

Confirme via hostname explicite : esterel29 recouvre au moins esterel29-3 et esterel29-4, physiquement
distincts, donnant des H differents sur la config identique (59.80 vs 57.65 dans le sweep 12-noeuds).
"-p esterel29" ne garantit PAS de retomber sur la meme carte. Meme mecanisme que musa (musa-3/4/5 deja vus).

REGLE DEFINITIVE : cibler la machine physique precise avec le suffixe (ex: -p esterel29-4), jamais juste
le nom de cluster, pour tout resultat destine a etre cite ou reproduit dans le papier. Ajouter `hostname`
en tete de tout script de resultat final.

## Benchmark final 4-splits — lb_dirv2_mdv3 sur esterel29 (reference stable, 4/4 sous-noeuds confirmes identiques)

| Split | ZSL | S_Acc | U_Acc | H champion | H baseline lb_ad_md | Delta |
|---|---|---|---|---|---|---|
| NTU-60 ss=5   | 80.31 | 77.24 | 69.10 | 72.94 | 74.30 | -1.36 |
| NTU-60 ss=12  | 66.33 | 60.67 | 60.09 | 60.38 | 56.99 | +3.39 |
| NTU-120 ss=10 | 79.91 | 64.28 | 65.71 | 64.99 | 59.46 | +5.53 |
| NTU-120 ss=24 | 65.31 | 57.49 | 51.97 | 54.59 | 54.01 | +0.58 |

Gain net sur 3/4 splits, ss=5 legerement sous la baseline (-1.36, dans la marge de variance observee).
Le signal de sur-ajustement a ss=12 suspecte precedemment etait en partie du au bug de reproductibilite
inter-noeuds (le run ss=5 initial, non-trace par hostname, donnait H=71.37 au lieu de 72.94).
esterel29 confirme stable sur les 4 sous-machines (-1 a -4), utilise comme reference pour ce benchmark.

## Chantier instruction Qwen — teste et clos, INSTRUCTION_NTU (deja en place) reste la meilleure

lb_dirv2_mdv3, ss=12, esterel29, 3 variantes d'instruction d'encodage :

| Instruction | ZSL | H | Delta vs default |
|---|---|---|---|
| aucune (--no_instruction) | 60.31 | 54.64 | -5.74 |
| custom alignement-squelette (longue, technique) | 64.74 | 56.66 | -3.72 |
| INSTRUCTION_NTU (default, deja utilisee partout) | 66.33 | 60.38 | reference |

Confirme que l'instruction aide bien (noinstr nettement pire), mais l'instruction courte et generique deja
en place bat une instruction plus longue et technique specifique au squelette/ShiftGCN. Hypothese : le detail
technique dilue le signal semantique de l'action plutot que de le renforcer. Pas de v2 a tenter, INSTRUCTION_NTU
conservee telle quelle pour toute la suite du chantier.

## Complement benchmark 4-splits : ZSL pur (stage 1) vs H final

| Split | ZSL champion | ZSL baseline | Delta ZSL | H champion | H baseline | Delta H |
|---|---|---|---|---|---|---|
| NTU-60 ss=5   | 80.31 | 85.18 | -4.87 | 72.94 | 74.30 | -1.36 |
| NTU-60 ss=12  | 66.33 | 63.30 | +3.03 | 60.38 | 56.99 | +3.39 |
| NTU-120 ss=10 | 79.91 | 77.86 | +2.05 | 64.99 | 59.46 | +5.53 |
| NTU-120 ss=24 | 65.31 | 63.26 | +2.05 | 54.59 | 54.01 | +0.58 |

Note : sur ss=5, la perte ZSL brute (-4.87) est plus marquee que la perte H finale (-1.36) — le gating
compense une partie du deficit de discrimination pure sur ce split, mais celui-ci reste reel au niveau ZSL.
