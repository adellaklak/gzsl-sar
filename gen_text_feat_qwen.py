"""
gen_text_feat_qwen.py
Genere les features texte pour DescVAE/FS-VAE avec Qwen3-VL-Embedding-8B.
ENVIRONNEMENT : ne PAS lancer dans l'env fsvae (torch 1.12).
Convention : text_feats/{le}/{nom}_{ntu}.npy (cf. train.py ligne 99).
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

SCRIPTS_DIR = Path(
    "/srv/storage/stars@storage3.sophia.grid5000.fr/qmerille/qmerille/llm/Qwen3-VL-Embedding-8B/scripts"
)
sys.path.append(str(SCRIPTS_DIR))
from qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402

MODEL_PATH = (
    "/srv/storage/stars@storage3.sophia.grid5000.fr/qmerille/qmerille/llm/Qwen3-VL-Embedding-8B"
)

INSTRUCTION_NTU = (
    "Represent this description of a human action for skeleton-based "
    "action recognition."
)


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu").float().numpy()
    return np.asarray(x, dtype=np.float32)


class Qwen3VLTextEncoder:
    def __init__(self, model_path: str, instruction: Optional[str],
                 batch_size: int = 32, mrl_dim: int = 512):
        print(f"Loading Qwen3-VL-Embedding-8B from {model_path}...", flush=True)
        self.model = Qwen3VLEmbedder(model_name_or_path=model_path)
        self.instruction = instruction
        self.batch_size = batch_size
        self.mrl_dim = mrl_dim
        print("Qwen3-VL-Embedding loaded.\n", flush=True)

    def encode(self, texts: List[str]) -> np.ndarray:
        all_emb = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            if self.instruction:
                inputs = [{"text": t, "instruction": self.instruction} for t in batch]
            else:
                inputs = [{"text": t} for t in batch]

            raw_embeddings = self.model.process(inputs)
            emb = np.stack([to_numpy(e) for e in raw_embeddings])

            assert emb.shape[1] >= self.mrl_dim, (
                f"Dimension du modele ({emb.shape[1]}) < mrl_dim ({self.mrl_dim})"
            )
            emb = emb[:, :self.mrl_dim]
            norms = np.linalg.norm(emb, axis=-1, keepdims=True)
            emb = emb / np.clip(norms, a_min=1e-8, a_max=None)

            all_emb.append(emb.astype(np.float32))
        return np.concatenate(all_emb, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Encode un CSV de descriptions NTU avec Qwen3-VL-Embedding-8B."
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--text_col", default=None)
    parser.add_argument("--mrl_dim", type=int, default=512)
    parser.add_argument("--no_instruction", action="store_true")
    parser.add_argument("--custom_instruction", type=str, default=None,
                         help="Remplace INSTRUCTION_NTU par ce texte")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    df = pd.read_csv(args.csv).sort_values("idx")

    text_col = args.text_col
    if text_col is None:
        for candidate in ("label", "dcp"):
            if candidate in df.columns:
                text_col = candidate
                break
        if text_col is None:
            raise ValueError(
                f"Impossible de determiner la colonne texte dans {args.csv} "
                f"(colonnes trouvees: {list(df.columns)}). Utilise --text_col."
            )

    texts = df[text_col].tolist()
    print(f"{len(texts)} entrees chargees depuis {args.csv} (colonne '{text_col}')")

    if args.no_instruction:
        instruction = None
    elif args.custom_instruction:
        instruction = args.custom_instruction
    else:
        instruction = INSTRUCTION_NTU
    encoder = Qwen3VLTextEncoder(MODEL_PATH, instruction,
                                  batch_size=args.batch_size, mrl_dim=args.mrl_dim)

    embeddings = encoder.encode(texts)
    print(f"Shape finale : {embeddings.shape}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    print(f"Sauvegarde : {out_path}")


if __name__ == "__main__":
    main()
