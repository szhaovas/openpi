# /// script
# requires-python = "==3.10.*"
# dependencies = [
#     "torch==2.0.0",
#     "torchvision>=0.15.0",
#     "omegaconf",
#     "torchmetrics==0.10.3",
#     "fvcore",
#     "xformers==0.0.18",
#     "iopath",
#     "submitit",
#     "cuml-cu11",
#     "numpy<2",
# ]
# ///

import glob
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_distances
from torchvision import transforms
from tqdm import tqdm

from src.easy_utils import extract_scheduler_nevals, update_csv_column

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    os.environ["XFORMERS_DISABLED"] = "1"

model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
model.eval().to(device)

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # from ImageNet
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)


def dino_diversity_among_images(
    img_dir: Path, embed_batch_size: int = 16
) -> float:
    image_filenames = glob.glob(str(img_dir / "*.png"))

    embeddings = []
    for i in range(0, len(image_filenames), embed_batch_size):
        images = [
            transform(Image.open(filename).convert("RGB"))
            for filename in image_filenames[i : i + embed_batch_size]
        ]

        with torch.no_grad():
            batch_emb = model(torch.stack(images).to(device)).cpu().numpy()

        embeddings.append(batch_emb)

    embeddings = np.vstack(embeddings)

    return cosine_distances(embeddings).sum() / (
        embeddings.shape[0] * (embeddings.shape[0] - 1)
    )  # exclude zeroes along the diagonal


def compute_dino_diversities(log_dir: Path) -> np.ndarray:
    ckpt_nevals = list(extract_scheduler_nevals(str(log_dir)).values())

    dino_diversities = []
    for nevals in tqdm(ckpt_nevals):
        dino_diversities.append(
            dino_diversity_among_images(log_dir / f"env_images_{nevals:08d}")
        )

    dino_diversities = np.array(dino_diversities)
    dino_diversities = dino_diversities[np.argsort(ckpt_nevals)]

    return dino_diversities


if __name__ == "__main__":
    log_dir = Path("outputs/cma_es/2026-04-14_190728")

    dino = compute_dino_diversities(log_dir)

    update_csv_column(
        log_dir / "summary.csv",
        "Avg.EmbDist(DINOv2)",
        dino.tolist(),
        Path("summary.csv"),
    )
