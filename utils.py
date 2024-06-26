from os import PathLike
from typing import Sequence

from PIL import Image
from numpy import uint8, round
from torch import Tensor

from data import EvaluationDataset


def save_results(results: Sequence[Tensor], output_dir: str | PathLike[str], dataset: EvaluationDataset,
                 batch_idx: int) -> None:
    for i, mask in enumerate(results):
        mask_np = mask.squeeze().cpu().numpy()
        mask_np = round(mask_np * 255).astype(uint8)
        mask_img = Image.fromarray(mask_np, mode="L")
        mask_img.save(f"{output_dir}/{dataset.images[batch_idx * len(results) + i].split('.')[0]}_label.png")
