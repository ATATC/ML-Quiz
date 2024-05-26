from typing import Any
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
from os import listdir, PathLike


class TrainingDataset(Dataset):
    def __init__(self, image_path: str | PathLike[str], mask_path: str | PathLike[str], transform: Any | None = None) -> None:
        self.image_dir: str = image_path
        self.mask_dir: str = mask_path
        self.transform: Any | None = transform
        self.images: tuple[str, ...] = tuple(listdir(image_path))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        image = Image.open(f"{self.image_dir}/{self.images[idx]}").convert("RGB")
        mask = Image.open(f"{self.mask_dir}/{self.images[idx].split('.')[0]}_label.png").convert("L")
        return self.transform(image) if self.transform else image, self.transform(mask) if self.transform else mask


class EvaluationDataset(Dataset):
    def __init__(self, image_path: str | PathLike[str], transform: Any | None = None) -> None:
        self.image_dir: str = image_path
        self.transform: Any | None = transform
        self.images: tuple[str, ...] = tuple(listdir(image_path))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tensor:
        image = Image.open(f"{self.image_dir}/{self.images[idx]}").convert("RGB")
        return self.transform(image) if self.transform else image
