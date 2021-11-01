import random

import torch
from torch import nn

from scripts.utils.wandb_utils import get_ckpt_path, load_model_from_wandb


def test_wandb_loader(force: bool = False, p: float = 0.5, wandb_path: str = ""):
    if not force:
        return

    if random.random() > p:
        return

    model = load_model_from_wandb(wandb_path)
    assert isinstance(model, nn.Module)

    ckpt_path = get_ckpt_path(wandb_path)
    ckpt = torch.load(ckpt_path)
    assert ckpt is not None


if __name__ == "__main__":
    test_wandb_loader()
