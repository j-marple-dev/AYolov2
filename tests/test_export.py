import random

import torch

from scripts.utils.wandb_utils import get_ckpt_path_from_wandb


def test_wandb_loader_for_export(force: bool = False, p: float = 0.5):
    if not force:
        return

    if random.random() > p:
        return

    # You should define wandb run path that you want to load.
    # e.g. wandb_path = "j-marple/AYolov2/3a1r9rb"
    wandb_path = "j-marple/AYolov2/5v1o0e54"
    ckpt_path = get_ckpt_path_from_wandb(wandb_path)
    ckpt = torch.load(ckpt_path)
    if isinstance(ckpt, dict):
        ckpt_model = ckpt["ema"] if "ema" in ckpt.keys() else ckpt["model"]
    else:
        ckpt_model = ckpt

    if ckpt_model:
        ckpt_model = ckpt_model.cpu().float()

    assert ckpt_model is not None


if __name__ == "__main__":
    test_wandb_loader_for_export()
