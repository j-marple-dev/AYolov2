"""Wandb utilities.

- Author: Hyunwook Kim
- Contact: hwkim@jmarple.ai
"""
import os
from typing import Optional

from torch import nn

import wandb
from scripts.utils.torch_utils import load_pytorch_model


def download_from_wandb(
    wandb_run: wandb.apis.public.Run,
    wandb_path: str,
    local_root: str,
    force: bool = False,
) -> str:
    """Download file from wandb.

    Args:
        wandb_run: wandb run object
        wandb_path: file path to download from wandb
        local_root: root directory to save file from wandb
        force: force download file from wandb

    Returns:
        download_path: downloaded file path
    """
    download_path = os.path.join(local_root, wandb_path)
    if force or not os.path.isfile(download_path):
        wandb_run.file(wandb_path).download(local_root, replace=True)

    return download_path


def summary(wandb_run: wandb.apis.public.Run, model: nn.Module = None) -> None:
    """Print summary of model from wandb.

    Args:
        wandb_run: wandb run object
        model: pytorch model from wandb
    """
    wandb_map50 = wandb_run.summary.get("mAP50", 0.0)
    print(f"Model from wandb (wandb url: {wandb_run.url})")
    print(f":: {wandb_run.project}/{wandb_run.name} - #{', #'.join(wandb_run.tags)}")
    print(f":: mAP@0.5: {wandb_map50:.4f}")
    if model and isinstance(model.model, nn.Module):
        n_param = sum([p.numel() for p in model.model.parameters()])
        print(f":: # parameters: {n_param:,d}")


def get_ckpt_path_from_wandb(
    wandb_path: str,
    weight_path: str = "best.pt",
    download_root: str = "wandb/downloads",
) -> str:
    """Load a checkpoint dictionary from a wandb run path.

    Args:A
        wandb_path: run path in wandb
        weight_path: weight path in wandb
        download_root: root directory to download files from wandb
    Returns:
        ckpt_path: checkpoint file (.pt file) path
    """
    api = wandb.Api()
    wandb_run = api.run(wandb_path)
    download_root = os.path.join(download_root, wandb_path)
    ckpt_path = download_from_wandb(wandb_run, weight_path, download_root)
    summary(wandb_run)
    return ckpt_path


def load_model_from_wandb(
    wandb_path: str,
    weight_path: str = "best.pt",
    download_root: str = "wandb/downloads",
    load_weights: bool = True,
    verbose: int = 1,
) -> Optional[nn.Module]:
    """Load a model from a wandb run path.

    Args:
        wandb_path: run path in wandb
        weight_path: weight path in wandb
        download_root: root directory to download files from wandb
        load_weights: load weights from wandb run path
        verbose: level to print model information
    Returns:
        PyTorch model,
        None if loading model from wandb is failed.
    """
    api = wandb.Api()
    wandb_run = api.run(wandb_path)
    download_root = os.path.join(download_root, wandb_path)
    if load_weights:
        ckpt_path = download_from_wandb(wandb_run, weight_path, download_root)
        model = load_pytorch_model(ckpt_path)

    if verbose > 0:
        summary(wandb_run)
    return model
