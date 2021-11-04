"""Create SWA model.

- Author: Hyunwook Kim
- Contact: hwkim@jmarple.ai
"""

import argparse
import gc
import glob
import os

import torch
import tqdm


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        required=True,
        help="directory of trained models to apply SWA",
    )
    parser.add_argument(
        "--swa_model_name", type=str, default="swa.pt", help="file name of SWA model"
    )
    parser.add_argument(
        "--best_num",
        type=int,
        default=5,
        help="the number of trained models to apply SWA",
    )
    return parser.parse_args()


def create_swa_model(model_dir: str, model_name: str, best_num: int) -> None:
    """Save mean weights to apply SWA.

        This function was implemented with reference to: https://github.com/hyz-xmaster/swa_object_detection/blob/master/swa/get_swa_model.py.

    Args:
        model_dir: directory of trained models to apply SWA
        model_name: file name of SWA model
        best_num: the number of trained models to apply SWA
    """
    # Load trained models and select best models
    model_paths = glob.glob(f"{model_dir}/epoch_*.pt")
    models_with_map50 = []
    for model_path in model_paths:
        model = torch.load(model_path)
        map50 = model["mAP50"]
        models_with_map50.append((map50, model_path))
        del model
        gc.collect()

    models_with_map50.sort(reverse=True)
    models = [
        torch.load(p) for i, (_, p) in enumerate(models_with_map50) if i < best_num
    ]

    # Extract model information
    model_num = len(models)
    model_type = "ema" if models[-1].get("ema") else "model"
    state_dict = models[-1][model_type].state_dict()
    model_keys = state_dict.keys()
    new_state_dict = state_dict.copy()

    # Apply SWA and save SWA model
    swa_model = models[-1]
    for key in tqdm.tqdm(model_keys, "Apply SWA"):
        sum_weight = 0.0
        for m in models:
            sum_weight += m[model_type].state_dict()[key]
        avg_weight = sum_weight / model_num
        new_state_dict[key] = avg_weight
    swa_model[model_type].load_state_dict(new_state_dict)
    save_dir = os.path.join(model_dir, model_name)
    torch.save(swa_model, save_dir)
    print(f"{model_dir}/{model_name} was saved.")


if __name__ == "__main__":
    args = get_parser()
    create_swa_model(
        model_dir=args.model_dir, model_name=args.swa_model_name, best_num=args.best_num
    )
