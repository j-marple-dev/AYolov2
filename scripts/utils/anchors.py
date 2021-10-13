"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from scripts.data_loader.data_loader import LoadImagesAndLabels
    from kindle.modules import YOLOHead

import numpy as np
import torch
import torch.nn as nn
import yaml
from kindle.modules import YOLOHead
from scipy.cluster.vq import kmeans
from tqdm import tqdm

from scripts.data_loader import LoadImagesAndLabels


def check_anchor_order(m: nn.Module) -> None:
    """Check anchor order against stride order for YOLOv5 Detect module."""
    # anchor area
    a = m.anchor_grid.prod(-1).view(-1)  # type: ignore
    # delta
    da = a[-1] - a[0]  # type: ignore
    # delta s
    ds = m.stride[-1] - m.stride[0]  # type: ignore
    if da.sign() != ds.sign():  # same order
        print("Reversing anchor order")
        m.anchors[:] = m.anchors.flip(0)  # type: ignore
        m.anchor_grid[:] = m.anchor_grid.flip(0)  # type: ignore


def kmean_anchors(
    path: Union[str, "LoadImagesAndLabels"] = "./data/coco128.yaml",  # noqa
    n: int = 9,
    img_size: int = 640,
    thr: float = 4.0,
    gen: int = 1000,
    verbose: bool = True,
) -> np.ndarray:
    """Create kmeans-evolved anchors from training dataset.

    Args:
        path: path to dataset *.yaml, or a loaded dataset
        n: number of anchors
        img_size: image size used for training
        thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
        gen: generations to evolve anchors using genetic algorithm

    Return:
        k: kmeans evolved anchors

    Usage:
        from utils.general import *; _ = kmean_anchors()
    """
    thr = 1.0 / thr

    def metric(
        k: Union[torch.Tensor, np.ndarray], wh: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[Union[torch.Tensor], Union[torch.Tensor]]:  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1.0 / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def fitness(k: np.ndarray) -> torch.Tensor:  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)  # type: ignore
        return (best * (best > thr).float()).mean()  # type: ignore  # fitness

    def print_results(k: np.ndarray) -> np.ndarray:
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)  # type: ignore
        bpr, aat = (
            (best > thr).float().mean(),
            (  # type: ignore
                x > thr
            )
            .float()
            .mean()
            * n,
        )  # best possible recall, anch > thr
        print(
            "thr=%.2f: %.4f best possible recall, %.2f anchors past thr"
            % (thr, bpr, aat)
        )
        print(
            "n=%g, img_size=%s, metric_all=%.3f/%.3f-mean/best, past_thr=%.3f-mean: "
            % (n, img_size, x.mean(), best.mean(), x[x > thr].mean()),  # type: ignore
            end="",
        )
        for i, x in enumerate(k):
            print(
                "%i,%i" % (round(x[0]), round(x[1])),  # type: ignore
                end=",  " if i < len(k) - 1 else "\n",
            )  # use in *.cfg
        return k

    if isinstance(path, str):  # *.yaml file
        with open(path) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        from utils.datasets import LoadImagesAndLabels

        dataset = LoadImagesAndLabels(data_dict["train"], augment=True, rect=True)
    else:
        dataset = path  # dataset

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print(
            "WARNING: Extremely small objects found. "
            "%g of %g labels are < 3 pixels in width or height." % (i, len(wh0))
        )
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels

    # Kmeans calculation
    print("Running kmeans for %g anchors on %g points..." % (n, len(wh)))
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unflitered
    k = print_results(k)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.tight_layout()
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    npr = np.random
    f_, sh, mp, s = (
        fitness(k),
        k.shape,
        0.9,
        0.1,
    )  # fitness, generations, mutation prob, sigma
    pbar = tqdm(
        range(gen), desc="Evolving anchors with Genetic Algorithm"
    )  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(
                0.3, 3.0
            )
        kg = (k.copy() * v).clip(min=2.0)
        fg = fitness(kg)
        if fg > f_:
            f_, k = fg, kg.copy()
            pbar.desc = "Evolving anchors with Genetic Algorithm: fitness = %.4f" % f_
            if verbose:
                print_results(k)

    return print_results(k)


def check_anchors(
    dataset: torch.utils.data.Dataset,
    model: nn.Module,
    thr: float = 4.0,
    imgsz: int = 640,
) -> None:
    """Check anchor fit to data, recompute if necessary."""
    print("\nAnalyzing anchors... ", end="")

    if isinstance(model, nn.DataParallel):
        assert isinstance(
            model.module.model, nn.Sequential
        ), "Model must be a sequential"
        m: "YOLOHead" = model.module.model[-1]  # noqa
    else:
        m = model.model[-1]  # type: ignore

    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)  # type: ignore
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    wh = torch.tensor(
        np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])  # type: ignore
    ).float()  # wh

    def metric(k: Union[torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute metric."""
        r = wh[:, None] / k[None]
        x = torch.min(r, 1.0 / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1.0 / thr).float().sum(1).mean()  # anchors above threshold
        bpr = (best > 1.0 / thr).float().mean()  # best possible recall
        return bpr, aat

    bpr, aat = metric(m.anchor_grid.clone().cpu().view(-1, 2))  # type: ignore
    print(
        "anchors/target = %.2f, Best Possible Recall (BPR) = %.4f" % (aat, bpr), end=""
    )
    if bpr < 0.98:  # threshold to recompute
        print(". Attempting to generate improved anchors, please wait...")
        # number of anchors
        na = m.anchor_grid.numel() // 2  # type: ignore
        new_anchors: Union[torch.Tensor, np.ndarray] = kmean_anchors(
            dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False
        )
        new_bpr = metric(new_anchors.reshape(-1, 2))[0]
        if new_bpr > bpr:  # replace anchors
            new_anchors = torch.tensor(new_anchors, device=m.anchors.device).type_as(  # type: ignore
                m.anchors  # type: ignore
            )
            m.anchor_grid[:] = new_anchors.clone().view_as(  # type: ignore
                m.anchor_grid  # type: ignore
            )  # for inference
            m.anchors[:] = new_anchors.clone().view_as(m.anchors) / m.stride.to(  # type: ignore
                m.anchors.device  # type: ignore
            ).view(
                -1, 1, 1
            )  # loss
            check_anchor_order(m)
            print(
                "New anchors saved to model. Update model *.yaml to use these anchors in the future."
            )
            print(new_anchors)
        else:
            print(
                "Original anchors better than new anchors. Proceeding with original anchors."
            )
    print("")  # newline
