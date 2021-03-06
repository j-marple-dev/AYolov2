"""Optuna objective with validator module.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict

import optuna
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import nn

from scripts.data_loader.data_loader import LoadImagesAndLabels
from scripts.objective.abstract_objective import AbstractObjective
from scripts.utils.logger import get_logger
from scripts.utils.torch_utils import count_param
from scripts.utils.train_utils import YoloValidator
from scripts.utils.wandb_utils import load_model_from_wandb

LOGGER = get_logger(__name__)


class ObjectiveValidator(AbstractObjective):
    """Objective class for validation parameters optimizer."""

    """Base mAP50 threshold.

    If the mAP50 of the test model is under this value,
    trial objective score becomes zero.
    """
    BASE_mAP50: float = 0.688
    gt_path = os.path.join("tests", "res", "instances_val2017.json")
    # json_path = "answersheet_4_04_000000.json"

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        cfg: Dict[str, Any],
        optim_cfg: str,
        data_cfg: Dict[str, Any],
        args: argparse.Namespace,
    ) -> None:
        """Objective optimizer.

        Args:
            model: torch model to validate and optimize params.
            device: a torch device.
            cfg: config to create validator.
            optim_cfg: optimizer config file path.
            data_cfg: dataset config.
            args: system arguments.
        """
        super().__init__(cfg, optim_cfg, data_cfg, args)
        self.model = model
        self.device = device
        self.model_params = count_param(self.model)

        # original yolov5x model
        self.baseline_model: nn.Module = load_model_from_wandb(
            "j-marple/AYolov2/1gxaqgk4"
        ).to(device).eval()
        self.baseline_n_params = count_param(self.baseline_model)

        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma

        self.baesline_map50: float
        self.baseline_t: float

        # for optimization with json
        self.command = f"python3 val2.py --weights {self.args.weights} --data {data_cfg['val_path']} --device {self.args.device} --batch-size {self.args.batch_size} --no_coco --n-skip {self.args.n_skip} --json-path {self.args.json_path}"

        if args.model_cfg:
            self.command += " --model-cfg {self.args.model_cfg}"

        if not args.rect:
            self.command += " --no-rect"

        if args.single_cls:
            self.command += " --single-cls"

        if args.half:
            self.command += " --half"

    def get_param(self, trial: optuna.trial.Trial) -> None:
        """Get hyper params."""
        self.cfg["train"]["img_size"] = trial.suggest_int(
            "img_width", **self.optim_cfg["img_width"]
        )
        self.cfg["hyper_params"]["conf_t"] = trial.suggest_float(
            "conf_thr", **self.optim_cfg["conf_thr"]
        )
        self.cfg["hyper_params"]["iou_t"] = trial.suggest_float(
            "iou_thr", **self.optim_cfg["iou_thr"]
        )

    def test_baseline(self) -> None:
        """Validate baseline model network."""
        baseline_dataset = LoadImagesAndLabels(
            self.data_cfg["val_path"],
            img_size=640,
            batch_size=self.cfg["train"]["batch_size"],
            rect=self.cfg["train"]["rect"],
            label_type="labels",
            cache_images=None,
            single_cls=False,
            stride=int(max(self.baseline_model.stride)),  # type: ignore
            pad=0.5,
            n_skip=self.args.n_skip,
            prefix="[val]",
            yolo_augmentation=None,
            augmentation=None,
            preprocess=None,
        )

        baseline_dataloader = torch.utils.data.DataLoader(
            baseline_dataset,
            batch_size=self.cfg["train"]["batch_size"],
            num_workers=min(os.cpu_count(), self.cfg["train"]["batch_size"]),  # type: ignore
            pin_memory=True,
            collate_fn=LoadImagesAndLabels.collate_fn,
        )

        baseline_validator = YoloValidator(
            self.baseline_model,
            baseline_dataloader,
            self.device,
            self.cfg,
            # compute_loss=isinstance(self.baseline_model, YOLOModel)
            # and hasattr(model, "hyp"),
            log_dir=None,
            incremental_log_dir=False,
            export=False,
        )
        LOGGER.info(
            f"Validating baseline model (n_parm: {self.baseline_n_params:,d}) ..."
        )
        t0 = time.monotonic()
        baseline_result = baseline_validator.validation(verbose=False)
        self.baseline_t = time.monotonic() - t0
        self.baseline_map50 = baseline_result[0][2]

    def calc_objective_fn(self, time: float, map50: float) -> float:
        """Calculate objective function.

        Objective function based on AIGC.

        Args:
            time: the time which the model(target) tooks.
            map50: the mAP@.5 result of the model(target).

        Returns:
            the result based on AIGC metric.
        """
        param_score = self.alpha * (self.baseline_n_params / self.model_params)
        time_score = self.beta * (self.baseline_t / time)
        map50_score = self.gamma * (map50 / self.baseline_map50)
        return param_score + time_score + map50_score

    def _run_with_model(self, trial: optuna.trial.Trial) -> float:
        """Run model and compare with baseline model.

        Calculate the AIGC metirc, and optimize image size, IoU(intersection of union) threshold, confidence threshold.

        Args:
            trial: an optuna trial.

        Returns:
            a result of AIGC metric.
        """
        if self.args.weights.endswith(".pt"):
            log_dir = os.path.join(self.args.weights[:-3])
        else:
            log_dir = os.path.join("wandb", "downloads", self.args.weights)

        val_dataset = LoadImagesAndLabels(
            self.data_cfg["val_path"],
            img_size=self.cfg["train"]["img_size"],
            batch_size=self.cfg["train"]["batch_size"],
            rect=self.cfg["train"]["rect"],
            label_type="labels",
            cache_images=None,
            single_cls=False,
            stride=int(max(self.model.stride)),  # type: ignore
            pad=0.5,
            n_skip=self.args.n_skip,
            prefix="[val]",
            yolo_augmentation=None,
            augmentation=None,
            preprocess=None,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.cfg["train"]["batch_size"],
            num_workers=min(os.cpu_count(), self.cfg["train"]["batch_size"]),  # type: ignore
            pin_memory=True,
            collate_fn=LoadImagesAndLabels.collate_fn,
        )

        validator = YoloValidator(
            self.model,
            val_loader,
            self.device,
            self.cfg,
            # compute_loss=isinstance(self.model, YOLOModel)
            # and hasattr(self.model, "hyp"),
            half=self.args.half,
            log_dir=log_dir,
            incremental_log_dir=True,
            export=True,
        )

        LOGGER.info(
            f"{trial.number}th trial, Running validation (conf_thr: {self.cfg['hyper_params']['conf_t']}, iou_thr: {self.cfg['hyper_params']['iou_t']}, img_size: {self.cfg['train']['img_size']})"
        )

        t0 = time.monotonic()
        val_result = validator.validation(verbose=False)
        time_took = time.monotonic() - t0
        map50_result = val_result[0][2]

        LOGGER.info(
            f"   |----- Finished with mAP50: {map50_result}, time_took: {time_took:.3f}s"
        )

        trial.set_user_attr("map50", map50_result)
        trial.set_user_attr("time_took", time_took)

        objective_score = self.calc_objective_fn(time_took, map50_result)

        if map50_result < ObjectiveValidator.BASE_mAP50:
            return map50_result * 0.1  # Punish lower mAP50 result than base mAP50.
        else:
            return objective_score

    def _run_with_json(self, trial: optuna.trial.Trial) -> float:
        """Calculate results for optimize with json.

        Args:
            trial: an optuna trial.

        Returns:
            a result of AIGC metric.
        """
        command = f" -iw {self.cfg['train']['img_size']} -ct {self.cfg['hyper_params']['conf_t']} -it {self.cfg['hyper_params']['iou_t']}"
        command = self.command + command

        LOGGER.info(f"Run: {command}")
        t0 = time.monotonic()
        os.system(command)
        time_took = time.monotonic() - t0

        anno = COCO(ObjectiveValidator.gt_path)
        try:
            pred = anno.loadRes(self.args.json_path)
        except IndexError:
            # Because of an empty json when no objects are detected.
            return 0
        cocoeval = COCOeval(anno, pred, "bbox")

        # Finding which images has been used.
        val_dataset = LoadImagesAndLabels(
            self.data_cfg["val_path"],
            img_size=self.cfg["train"]["img_size"],
            batch_size=self.cfg["train"]["batch_size"],
            rect=self.cfg["train"]["rect"],
            label_type="labels",
            cache_images=None,
            single_cls=False,
            stride=int(max(self.model.stride)),  # type: ignore
            pad=0.5,
            n_skip=self.args.n_skip,
            prefix="[val]",
            yolo_augmentation=None,
            augmentation=None,
            preprocess=None,
        )
        cocoeval.params.imgIds = [
            int(Path(path).stem) for path in val_dataset.img_files
        ]

        cocoeval.evaluate()
        cocoeval.accumulate()
        cocoeval.summarize()

        map50_result = cocoeval.stats[1]

        trial.set_user_attr("map50", map50_result)
        trial.set_user_attr("time_took", time_took)

        objective_score = self.calc_objective_fn(time_took, map50_result)

        if map50_result < ObjectiveValidator.BASE_mAP50:
            return map50_result * 0.1
        else:
            return objective_score

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Run model and compare with baseline model.

        Calculate the AIGC metirc, and optimize image size, IoU(intersection of union) threshold, confidence threshold.

        Args:
            trial: an optuna trial.

        Returns:
            a result of AIGC metric.
        """
        assert hasattr(self, "baseline_t") and hasattr(
            self, "baseline_map50"
        ), "Baseline information is missing. Required action: either baseline_t and baseline_map50 are set or self.test_baseline() has been called."

        LOGGER.info(f"Starting {trial.number}th trial.")
        if len(trial.study.best_trials):
            LOGGER.info("Showing previous best trials ...")

            for i, best_trial in enumerate(trial.study.best_trials):
                LOGGER.info(
                    f"[{(i+1):02d}:Best] {best_trial.number}th trial, Params: {best_trial.params}, Score: {best_trial.values}, Attr: {best_trial.user_attrs}"
                )

        self.get_param(trial)
        if not self.args.run_json:
            return self._run_with_model(trial)

        else:
            return self._run_with_json(trial)
