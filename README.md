# AYOLOv2
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
This repository is based on [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5).
The main goal of this repository is to rewrite object detection pipeline with better code structure for better portability and adapability to apply new experiment methods.
The object detection pipeline is based on [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5).

## What's inside
1. YOLOv5 based portable model (model built with [kindle](https://github.com/JeiKeiLim/kindle))
2. Model conversion (TorchScript, ONNX, TensorRT) support
3. Tensor decomposition model with pruning optimization
4. Stochastic Weight Averaging(SWA) support
5. Auto search for NMS parameter optimization
6. W&B support with model save and load functionality
7. Representative Learning (Experimental)
8. Distillation via soft teacher method (Experimental)
9. C++ inference (WIP)
10. AutoML - searching efficient architecture for the given dataset(incoming!)

# How to start
<details open>
  <summary>Install</summary>

  - [Conda virtual environment](https://docs.conda.io/en/latest/miniconda.html) or [docker](https://www.docker.com) is required to setup the environment
  ```bash
  git clone https://github.com/j-marple-dev/AYolov2.git
  cd AYolov2
  ./run_check.sh init
  # Equivalent to
  # conda env create -f environment.yml
  # pre-commit install --hook-type pre-commit --hook-type pre-push
  ```

  ### Using Docker
  #### Building a docker image
  ```bash
  ./run_docker.sh build
  # You can add build options
  # ./run_docker.sh build --no-cache
  ```

  #### Running the container
  This will mount current repository directory from local disk to docker image
  ```bash
  ./run_docker.sh run
  # You can add running options
  # ./run_docker.sh run -v $DATASET_PATH:/home/user/dataset
  ```

  #### Executing the last running container
  ```bash
  ./run_docker.sh exec
  ```
</details>
<details open>
  <summary>Train a model</summary>

- Example

  ```python
  python3 train.py --model $MODEL_CONFIG_PATH --data $DATA_CONFIG_PATH --cfg $TRAIN_CONFIG_PATH
  # i.e.
  # python3 train.py --model res/configs/model/yolov5s.yaml --data res/configs/data/coco.yaml --cfg res/configs/cfg/train_config.yaml
  # Logging and upload trained weights to W&B
  # python3 train.py --model res/configs/model/yolov5s.yaml --wlog --wlog_name yolov5s
  ```

  <details>
    <summary>Prepare dataset</summary>

    - Dataset config file

    ```yaml
    train_path: "DATASET_ROOT/images/train"
    val_path: "DATASET_ROOT/images/val"

    # Classes
    nc: 10  # number of classes
    dataset: "DATASET_NAME"
    names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light']  # class names
    ```

    - Dataset directory structure
      - One of `labels` or `segments` directory must exists.
      - Training label type(`labels` or `segments`) will be specified in training config.
      - images and labels or segments must have matching file name with .txt extension.

    ```bash
    DATASET_ROOT
    │
    ├── images
    │   ├── train
    │   └── val
    ├── labels
    │   ├── train
    │   └── val
    ├── segments
    │   ├── train
    │   └── val
    ```

  </details>

  <details>
    <summary>Training config</summary>

    ```yaml
  train:
    weights: ''  # Model weight path
    epochs: 300
    batch_size: 64
    image_size: 640   # Train, val image size
    rect: false  # Use rectangular training
    resume: false  # resume previous training
    validate_period: 1  # Run validation on every x epoch
    auto_anchor: true  # Check anchors and auto-fix anchors
    cache_image: null  # Use caching images. This should be either 'mem', 'disk', 'dynamic_mem', 'dynamic_disk'
    n_skip: 0  # skip image by n which reduces images to be used
    image_weights: false  # Use weighted image selection for training
    device: ""  # CUDA device. "" will use all GPUs. EX) '0,2' or 'cpu'
    multi_scale: false  # Use multi scaled training (+/- 50% image size)
    single_cls: false  # Train multi-class data as single-class
    sync_bn: false  # Use SyncBatchNorm, only available in DDP mode
    workers: 18  # Maximum number of dataloader workers
    linear_lr: false  # Use linear learning rate
    label_smoothing: 0.0  # Label smoothing epsilon
    freeze: 0  # Number of layers to freeze.
    save_period: -1  # Save checkpoint on every x epochs (disabled if < 1)
    log_dir: "exp"  # Default log root directory
    plot: true  # plot results or not.
    label_type: "segments"  # Label type to use ("labels", "segments")
    patience: 30  # for early stopping

  hyper_params:
    optimizer: 'SGD'
    optimizer_params:
      lr: 0.01  # lr0
      momentum: 0.937
      nesterov: true
    # optimizer: 'Adam'
    # optimizer_param:
    #   betas: [0.937, 0.999]

    lrf: 0.1  # final OneCycleLR learning rate (lr0 * lrf)
    momentum: 0.937  # SGD momentum/Adam beta1
    weight_decay: 0.0005  # optimizer weight decay 5e-4
    warmup_epochs: 3.0  # warmup epochs (fractions ok)
    warmup_momentum: 0.8  # warmup initial momentum
    warmup_bias_lr: 0.1  # warmup initial bias lr
    box: 0.05  # box loss gain
    cls: 0.5  # cls loss gain
    cls_pw: 1.0  # cls BCELoss positive_weight
    obj: 1.0  # obj loss gain (scale with pixels)
    obj_pw: 1.0  # obj BCELoss positive_weight
    conf_t: 0.1  # confidence threshold
    iou_t: 0.20  # IoU training threshold
    anchor_t: 4.0  # anchor-multiple threshold
    fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)

  yolo_augmentation:
    augment: true  # Use augmentation
    hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
    hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
    hsv_v: 0.4  # image HSV-Value augmentation (fraction)
    degrees: 0.0  # image rotation (+/- deg)
    translate: 0.1  # image translation (+/- fraction)
    scale: 0.5  # image scale (+/- gain)
    shear: 0.0  # image shear (+/- deg)
    perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
    mosaic: 1.0  # image mosaic (probability)
    mixup: 0.0  # image mixup (probability)
    copy_paste: 0.1  # segment copy-paste (probability)
    copy_paste2:  # config for copy-paste2
      p: 0.0  # copy-paste probability per object.
      n_img: 3  # Number of images to be used for copy-paste2
      area_thr: 200  # area threshold for copy paste (ex. 2 * 5 = 10)
      ioa_thr: 0.3  # IoA threshold for existing object and pasted object.
      scale_min: 0.35  # scale factor min value
      scale_max: 1.0  # scale factor max value

  augmentation:
    - policy:
        Blur: {p: 0.01}
        MedianBlur: {p: 0.01}
        ToGray: {p: 0.01}
        CLAHE: {p: 0.01}
        RandomBrightnessContrast: {p: 0.0}
        RandomGamma: {p: 0.0}
        ImageCompression: {quality_lower: 75, p: 0.0}
      prob: 1.0
    - policy:
        HorizontalFlip: {p: 0.5}
        VerticalFlip: {p: 0.0}
      prob: 1.0
    ```

  </details>

  <details>
    <summary>Model config</summary>

    - Model is defined by yaml file with [kindle](https://github.com/JeiKeiLim/kindle)
    - Please refer to https://github.com/JeiKeiLim/kindle

  </details>

  <details>
    <summary>Multi-GPU training</summary>

    - Please use torch.distributed.run module for multi-GPU Training

    ```bash
    python3 -m torch.distributed.run --nproc_per_node $N_GPU train.py --model $MODEL_CONFIG_PATH --data $DATA_CONFIG_PATH --cfg $TRAIN_CONFIG_PATH
    ```
      - N_GPU: Number of GPU to use

  </details>

</details>


<details open>
  <summary>Run a model</summary>

  - Validate from local weights
  ```bash
  python3 val.py --weights $WEIGHT_PATH --data-cfg $DATA_CONFIG_PATH
  ```
</details>


## Pretrained models
| Name  | W&B URL | img_size |    mAP<sup>val<br>0.5:0.95</sup>    |         mAP<sup>val<br>0.5</sup>         |    params|
|-------|---------------------------------------------------------------------------------------|---|----|----|----------|
|YOLOv5s|<sub>[j-marple/AYolov2/179awdd1](https://wandb.ai/j-marple/AYolov2/runs/179awdd1)</sub>|640|37.7|57.2| 7,235,389|
|YOLOv5m(WIP)|<sub>[j-marple/AYolov2/sybi3bnq](https://wandb.ai/j-marple/AYolov2/runs/sybi3bnq)</sub>|640|48.4|65.4|21,190,557|
|YOLOv5l(WIP)|<sub>[j-marple/AYolov2/1beuv3fd](https://wandb.ai/j-marple/AYolov2/runs/1beuv3fd)</sub>|640|51.3|67.8|46,563,709|
|YOLOv5x(WIP)|<sub>[j-marple/AYolov2/1gxaqgk4](https://wandb.ai/j-marple/AYolov2/runs/1gxaqgk4)</sub>|640|52.9|69.2|86,749,405|

</details>

# Applying SWA
<details>
  <summary> Stochastic Weight Averaging</summary>

  There are three steps to apply SWA (Stochastic Weight Averaging):

  1. Fine-tune pre-trained model
  2. Create SWA model
  3. Test SWA model

  ## 1. Fine-tune pre-trained model
  ### Example
  ```bash
  $ python train.py --model yolov5l_kindle.pt \
                    --data res/configs/data/coco.yaml \
                    --cfg res/configs/cfg/finetune.yaml \
                    --wlog --wlog_name yolov5l_swa \
                    --use_swa
  ```

  ## 2. Create SWA model
  ### Example
  ```bash
  $ python create_swa_model.py --model_dir exp/train/2021_1104_runs/weights \
                               --swa_model_name swa_best5.pt \
                               --best_num 5
  ```
  ### Usage
  ```bash
  $ python create_swa_model.py --help
  usage: create_swa_model.py [-h] --model_dir MODEL_DIR
                             [--swa_model_name SWA_MODEL_NAME]
                             [--best_num BEST_NUM]

  optional arguments:
    -h, --help            show this help message and exit
    --model_dir MODEL_DIR
                          directory of trained models to apply SWA (default: )
    --swa_model_name SWA_MODEL_NAME
                          file name of SWA model (default: swa.pt)
    --best_num BEST_NUM   the number of trained models to apply SWA (default: 5)
  ```

  ## 3. Test SWA model
  ### Example
  ```bash
  $ python val.py --weights exp/train/2021_1104_runs/weights/swa_best5.pt \
                  --model-cfg '' \
                  --data-cfg res/configs/data/coco.yaml \
                  --conf-t 0.1 --iou-t 0.2
  ```
</details>

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://limjk.ai"><img src="https://avatars.githubusercontent.com/u/10356193?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jongkuk Lim</b></sub></a><br /><a href="https://github.com/j-marple-dev/AYolov2/commits?author=JeiKeiLim" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/ulken94"><img src="https://avatars.githubusercontent.com/u/58245037?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Haneol Kim</b></sub></a><br /><a href="https://github.com/j-marple-dev/AYolov2/commits?author=ulken94" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/HSShin0"><img src="https://avatars.githubusercontent.com/u/44793742?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Hyungseok Shin</b></sub></a><br /><a href="https://github.com/j-marple-dev/AYolov2/commits?author=HSShin0" title="Code">💻</a></td>
    <td align="center"><a href="https://wooks.page/docs"><img src="https://avatars.githubusercontent.com/u/32764235?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Hyunwook Kim</b></sub></a><br /><a href="https://github.com/j-marple-dev/AYolov2/commits?author=wooks527" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
