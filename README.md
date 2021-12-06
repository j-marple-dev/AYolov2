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
