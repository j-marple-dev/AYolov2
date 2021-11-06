# AYOLOv2
This repository is based on [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5).

# Environment setup
## Prerequisites
- Anaconda
- Docker (Optional)
- Clone this repository
  ```bash
  $ https://github.com/j-marple-dev/AYolov2.git
  $ cd AYolov2
  ```

## Installation
### Using conda
```bash
$ conda env create -f environment.yml
```

### Using Docker
#### 1. Docker Build
```bash
$ ./run_docker.sh build
```

#### 1.2 Docker Run (WIP)
```bash
```

# Applying SWA
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
