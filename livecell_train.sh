#!/bin/sh

python tools/train_net.py  --config-file ~/detectron2-ResNeSt/configs/LiveCell/model/anchor_based/livecell_config.yaml --eval-only OUTPUT_DIR ~/resnest_output MODEL.WEIGHTS http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_based/ALL/LIVECell_anchor_based_model.pth
