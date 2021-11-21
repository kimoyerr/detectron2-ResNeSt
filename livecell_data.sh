#!/bin/sh

aws s3 sync s3://livecell-dataset ~/livecell-dataset
unzip ~/livecell-dataset/LIVECell_dataset_2021/images.zip -d ~/livecell-dataset/LIVECell_dataset_2021/

