#!/bin/bash

export nnUNet_raw="/Users/atatc/PycharmProjects/ML-Quiz/raw"
export nnUNet_preprocessed="/Users/atatc/PycharmProjects/ML-Quiz/preprocessed"
export nnUNet_results="/Users/atatc/PycharmProjects/ML-Quiz/weights"
nnUNetv2_train 1 2d 0 -device cpu