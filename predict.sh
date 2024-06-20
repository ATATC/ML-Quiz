#!/bin/bash

export nnUNet_raw="/Users/atatc/PycharmProjects/ML-Quiz/raw"
export nnUNet_preprocessed="/Users/atatc/PycharmProjects/ML-Quiz/preprocessed"
export nnUNet_results="/Users/atatc/PycharmProjects/ML-Quiz/weights"
nnUNetv2_predict -i imagesTs-Internal -o results-Internal -d 1 -c 2d -f all--save_probabilities -device cpu