#!/bin/bash

export nnUNet_raw="/Users/atatc/PycharmProjects/ML-Quiz/raw"
export nnUNet_preprocessed="/Users/atatc/PycharmProjects/ML-Quiz/preprocessed"
export nnUNet_results="/Users/atatc/PycharmProjects/ML-Quiz/weights"
nnUNetv2_find_best_configuration DATASET_NAME_OR_ID -c CONFIGURATIONS