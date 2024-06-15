#!/bin/bash

export nnUNet_raw="/Users/atatc/PycharmProjects/ML-Quiz/raw"
export nnUNet_preprocessed="/Users/atatc/PycharmProjects/ML-Quiz/preprocessed"
export nnUNet_results="/Users/atatc/PycharmProjects/ML-Quiz/weights"
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity