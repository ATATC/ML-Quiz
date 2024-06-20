# ML-Quiz

This quiz aims to automate the segmentation of blood cells using both a CNN-based and a Transformer-based neural network
with the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework.

## Methodology

In this task, a revision of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) is proposed where it is modified to suit
the task. The repository is available [here](https://github.com/ATATC/ML-Quiz-nnUNet) on GitHub. In this fork of the
framework, a new trainer for Swin-Unet is included with all arguments set to default. Each model is trained with 1000
epochs using a single fold `all`.

## Results

The results from the two models are compared to cross validate.

The scores are formatted as the following:
NAME: DICE_SIMILARITY_COEFFICIENT_SCORE (ERROR_BAR)

### Internal

Whole Cell DC: 98.10 (-23.56/+1.56)
Nucleus DC: 96.30 (-10.50/+3.23)

### External

Whole Cell DC: 98.15 (-9.18/+1.38)
Nucleus DC: 97.22 (-23.68/+2.15)

## Outlook

Due to the time efficiency, 5-fold cross validation is not enabled during training, which can be done in the future.