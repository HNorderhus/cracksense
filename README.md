# DeepLabV3Resnet101 Crack Segmentation Project

![inferenced crack](imgs/608_result_ignore.png)

## Overview

This repository hosts the Master's thesis project "CrackSense: A Pruned Neural Network Approach for Crack Detection on Embedded Devices." 

The project emphasizes training DeepLabV3-Resnet101 models on the S2DS dataset for crack detection, with a unique focus on model pruning—both iteratively and in one shot—based on various mathematical importance criteria such as L1 Norm, L2 Norm, and Taylor Expansion. Additionally, the repository includes utilities for model inference, performance evaluation, and detailed class-wise metrics exportation to Excel, facilitating comprehensive analysis and application in embedded device environments.

## How to get started

Clone the repository and install the environment.

```
conda env create -f environment.yml
```

## Dataset structure

The dataset is expected to be structured in the following way:

```
├── test
│   ├── Images
│   │   ├── 678.png
│   │   ...
│   ├── Labels_grayscale
│   │   ├── 678.png
│   │   ...
│   └── Labels_RGB
│       ├── 678.png
│       ...
├── train
│   ├── Images
│   │   ├── 000.png
│   │   ...
│   ├── Labels_grayscale
│   │   ├── 000.png
│   │   ...
│   └── Labels_RGB
│       ├── 000.png
│       ...
└── val
    ├── Images
    │   ├── 563.png
    │   ...
    ├── Labels_grayscale
    │   ├── 563.png
    │   ...
    └── Labels_RGB
        ├── 563.png
        ...
```

## How to use


