# Understanding the Privacy–Fairness Tradeoff  
A Study of the Impact of Differential Privacy on Algorithmic Fairness

## Overview

This project studies the relationship between differential privacy and algorithmic fairness in machine learning systems. Specifically, we evaluate how privacy-preserving training affects model accuracy and fairness across different sensitive attributes.

We trained logistic regression models under both non-private and differentially private (DP) settings and compared their behavior across multiple privacy budgets. Fairness was evaluated using demographic parity difference (DPD) and equalized odds difference (EOD).

This work was developed as part of **CMU 19-605: Engineering Privacy in Software**.



## Project Goals

The primary objectives of this project were:

- Implement classification models with and without differential privacy
- Evaluate the impact of privacy on prediction accuracy
- Measure fairness across demographic groups
- Compare fairness behavior across multiple sensitive attributes
- Analyze how privacy strength (ε) influences fairness outcomes




### Directory Description

#### `data/`

Contains the Adult Income dataset files obtained from the UCI Machine Learning Repository.

- `adult.data` — training data
- `adult.test` — test data
- `adult.names` — attribute metadata



#### `scripts/`

Contains scripts for the **Adult Income dataset** pipeline.

- `prepare_adult.py`  
  Preprocesses Adult dataset:
  - Handles categorical encoding
  - Normalizes features
  - Splits training/testing data
  - Identifies sensitive attributes (gender and race)

- `train_baseline.py`  
  Trains logistic regression model **without differential privacy**.

- `train_dp.py`  
  Trains logistic regression model using **DP-SGD via Opacus**.



#### `celeb_scripts/`

Contains scripts used for the **CelebA dataset** experiments.

These scripts follow the same pipeline structure as the Adult dataset workflow.



#### `celeb_outputs/`

Stores output results and generated evaluation artifacts from CelebA experiments.

