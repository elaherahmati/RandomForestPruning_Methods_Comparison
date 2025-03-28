# 🎯 Fairness-Aware Pruning Strategies for Random Forests

This project explores how different pruning strategies affect the fairness and performance of Random Forest classifiers on ACS datasets. It supports multiple pruning techniques and fairness metrics and is designed for reproducible experimentation.

## ✅ Features

- Supports six pruning methods:
  - `RandomPruningClassifier`
  - `GreedyPruningClassifier`
  - `MIQPPruningClassifier`
  - `ClusterPruningClassifier`
  - `RankPruningClassifier`
  - `BaseModel` (unpruned baseline)

- Computes key fairness metrics:
  - Demographic Parity, Equalized Odds, Predictive Parity
  - Tracks performance: Accuracy
  - Works with pre-saved train/test splits and saves experiment results to Excel




# Experiment Runner

This repository contains a script for running experiments with different model types and pruning levels on the ACS dataset. It is designed to work with SLURM job scheduling.

## ⚙️ How to Run

```bash
python run_experiment.py \
  --modelname MIQPPruningClassifier \
  --prune_level 20 \
  --sensitive_feature SEX \
  --task ACSIncome



Requirements
Install dependencies:
pip install -r requirements.txt


Outputs
results/
└── {Model}/{Task}/n_repeats_{n}/nprune_{k}/
    ├── raw/{model}_raw.xlsx         # Raw fold results
    └── {model}.xlsx 



Research Context
This project supports the thesis:

"Pruning Strategies in Random Forests: The Interplay Between Model Compression and Fairness"
