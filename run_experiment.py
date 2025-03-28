"""
run_experiment.py

A command-line script for running experiments with different models and pruning levels
on the ACS dataset using SLURM job scheduling. This script accepts command-line arguments
for model name, prune level, and dataset task.

Usage (example):
    python run_experiment.py --modelname ClusterPruningClassifier --prune_level 20 --task ACSIncome

Arguments:
    --modelname     Name of the model (e.g., ClusterPruningClassifier)
    --prune_level   Prune level (int, e.g., 20)
    --task          Task name from the ACS dataset
"""

#import necessary libraries

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio)
from fairlearn.metrics import (
    true_positive_rate,
    true_negative_rate,
    false_positive_rate,
    false_negative_rate)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PyPruning.MIQPPruningClassifier import MIQPPruningClassifier
from PyPruning.RandomPruningClassifier import RandomPruningClassifier
from PyPruning.GreedyPruningClassifier import GreedyPruningClassifier,complementariness
from PyPruning.ClusterPruningClassifier import ClusterPruningClassifier,cluster_accuracy
from PyPruning.RankPruningClassifier import RankPruningClassifier,individual_neg_auc
from PyPruning.PruningClassifier import PruningClassifier
from tqdm import tqdm
import scipy.stats as stats
import itertools
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from typing import List, Optional, Dict
from folktables import ACSDataSource, ACSIncome,ACSTravelTime


#Define Fairness Metric functions
def predictive_parity(y_test, y_pred, sensitive_feature):
    unique_groups = np.unique(sensitive_feature)
    if len(unique_groups) == 2:
        group_0, group_1 = unique_groups
        ppv_0 = precision_score(y_test[sensitive_feature == group_0], y_pred[sensitive_feature == group_0])
        ppv_1 = precision_score(y_test[sensitive_feature == group_1], y_pred[sensitive_feature == group_1])
        return abs(ppv_1 - ppv_0)
    return np.nan  # Cannot compute if more than 2 groups


def ensure_directory_exists(directory_path: str) -> None:
    """Ensures the given directory exists."""
    os.makedirs(directory_path, exist_ok=True)

def convert_to_numeric(input_list):
    unique_items = sorted(set(input_list))
    mapping = {item: str(index) for index, item in enumerate(unique_items)}
    converted_list = [mapping[item] for item in input_list]
    return converted_list


import argparse
# Argument Parser Setup
parser = argparse.ArgumentParser(description="Run Experiment with Different Models and Prune Levels")
parser.add_argument("--modelname", type=str, required=True, help="Model name")
parser.add_argument("--prune_level", type=int, required=True, help="Prune level")
parser.add_argument("--task", type=str, required=True, help="ACS dataset task")
args = parser.parse_args()


selected_task = args.task  
sensitive_feature_name = "SEX"
class_label = 'Class'
n_repeats = 10
n_splits = 5
n_prune = args.prune_level
model_name = args.modelname
n_base = 100

pruning_models = {
    'RandomPruningClassifier': RandomPruningClassifier,
    'GreedyPruningClassifier': GreedyPruningClassifier,
    'ClusterPruningClassifier': ClusterPruningClassifier,
    'RankPruningClassifier': RankPruningClassifier,
    'MIQPPruningClassifier': MIQPPruningClassifier,
    'BaseModel': RankPruningClassifier  # This is base model
}

# Check if model_name exists
if model_name not in pruning_models:
    raise ValueError(f"Model {model_name} not found!")

# Path to saved splits
split_file = f"train_test_splits_TrainTest_{selected_task}.pkl"

# Load train-test splits
with open(split_file, "rb") as f:
    train_val_splits = pickle.load(f)


sensitive_feature_classes = list(set(
    tuple(val[sensitive_feature_name].unique().tolist())  # Convert NumPy array to tuple
    for repeat in train_val_splits
    for fold in train_val_splits[repeat]
    for val in [train_val_splits[repeat][fold][0], train_val_splits[repeat][fold][2]]  # X_train = [0], X_test = [2]
))

sensitive_feature_classes = convert_to_numeric(sensitive_feature_classes)


# Extract an example `X_train` from any repeat & fold
example_X_train = next(iter(train_val_splits.values()))  # Get first repeat
example_X_train = next(iter(example_X_train.values()))  # Get first fold
X_train = example_X_train[0]  # X_train is at index 0
sensitive_feature_index = X_train.columns.get_loc(sensitive_feature_name)
model_class = pruning_models[model_name]
experiment_name = f'{model_name}-{sensitive_feature_name}'

model = RandomForestClassifier(n_estimators=n_base, max_depth=3, min_samples_split=10, bootstrap=True,random_state=42)



accuracy_list=[]
predictive_parity_diff_list,dp_diff_list, eo_diff_list = [], [], [], [],[]


for repeat in range(n_repeats):
    for fold  in range(n_splits):

        Xtrain, y_train, Xtest, y_test = train_val_splits[repeat][fold]
        sensitive_feature = Xtest[sensitive_feature_name]
        model.fit(Xtrain, y_train)
       
        #prune the model
        if model_name == 'BaseModel':
            y_pred = model.predict(Xtest)
            pruned_pred_train = model.predict(Xtrain)
                       
        elif model_name == "GreedyPruningClassifier":

            pruned_model = model_class(n_estimators=n_prune, metric = complementariness)
            pruned_model.prune(Xtrain, y_train, model.estimators_)
            y_pred = pruned_model.predict(Xtest)
            pruned_pred_train = pruned_model.predict(Xtrain)

        elif model_name == "MIQPPruningClassifier":
            pruned_model = model_class(n_estimators=n_prune)
            pruned_model.prune(Xtrain, y_train, model.estimators_)
            y_pred = pruned_model.predict(Xtest)
            pruned_pred_train = pruned_model.predict(Xtrain)

        elif model_name == "ClusterPruningClassifier":
            pruned_model = model_class(n_estimators=n_prune, select_estimators = cluster_accuracy, cluster_mode = "probabilities")
            pruned_model.prune(Xtrain, y_train, model.estimators_)
            y_pred = pruned_model.predict(Xtest)
            pruned_pred_train = pruned_model.predict(Xtrain)
            
        elif model_name == "RankPruningClassifier":
            pruned_model = model_class(n_estimators=n_prune, metric = individual_neg_auc)
            pruned_model.prune(Xtrain, y_train, model.estimators_)
            y_pred = pruned_model.predict(Xtest)
            pruned_pred_train = pruned_model.predict(Xtrain)

        else:
            pruned_model = model_class(n_estimators=n_prune)
            pruned_model.prune(Xtrain, y_train, model.estimators_)
            y_pred = pruned_model.predict(Xtest)
            pruned_pred_train = pruned_model.predict(Xtrain)
            

        dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_feature)
        eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_feature)
        accuracy = accuracy_score(y_test, y_pred)
        predictive_parity_diff = predictive_parity(y_test, y_pred, sensitive_feature)
        dp_diff_list.append(dp_diff)
        eo_diff_list.append(eo_diff)
        accuracy_list.append(accuracy)
        predictive_parity_diff_list.append(predictive_parity_diff)
        

    final_result = {'Demographic Parity Difference': dp_diff_list,
        'Equalized Odds Difference': eo_diff_list,
        'Accuracy': accuracy_list,
        'n_prune': n_prune,
        'Predictive Parity Difference' : predictive_parity_diff_list}

    base_results_dir = os.getenv("BASE_RESULTS_DIR")
    results_dir = os.path.join(base_results_dir, f"Task_{selected_task}", f"n_repeats_{n_repeats}", f"nprune_{n_prune}")
    os.makedirs(results_dir, exist_ok=True)
    ensure_directory_exists(results_dir)
    raw_dir = os.path.join(results_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    ensure_directory_exists(raw_dir)
    final_result_df = pd.DataFrame(final_result)
    file_path = os.path.join(raw_dir, f'{experiment_name}_raw.xlsx')
    final_result_df.to_excel(file_path, index=False)

print(f"all result saved to:{results_dir}")

