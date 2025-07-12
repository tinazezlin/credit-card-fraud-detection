from typing import Tuple, Dict
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import get_scorer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
#from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import time
import logging

import warnings

warnings.filterwarnings("ignore")  # To keep output clean

def evaluate_baseline_models_across_feature_sets(
    preprocessed_data: pd.DataFrame,
    feature_vectors: Dict[str, list],
) -> pd.DataFrame:
    """
    Evaluate multiple models across different feature vectors using cross-validation.

    Args:
        preprocessed_data: Full preprocessed dataset including 'Class'.
        feature_vectors: Dictionary of feature names per feature selection method.

    Returns:
        DataFrame summarizing performance of each model on each feature set.
    """
    results = []
    y = preprocessed_data["Class"]
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
        "SVM": SVC(probability=True, random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM": lgb.LGBMClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": make_scorer(average_precision_score),
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "accuracy": "accuracy"
    }

    for feature_set_name, features in feature_vectors.items():
        X_subset = preprocessed_data[features]

        for model_name, model in models.items():
            
            print(f"Training model: {model_name}")
            scores = cross_validate(
                model, X_subset, y, cv=skf, scoring=scoring, n_jobs=-1
            )

            results.append({
                "model": model_name,
                "feature_set": feature_set_name,
                "roc_auc": scores["test_roc_auc"].mean(),
                "pr_auc": scores["test_pr_auc"].mean(),
                "f1_score": scores["test_f1"].mean(),
                "precision": scores["test_precision"].mean(),
                "recall": scores["test_recall"].mean(),
                "accuracy": scores["test_accuracy"].mean(),
                "n_features": len(features),
            })

    return pd.DataFrame(results).sort_values(by=["model", "roc_auc"], ascending=[True, False])



def evaluate_baseline_models(filtered_features: pd.DataFrame,  log_file: str = "model_training.log") -> pd.DataFrame:
    
        # Set up logging inside the function with the provided log_file
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    X = filtered_features.drop(columns=["Class"])
    y = filtered_features["Class"]

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
        "SVM": SVC(probability=True, random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM": lgb.LGBMClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
    }

    scoring = {
        'recall': 'recall',
        'precision': 'precision',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    results = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        
        print(f"Training model: {name}")
        logging.info(f"Training model: {name}")
        
        start_time = time.time()
        # Stratified CV !!
        scores = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        end_time = time.time()
        
        training_time = end_time - start_time
        print(f"Done. Training time: {training_time:.2f} seconds")
        logging.info(f"Done. Training time for {name}: {training_time:.2f} seconds")

        results.append({
            'model': name,
            'recall': scores['test_recall'].mean(),
            'precision': scores['test_precision'].mean(),
            'f1': scores['test_f1'].mean(),
            'roc_auc': scores['test_roc_auc'].mean()
        })

    return pd.DataFrame(results).sort_values(by='recall', ascending=False)    

# We select top 5 models based on performance in default mode
def evaluate_models_with_resampling(filtered_features: pd.DataFrame, model_name: str) -> pd.DataFrame:
    X = filtered_features.drop(columns=["Class"])
    y = filtered_features["Class"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    resamplers = {
    #    "None": None,
        "SMOTE": SMOTE(random_state=42),
        "RandomOverSampler": RandomOverSampler(random_state=42),
        "RandomUnderSampler": RandomUnderSampler(random_state=42),
    }

    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    }

    scoring = {
        'recall': 'recall',
        'precision': 'precision',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    results = []
    
    model = models[model_name]
    
    
    for resampler_name, resampler in resamplers.items():
        print(f"Training {model_name} with resampling method {resampler_name}")
        pipeline = ImbPipeline([("resampler", resampler), ("classifier", model)])
        
        start_time = time.time()
        scores = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        end_time = time.time()
        
        training_time = end_time - start_time
        print(f"Done. Training time: {training_time:.2f} seconds")

        results.append({
            "model": model_name,
            "resampler": resampler_name,
            "recall": scores["test_recall"].mean(),
            "precision": scores["test_precision"].mean(),
            "f1": scores["test_f1"].mean(),
            "roc_auc": scores["test_roc_auc"].mean(),
            "training_time": training_time
        })

    return pd.DataFrame(results) 

"""     for model_name, model in models.items():
        for resampler_name, resampler in resamplers.items():
            
            print(f"Training model: {model_name} with resampling method {resampler_name}")
            if resampler:
                pipeline = ImbPipeline([("resampler", resampler), ("classifier", model)])
            else:
                pipeline = ImbPipeline([("classifier", model)])
                
            start_time = time.time()
            scores = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=-1)
            end_time = time.time()
        
            training_time = end_time - start_time
            print(f"Done. Training time: {training_time:.2f} seconds")
        
            results.append({
                "model": model_name,
                "resampler": resampler_name,
                "recall": scores["test_recall"].mean(),
                "precision": scores["test_precision"].mean(),
                "f1": scores["test_f1"].mean(),
                "roc_auc": scores["test_roc_auc"].mean(),
                "training_time": training_time
            })

    return pd.DataFrame(results) """