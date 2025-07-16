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
from imblearn.ensemble import EasyEnsembleClassifier


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
     #   "SVM": SVC(probability=True, random_state=42),
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
        
        print(f"Training for feature set: {feature_set_name}")
        print(features)
        
        X_subset = preprocessed_data[features]

        for model_name, model in models.items():
            
            print(f"Training model: {model_name}")
            scores = cross_validate(
                model, X_subset, y, cv=skf, scoring=scoring, n_jobs=1
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
def evaluate_models_with_resampling(preprocessed_data: pd.DataFrame, selected_feature_vectors: dict,) -> pd.DataFrame:

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_feature_sets = {
        "CatBoost": "lgbm_top_25",
        "KNN": "corr_target_top_10",
        "RandomForest": "info_gain_top_10",
        "XGBoost": "lgbm_top_20",
     #   "SVM": "intersection_2plus_25",
    }

    resamplers = {
    #    "None": None,
        "SMOTE": SMOTE(random_state=42),
        "RandomOverSampler": RandomOverSampler(random_state=42),
        "RandomUnderSampler": RandomUnderSampler(random_state=42),
    }

    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        #"SVM": SVC(probability=True, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    }

    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": make_scorer(average_precision_score),
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "accuracy": "accuracy"
    }

    results = []
    
    #model = models[model_name]
    
    for model_name, feature_set_name in best_feature_sets.items():
        print(f"\nEvaluating model: {model_name} on feature set: {feature_set_name}")
        model = models[model_name]
        selected_features = selected_feature_vectors[feature_set_name]
        
        selected_features_with_label = selected_features + ["Class"]
        Xy = preprocessed_data[selected_features_with_label]
        X = Xy.drop(columns=["Class"])
        y = Xy["Class"]

        for resampler_name, resampler in resamplers.items():
            print(f"  → With resampler: {resampler_name}")
            pipeline = ImbPipeline([
                ("resampler", resampler),
                ("classifier", model),
            ])

            start_time = time.time()
            scores = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=-1)
            end_time = time.time()

            results.append({
                "model": model_name,
                "feature_set": feature_set_name,
                "resampler": resampler_name,
                "recall": scores["test_recall"].mean(),
                "precision": scores["test_precision"].mean(),
                "f1_score": scores["test_f1"].mean(),
                "roc_auc": scores["test_roc_auc"].mean(),
                "pr_auc": scores["test_pr_auc"].mean(),
                "accuracy": scores["test_accuracy"].mean(),
                "training_time": end_time - start_time,
                "n_features": X.shape[1],
            })

    return pd.DataFrame(results)

def evaluate_models_with_resampling_svm(preprocessed_data: pd.DataFrame, selected_feature_vectors: dict,) -> pd.DataFrame:

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_feature_sets = {
        "SVM": "intersection_2plus_25",
    }

    resamplers = {
    #    "None": None,
        #"SMOTE": SMOTE(random_state=42),
        "RandomOverSampler": RandomOverSampler(random_state=42),
        "RandomUnderSampler": RandomUnderSampler(random_state=42),
        "SMOTE": SMOTE(random_state=42)
    }

    models = {
        "SVM": SVC(kernel="linear", random_state=42),  # removed probability=True
    }

    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": make_scorer(average_precision_score),
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "accuracy": "accuracy"
    }

    results = []
        
    for model_name, feature_set_name in best_feature_sets.items():
        print(f"\nEvaluating model: {model_name} on feature set: {feature_set_name}")
        model = models[model_name]
        selected_features = selected_feature_vectors[feature_set_name]
        
        selected_features_with_label = selected_features + ["Class"]
        Xy = preprocessed_data[selected_features_with_label]
        X = Xy.drop(columns=["Class"])
        y = Xy["Class"]

        for resampler_name, resampler in resamplers.items():
            print(f"  → With resampler: {resampler_name}")
            pipeline = ImbPipeline([
                ("resampler", resampler),
                ("classifier", model),
            ])

            start_time = time.time()
            scores = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=1)
            end_time = time.time()

            results.append({
                "model": model_name,
                "feature_set": feature_set_name,
                "resampler": resampler_name,
                "recall": scores["test_recall"].mean(),
                "precision": scores["test_precision"].mean(),
                "f1_score": scores["test_f1"].mean(),
                "roc_auc": scores["test_roc_auc"].mean(),
                "pr_auc": scores["test_pr_auc"].mean(),
                "accuracy": scores["test_accuracy"].mean(),
                "training_time": end_time - start_time,
                "n_features": X.shape[1],
            })

    return pd.DataFrame(results) 


def evaluate_models_with_resampling_all_features(preprocessed_data: pd.DataFrame) -> pd.DataFrame:

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
        #"SVM": SVC(probability=True, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    }

    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": make_scorer(average_precision_score),
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "accuracy": "accuracy"
    }

    results = []
    
    #model = models[model_name]
    
    for model_name, model in models.items():
        print(f"\nEvaluating model: {model_name}")
        model = models[model_name]

        X = preprocessed_data.drop(columns=["Class"])
        y = preprocessed_data["Class"]

        for resampler_name, resampler in resamplers.items():
            print(f"  → With resampler: {resampler_name}")
            pipeline = ImbPipeline([
                ("resampler", resampler),
                ("classifier", model),
            ])

            start_time = time.time()
            scores = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=-1)
            end_time = time.time()

            results.append({
                "model": model_name,
                "feature_set": "all_features",
                "resampler": resampler_name,
                "recall": scores["test_recall"].mean(),
                "precision": scores["test_precision"].mean(),
                "f1_score": scores["test_f1"].mean(),
                "roc_auc": scores["test_roc_auc"].mean(),
                "pr_auc": scores["test_pr_auc"].mean(),
                "accuracy": scores["test_accuracy"].mean(),
                "training_time": end_time - start_time,
                "n_features": X.shape[1],
            })

    return pd.DataFrame(results) 

# Easy Ensamble

def generate_easy_ensemble_sets(X, y, n_subsets=10, random_state=42):
    rus = RandomUnderSampler(random_state=random_state)
    rng = np.random.RandomState(random_state)

    sets = []
    for _ in range(n_subsets):
        X_res, y_res = rus.fit_resample(X, y)
        sets.append((X_res, y_res))
        # To get different subset every time
        rus.random_state = rng.randint(0, 99999)
    return sets

def train_on_custom_easy_ensemble(model, X_resampled_sets, y_resampled_sets):
    results = []
    for i, (X_resampled, y_resampled) in enumerate(zip(X_resampled_sets, y_resampled_sets)):
        
        scoring = {
            "roc_auc": "roc_auc",
            "pr_auc": make_scorer(average_precision_score),
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
            "accuracy": "accuracy"
        }
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
        print(f"  → Training on EasyEnsemble subset {i+1}")
        scores = cross_validate(model, X_resampled, y_resampled, cv=skf, scoring=scoring, n_jobs=-1)
        results.append({metric: np.mean(scores[f'test_{metric}']) for metric in scoring})
    return results


def evaluate_models_with_custom_easy_ensemble(preprocessed_data: pd.DataFrame, 
                                       selected_feature_vectors: dict,
                                       n_subsets: int = 10,
                                       random_state: int = 42) -> pd.DataFrame:
    
    best_feature_sets = {
        "CatBoost": "lgbm_top_25",
        "KNN": "corr_target_top_10",
        "RandomForest": "info_gain_top_10",
        "XGBoost": "lgbm_top_20",
     #   "SVM": "intersection_2plus_25",
    }
    
    models = {
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=random_state),
        #"SVM": SVC(probability=True, random_state=random_state),
    }
    
    all_results = []
    
    for model_name, feature_set_name in best_feature_sets.items():
        print(f"\nEvaluating model: {model_name} on feature set: {feature_set_name}")
        model = models[model_name]
        selected_features = selected_feature_vectors[feature_set_name]
        
        selected_features_with_label = selected_features + ["Class"]
        Xy = preprocessed_data[selected_features_with_label]
        X = Xy.drop(columns=["Class"])
        y = Xy["Class"]
        
        # Generate resampled subsets
        Xy_sets = generate_easy_ensemble_sets(X, y, n_subsets=n_subsets, random_state=random_state)
        X_resampled_sets, y_resampled_sets = zip(*Xy_sets)
        
        # Train on these subsets
        start_time = time.time()
        results = train_on_custom_easy_ensemble(model, X_resampled_sets, y_resampled_sets)
        total_time = time.time() - start_time
        
        # Aggregate metrics
        metrics = results[0].keys()
        aggregated = {metric: np.mean([res[metric] for res in results]) for metric in metrics}
        
        print(f"Aggregated results for {model_name} on {feature_set_name}: {aggregated}")
        
        all_results.append({
            "model": model_name,
            "feature_set": feature_set_name,
            "resampler": "CustomEasyEnsemble",
            **aggregated,
            "training_time": total_time,
            "n_features": len(selected_features) - 1,  # minus Class column
            "n_subsets": n_subsets,
        })
        
    return pd.DataFrame(all_results)

def evaluate_models_with_custom_easy_ensemble_all_features(preprocessed_data: pd.DataFrame, 
                                       n_subsets: int = 10,
                                       random_state: int = 42) -> pd.DataFrame:
    
    models = {
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=random_state),
        #"SVM": SVC(probability=True, random_state=random_state),
    }
    
    all_results = []
    
    for model_name, model in models.items():
        print(f"\nEvaluating model: {model_name}")
        model = models[model_name]
        
        X = preprocessed_data.drop(columns=["Class"])
        y = preprocessed_data["Class"]
        
        # Generate resampled subsets
        Xy_sets = generate_easy_ensemble_sets(X, y, n_subsets=n_subsets, random_state=random_state)
        X_resampled_sets, y_resampled_sets = zip(*Xy_sets)
        
        # Train on these subsets
        start_time = time.time()
        results = train_on_custom_easy_ensemble(model, X_resampled_sets, y_resampled_sets)
        total_time = time.time() - start_time
        
        # Aggregate metrics
        metrics = results[0].keys()
        aggregated = {metric: np.mean([res[metric] for res in results]) for metric in metrics}
        
        print(f"Aggregated results for {model_name} : {aggregated}")
        
        all_results.append({
            "model": model_name,
            "feature_set": "all_features",
            "resampler": "CustomEasyEnsemble",
            **aggregated,
            "training_time": total_time,
            "n_features": X.shape[1],
            "n_subsets": n_subsets,
        })
        
    return pd.DataFrame(all_results)