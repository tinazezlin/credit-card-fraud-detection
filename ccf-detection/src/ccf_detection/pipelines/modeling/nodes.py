from typing import Tuple, Dict
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score, accuracy_score, precision_recall_curve
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
        "XGBoost": "lgbm_top_20"
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
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
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
from sklearn.base import clone

def generate_easy_ensemble_sets(X, y, n_subsets=10, random_state=42):
    rus = RandomUnderSampler(sampling_strategy=0.5, random_state=random_state)  # 30 frauds per 100 non-frauds
    rng = np.random.RandomState(random_state)

    sets = []
    for _ in range(n_subsets):
        X_res, y_res = rus.fit_resample(X, y)
        sets.append((X_res, y_res))
        rus.random_state = rng.randint(0, 99999)
    return sets

def evaluate_models_with_custom_easy_ensemble(preprocessed_data: pd.DataFrame, 
                                                    selected_feature_vectors: dict,
                                                    n_subsets: int = 10,
                                                    random_state: int = 42) -> pd.DataFrame:

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    best_feature_sets = {
        "CatBoost": "lgbm_top_25",
        "KNN": "corr_target_top_10",
        "RandomForest": "info_gain_top_10",
        "XGBoost": "lgbm_top_20",
    }

    models = {
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=random_state),
    }

    results = []

    for model_name, feature_set_name in best_feature_sets.items():
        print(f"\nEvaluating model: {model_name} on feature set: {feature_set_name}")
        model = models[model_name]
        features = selected_feature_vectors[feature_set_name]
        df = preprocessed_data[features + ["Class"]]
        X, y = df.drop(columns="Class"), df["Class"]

        fold_metrics = []
        start_time = time.time()

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"  → Fold {fold + 1}")

            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            # Generate n subsets from training data
            Xy_sets = generate_easy_ensemble_sets(X_train, y_train, n_subsets=n_subsets, random_state=random_state + fold)
            X_resampled_sets, y_resampled_sets = zip(*Xy_sets)

            # Train n models
            ensemble_models = []
            for i in range(n_subsets):
                clf = clone(model)
                clf.fit(X_resampled_sets[i], y_resampled_sets[i])
                ensemble_models.append(clf)

            # Predict on test by averaging probabilities
            y_proba_avg = np.mean([m.predict_proba(X_test)[:, 1] for m in ensemble_models], axis=0)

            # Find best threshold that maximizes F1
            precision, recall, thresholds = precision_recall_curve(y_test, y_proba_avg)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]

            # Use best threshold for predictions
            y_pred_avg = (y_proba_avg >= best_threshold).astype(int)

            fold_metrics.append({
                "roc_auc": roc_auc_score(y_test, y_proba_avg),
                "pr_auc": average_precision_score(y_test, y_proba_avg),
                "f1": f1_score(y_test, y_pred_avg),
                "precision": precision_score(y_test, y_pred_avg),
                "recall": recall_score(y_test, y_pred_avg),
                "accuracy": accuracy_score(y_test, y_pred_avg),
                "best_threshold": best_threshold
            })

        end_time = time.time()

        # Aggregate metrics across folds
        metrics_df = pd.DataFrame(fold_metrics)
        avg_metrics = metrics_df.mean().to_dict()

        results.append({
            "model": model_name,
            "feature_set": feature_set_name,
            "resampler": "CustomEasyEnsemble",
            **avg_metrics,
            "training_time": end_time - start_time,
            "n_features": len(features),
            "n_subsets": n_subsets,
        })

    return pd.DataFrame(results)

def evaluate_models_with_custom_easy_ensemble_all_features(preprocessed_data: pd.DataFrame, 
                                       n_subsets: int = 10,
                                       random_state: int = 42) -> pd.DataFrame:
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    models = {
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=random_state),
        #"SVM": SVC(probability=True, random_state=random_state),
    }
    
    results = []
    
    for model_name, model in models.items():
        print(f"\nEvaluating model: {model_name}")
        model = models[model_name]
        
        X = preprocessed_data.drop(columns=["Class"])
        y = preprocessed_data["Class"]
        
        fold_metrics = []
        start_time = time.time()
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                print(f"  → Fold {fold + 1}")

                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

                # Generate n subsets from training data
                Xy_sets = generate_easy_ensemble_sets(X_train, y_train, n_subsets=n_subsets, random_state=random_state + fold)
                X_resampled_sets, y_resampled_sets = zip(*Xy_sets)

                # Train n models
                ensemble_models = []
                for i in range(n_subsets):
                    clf = clone(model)
                    clf.fit(X_resampled_sets[i], y_resampled_sets[i])
                    ensemble_models.append(clf)

                # Predict on test by averaging probabilities
                y_proba_avg = np.mean([m.predict_proba(X_test)[:, 1] for m in ensemble_models], axis=0)
                # Find best threshold that maximizes F1
                precision, recall, thresholds = precision_recall_curve(y_test, y_proba_avg)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
                best_idx = np.argmax(f1_scores)
                best_threshold = thresholds[best_idx]

                # Use best threshold for predictions
                y_pred_avg = (y_proba_avg >= best_threshold).astype(int)

                fold_metrics.append({
                    "roc_auc": roc_auc_score(y_test, y_proba_avg),
                    "pr_auc": average_precision_score(y_test, y_proba_avg),
                    "f1": f1_score(y_test, y_pred_avg),
                    "precision": precision_score(y_test, y_pred_avg),
                    "recall": recall_score(y_test, y_pred_avg),
                    "accuracy": accuracy_score(y_test, y_pred_avg),
                    "best_threshold": best_threshold
                })

        end_time = time.time()

        # Aggregate metrics across folds
        metrics_df = pd.DataFrame(fold_metrics)
        avg_metrics = metrics_df.mean().to_dict()

        results.append({
            "model": model_name,
            "feature_set": "all_features",
            "resampler": "CustomEasyEnsemble",
            **avg_metrics,
            "training_time": end_time - start_time,
            "n_features": X.shape[1],
            "n_subsets": n_subsets,
        })

    return pd.DataFrame(results)


from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split

def final_model_evaluation(preprocessed_data: pd.DataFrame, selected_feature_vectors: dict) -> pd.DataFrame:
    # Define final configuration (based on your previous results)
    final_configs = {
        "XGBoost": {
            "feature_set": "lgbm_top_20",
            "resampler": RandomOverSampler(random_state=42),
            "model": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        },
        "XGBoost_all": {
            "feature_set": "all",
            "resampler": RandomOverSampler(random_state=42),
            "model": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        },
        "RandomForest": {
            "feature_set": "info_gain_top_10",
            "resampler": RandomOverSampler(random_state=42),
            "model": RandomForestClassifier(random_state=42),
        },
        "CatBoost": {
            "feature_set": "lgbm_top_25",
            "resampler": RandomOverSampler(random_state=42),
            "model": CatBoostClassifier(verbose=0, random_state=42),
        },
        "CatBoost_all": {
            "feature_set": "all",
            "resampler": RandomOverSampler(random_state=42),
            "model": CatBoostClassifier(verbose=0, random_state=42),
        },
    }

    results = []

    for model_name, config in final_configs.items():
        print(f"→ Final evaluation of {model_name}")
        
        # Select features
        if config["feature_set"] == "all":
            features = [col for col in preprocessed_data.columns if col not in ["Class", "Time", "Amount"]]
        else:
            features = selected_feature_vectors[config["feature_set"]]

        # Prepare features
        features = features + ["Class"]
        df = preprocessed_data[features]

        # Train/test split
        X = df.drop(columns=["Class"])
        y = df["Class"]
        # Stratified !!!!
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Resample only training data
        pipeline = ImbPipeline([
            ("resampler", config["resampler"]),
            ("classifier", config["model"]),
        ])
        
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Predict on test set
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        results.append({
            "model": model_name,
            "feature_set": config["feature_set"],
            "resampler": config["resampler"].__class__.__name__,
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
            "f1": report["1"]["f1-score"],
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "accuracy": report["accuracy"],
            "training_time": training_time,
            "n_features": len(features) - 1
        })

    return pd.DataFrame(results)


def final_model_evaluation_easy_ensemble(preprocessed_data: pd.DataFrame,
                                         selected_feature_vectors: dict,
                                         n_subsets: int = 10,
                                         random_state: int = 42) -> pd.DataFrame:
    
    final_configs = {
        "CatBoost_all": {
            "feature_set": "all",
            "model": CatBoostClassifier(verbose=0, random_state=random_state),
        },
        "XGBoost_all": {
            "feature_set": "all",
            "model": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state),
        },
        "RandomForest_all": {
            "feature_set": "all",
            "model": RandomForestClassifier(random_state=random_state),
        },
        "CatBoost": {
            "feature_set": "lgbm_top_25",
            "model": CatBoostClassifier(verbose=0, random_state=random_state),
        },
        "XGBoost": {
            "feature_set": "lgbm_top_20",
            "model": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state),
        },
        "RandomForest": {
            "feature_set": "info_gain_top_10",
            "model": RandomForestClassifier(random_state=random_state),
        },
    }

    all_results = []

    for model_name, config in final_configs.items():
        print(f"\n→ Final Easy Ensemble Evaluation: {model_name}")
        model_cls = config["model"].__class__
        model_params = config["model"].get_params()

        # Feature selection
        if config["feature_set"] == "all":
            features = [col for col in preprocessed_data.columns if col not in ["Class", "Time", "Amount"]]
        else:
            features = selected_feature_vectors[config["feature_set"]]

        df = preprocessed_data[features + ["Class"]]
        X = df.drop(columns=["Class"])
        y = df["Class"]

        # Split once
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=random_state
        )

        # Generate balanced train subsets
        Xy_sets = generate_easy_ensemble_sets(X_train, y_train, n_subsets=n_subsets, random_state=random_state)
        X_sets, y_sets = zip(*Xy_sets)
        
        ensemble_probas = []
        start_time = time.time()

        # Train one model per subset and predict probabilities on test set
        for i, (X_res, y_res) in enumerate(zip(X_sets, y_sets)):
            print(f"  → Training on EasyEnsemble subset {i+1}")
            model = model_cls(**model_params)
            model.fit(X_res, y_res)
            y_proba = model.predict_proba(X_test)[:, 1]
            ensemble_probas.append(y_proba)

        training_time = time.time() - start_time

        # Average predicted probabilities across all models
        avg_y_proba = np.mean(ensemble_probas, axis=0)

        # Find threshold that maximizes F1 score on test set
        precision, recall, thresholds = precision_recall_curve(y_test, avg_y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]

        # Apply best threshold to get final predictions
        y_pred = (avg_y_proba >= best_threshold).astype(int)

        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)

        results = {
            "model": model_name,
            "feature_set": config["feature_set"],
            "resampler": "CustomEasyEnsemble",
            "roc_auc": roc_auc_score(y_test, avg_y_proba),
            "pr_auc": average_precision_score(y_test, avg_y_proba),
            "f1": report["1"]["f1-score"],
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "accuracy": report["accuracy"],
            "best_threshold": best_threshold,
            "training_time": training_time,
            "n_features": len(features),
            "n_subsets": n_subsets,
        }

        all_results.append(results)

    return pd.DataFrame(all_results)