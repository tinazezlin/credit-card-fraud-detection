import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier




def remove_highly_correlated_features(data: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    corr_matrix = data.drop(columns="Class").corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(to_drop)
    return data.drop(columns=to_drop)


def compute_feature_target_correlation(data: pd.DataFrame) -> pd.DataFrame:
    correlations = data.corr()['Class'].drop('Class')
    return (
        correlations
        .reindex(correlations.abs().sort_values(ascending=False).index)
        .reset_index()
        .rename(columns={'index': 'feature', 'Class': 'correlation_with_target'})
    )



def compute_information_gain(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute information gain (mutual information) between each feature and the target.

    Args:
        data: DataFrame that includes all features and the 'Class' column.

    Returns:
        DataFrame with columns: 'feature', 'information_gain', sorted descending.
    """
    # Drop unscaled 'Amount' and 'Time' to avoid skew; scaled versions should be used instead if present.
    X = data.drop(columns=['Class', 'Amount', 'Time'])
    y = data['Class']

    info_gain = mutual_info_classif(X, y, random_state=42)

    result = pd.DataFrame({
        'feature': X.columns,
        'information_gain': info_gain
    }).sort_values(by='information_gain', ascending=False).reset_index(drop=True)

    return result



def compute_extra_trees_importance(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute feature importance using ExtraTreesClassifier.

    Args:
        data: DataFrame including all features and the target column 'Class'.

    Returns:
        DataFrame with columns: 'feature', 'importance', sorted descending.
    """
    X = data.drop(columns=['Class', 'Amount', 'Time'])
    y = data['Class']

    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)

    return importance_df

def compute_lightgbm_feature_importance(data: pd.DataFrame) -> pd.DataFrame:
    """
    Train a LightGBM classifier and return feature importances.

    Args:
        data: DataFrame that includes all features and the 'Class' column.

    Returns:
        DataFrame with 'feature' and 'importance', sorted descending.
    """
    X = data.drop(columns=['Class', 'normAmount', 'normTime'])
    X_norm = data.drop(columns=['Class', 'Amount', 'Time'])
    y = data["Class"]

    model = lgb.LGBMClassifier(random_state=42, deterministic=True)
    model.fit(X, y)
    
    model_norm = lgb.LGBMClassifier(random_state=42, deterministic=True)
    model_norm.fit(X_norm, y)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    
    importance_df_norm = pd.DataFrame({
        "feature": X_norm.columns,
        "importance": model_norm.feature_importances_
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return importance_df, importance_df_norm


def generate_feature_vectors(
    cleaned_features_corr: pd.DataFrame,
    correlation_with_target: pd.DataFrame,
    information_gain_scores: pd.DataFrame,
    extra_trees_importance: pd.DataFrame,
    lightgbm_feature_importance: pd.DataFrame,
    lightgbm_feature_importance_norm: pd.DataFrame
) -> dict:
    """
    Generate different feature sets based on the results of feature selection methods.

    Returns:
        Dictionary of feature lists keyed by selection method.
    """
    feature_vectors = {}

    # Top-k features by each method (e.g., top 15)
    k_values = [10, 15, 20, 25]  # or any you want to explore

    for k in k_values:
        feature_vectors[f"info_gain_top_{k}"] = information_gain_scores.head(k)["feature"].tolist()
        feature_vectors[f"corr_target_top_{k}"] = correlation_with_target.head(k)["feature"].tolist()
        feature_vectors[f"extra_trees_top_{k}"] = extra_trees_importance.head(k)["feature"].tolist()
        feature_vectors[f"lgbm_top_{k}"] = lightgbm_feature_importance.head(k)["feature"].tolist()
        feature_vectors[f"lgbm_top_norm_{k}"] = lightgbm_feature_importance_norm.head(k)["feature"].tolist()


        # Combine most common features across methods
        from collections import Counter
        combined = (
            feature_vectors[f"corr_target_top_{k}"] +
            feature_vectors[f"info_gain_top_{k}"] +
            feature_vectors[f"extra_trees_top_{k}"] +
            feature_vectors[f"lgbm_top_{k}"] +
            feature_vectors[f"lgbm_top_norm_{k}"]
        )
        most_common = [
            item for item, count in Counter(combined).items() if count >= 2
        ]
        feature_vectors[f"intersection_2plus_{k}"] = most_common

    return feature_vectors

def filter_selected_features(data: pd.DataFrame, feature_vectors) -> pd.DataFrame:
    """
    Filter the DataFrame to keep only selected important features.

    Args:
        data: The preprocessed DataFrame with all features and 'Class'.
        feature_vectors: Dictionary of feature lists keyed by selection method.

    Returns:
        Filtered DataFrame containing only selected features + 'Class'.
    """
    selected_features = feature_vectors["intersection_2plus"]
    data_filtered = data[selected_features + ['Class']]
    
    return data_filtered

def all_features(data: pd.DataFrame,) -> pd.DataFrame:
    """
    Filter the DataFrame to keep all features, excluding not normalized Amunt and Time.

    Args:
        data: The preprocessed DataFrame with all features and 'Class'.
    Returns:
        DataFrame containing all normalized features + 'Class'.
    """
    data_filtered = data.drop(columns=['Amount', 'Time'])
    
    return data_filtered