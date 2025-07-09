from kedro.pipeline import Pipeline, node, pipeline
from .nodes import remove_highly_correlated_features, compute_feature_target_correlation, compute_information_gain, compute_extra_trees_importance, compute_lightgbm_feature_importance, generate_feature_vectors, filter_selected_features

def create_pipeline(**kwargs) -> Pipeline:
    preprocess_data = node(
            func=remove_highly_correlated_features,
            inputs="preprocessed_data",
            outputs="cleaned_features_corr",
            name="remove_high_corr_features_node"
        )
    
    feaure_target_correlation = node(
            func=compute_feature_target_correlation,
            inputs="preprocessed_data",
            outputs="correlation_with_target",
            name="feature_target_correlation_node"
        )
    
    information_gain = node(
            func=compute_information_gain,
            inputs="preprocessed_data",
            outputs="information_gain_scores",
            name="compute_information_gain_node"
        )
    
    extra_trees_importance = node(
            func=compute_extra_trees_importance,
            inputs="preprocessed_data",
            outputs="extra_trees_importance",
            name="compute_extra_trees_importance_node"
        )
    
    lightgbm_feature_importance = node(
            func=compute_lightgbm_feature_importance,
            inputs="preprocessed_data",
            outputs=["lightgbm_feature_importance","lightgbm_feature_importance_norm"],
            name="compute_lightgbm_feature_importance_node"
        )
    
    generate_features = node(
            func=generate_feature_vectors,
            inputs=[
                "cleaned_features_corr",
                "correlation_with_target",
                "information_gain_scores",
                "extra_trees_importance",
                "lightgbm_feature_importance",
                "lightgbm_feature_importance_norm"
            ],
            outputs="selected_feature_vectors",
            name="generate_feature_vectors_node"
        )
    
    filter_selected_features_node = node(
            func=filter_selected_features,
            inputs=["preprocessed_data", "selected_feature_vectors"],
            outputs="filtered_data",
            name="filter_selected_features_node"
        )


    return Pipeline([
        preprocess_data,
        feaure_target_correlation,
        information_gain,
        extra_trees_importance,
        lightgbm_feature_importance,
        generate_features,
        filter_selected_features_node        
    ])