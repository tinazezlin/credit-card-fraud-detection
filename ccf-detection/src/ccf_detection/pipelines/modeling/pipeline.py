from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_baseline_models, evaluate_models_with_resampling, evaluate_baseline_models_across_feature_sets

def create_pipeline(**kwargs) -> Pipeline:
    
    evaluate_models_across_features_node = node(
        func=evaluate_baseline_models_across_feature_sets,
        inputs=["preprocessed_data", "selected_feature_vectors"],
        outputs="model_evaluation_results_across_features",
        name="evaluate_models_across_features_node"
    )
        
    baseline_models =  node(
            func=evaluate_baseline_models,
            inputs=["filtered_data","params:time_log_path_filtered_features"],
            outputs="baseline_model_scores",
            name="evaluate_baseline_models_node"
        )
    
    baseline_models_norm_features =  node(
            func=evaluate_baseline_models,
            inputs=["all_features_norm_data","params:time_log_path_all_features"],
            outputs="baseline_model_scores_norm_features",
            name="evaluate_baseline_models_norm_features_node"
        )
    
   # top_models = node(
   #         func=evaluate_models_with_resampling,
   #         inputs="filtered_data",  # from your feature selection
   #         outputs="model_resampling_results",
   #         name="evaluate_models_with_resampling_node"
   #     )
    
    # TOP 5 MODELS AND RESAMPLING
    rf_model = node(
            func=evaluate_models_with_resampling,
            inputs=["filtered_data", "params:RF"],  # from your feature selection
            outputs="rf_resampling_results",
            name="evaluate_rf_with_resampling_node"
        )
    
    knn_model = node(
            func=evaluate_models_with_resampling,
            inputs=["filtered_data", "params:KNN"],  # from your feature selection
            outputs="knn_resampling_results",
            name="evaluate_knn_with_resampling_node"
        )
    
    svm_model = node(
            func=evaluate_models_with_resampling,
            inputs=["filtered_data", "params:SVM"],  # from your feature selection
            outputs="svm_resampling_results",
            name="evaluate_svm_with_resampling_node"
        )
    
    XGBoost_model = node(
            func=evaluate_models_with_resampling,
            inputs=["filtered_data", "params:XGBoost"],  # from your feature selection
            outputs="XGBoost_resampling_results",
            name="evaluate_XGBoost_with_resampling_node"
        )
    
    CatBoost_model = node(
            func=evaluate_models_with_resampling,
            inputs=["filtered_data", "params:CatBoost"],  # from your feature selection
            outputs="CatBoost_resampling_results",
            name="evaluate_CatBoost_with_resampling_node"
        )


    return Pipeline([
        evaluate_models_across_features_node,
        baseline_models,
        baseline_models_norm_features,
        #top_models
        rf_model,
        knn_model,
        svm_model,
        XGBoost_model,
        CatBoost_model
    ])