from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_baseline_models, evaluate_models_with_resampling, evaluate_baseline_models_across_feature_sets, evaluate_models_with_resampling_svm, evaluate_models_with_custom_easy_ensemble, evaluate_models_with_resampling_all_features, evaluate_models_with_custom_easy_ensemble_all_features, final_model_evaluation, final_model_evaluation_easy_ensemble

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
    
    top_models = node(
            func=evaluate_models_with_resampling,
            inputs=["preprocessed_data", "selected_feature_vectors"],  # from your feature selection
            outputs="model_resampling_results",
            name="evaluate_models_with_resampling_node"
        )
    
    top_models_svm = node(
            func=evaluate_models_with_resampling_svm,
            inputs=["preprocessed_data", "selected_feature_vectors"],  # from your feature selection
            outputs="model_resampling_results_svm",
            name="evaluate_models_with_resampling_svm_node"
        )
    
    top_models_all_features = node(
            func=evaluate_models_with_resampling_all_features,
            inputs="all_features_norm_data",  
            outputs="model_resampling_results_all_features",
            name="evaluate_models_with_resampling_all_features_node"
        )
    
    top_models_custom_easy_ensamble = node(
        func=evaluate_models_with_custom_easy_ensemble,
        inputs=["preprocessed_data", "selected_feature_vectors"],  # from your feature selection
        outputs="model_custom_easy_ensamble_results",
        name="evaluate_models_with_custom_easy_esamble_node"
    )
    
    top_models_custom_easy_ensamble_all_features = node(
        func=evaluate_models_with_custom_easy_ensemble_all_features,
        inputs="all_features_norm_data",   
        outputs="model_custom_easy_ensamble_results_all_features",
        name="evaluate_models_with_custom_easy_esamble_all_features_node"
    )
    
    final_model_eval_node = node(
        func=final_model_evaluation,
        inputs=["preprocessed_data", "selected_feature_vectors"],
        outputs="final_model_evaluation_results",
        name="final_model_evaluation_node"
    )
    
    final_model_eval_custom_ee_node = node(
        func=final_model_evaluation_easy_ensemble,
        inputs=["preprocessed_data", "selected_feature_vectors"],
        outputs="final_model_evaluation_custom_ee_results",
        name="final_model_evaluation_custom_ee_node"
    )



    return Pipeline([
        evaluate_models_across_features_node,
        baseline_models,
        baseline_models_norm_features,
        top_models,
        top_models_svm,
        top_models_all_features,
        top_models_custom_easy_ensamble,
        top_models_custom_easy_ensamble_all_features,
        final_model_eval_node,
        final_model_eval_custom_ee_node
    ])