"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines.data_preprocessing.pipeline import create_pipeline as create_data_preprocessing_pipeline
from .pipelines.feature_selection.pipeline import create_pipeline as create_feature_selection
from .pipelines.modeling.pipeline import create_pipeline as evaluate_models





def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
   # Register the default pipeline (or any other pipelines you define)
   
    data_preprocessing = create_data_preprocessing_pipeline()
    feature_selection = create_feature_selection()
    model_evaluation = evaluate_models()

    return {
        "__default__": data_preprocessing,
        "data_preprocessing": data_preprocessing,
        "feature_selection": feature_selection,
        "model_evaluation": model_evaluation

    }

