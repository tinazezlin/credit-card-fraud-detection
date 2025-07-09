from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_creditcard_data

def create_pipeline(**kwargs) -> Pipeline:
    preprocess_data = node(
        func=preprocess_creditcard_data, 
        inputs="raw_creditcard_data",
        outputs="preprocessed_data", 
        name="preprocess_creditcard_data_node"
    )

    return Pipeline([
        preprocess_data
    ])