import pandas as pd
import mlflow.pyfunc
from mlflow.pyfunc import PythonModelContext
from typing import List, Any
from ..pipeline.inference_pipeline import make_unsupervised_intent_classification
from ..pipeline.training_pipeline import preprocess_for_query_intent_classification


class PreprocessingWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper class to keep all references to preprocess data for Category Classification"""

    def __init__(self,
                 partial_clean_fn,
                 preprocess_fn,
                 numeric_columns_to_impute,
                 text_columns_to_encode,
                 similarity_features,
                 categories,
                 category_embeddings,
                 numeric_stats,
                 matrix_creation_fn,
                 basic_features,
                 embeddings_features
                 ):

        self.partial_clean_fn = partial_clean_fn
        self.preprocess_fn = preprocess_fn
        self.numeric_columns_to_impute = numeric_columns_to_impute
        self.text_columns_to_encode = text_columns_to_encode
        self.similarity_features = similarity_features
        self.categories = categories
        self.category_embeddings = category_embeddings
        self.numeric_stats = numeric_stats
        self.matrix_creation_fn = matrix_creation_fn
        self.basic_features = basic_features
        self.embeddings_features = embeddings_features

    def predict(self, context: PythonModelContext, model_input: pd.DataFrame):
        _model_input = model_input.copy()

        features = self.preprocess_fn(_model_input,
                                      self.categories,
                                      self.category_embeddings,
                                      self.numeric_stats,
                                      self.numeric_columns_to_impute,
                                      self.text_columns_to_encode,
                                      self.similarity_features,
                                      self.partial_clean_fn)

        return self.matrix_creation_fn(features, self.basic_features, self.embeddings_features)


class UnsupervisedIntentClassificationWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper class to keep all references to preprocess data for Unsupervised Intent Classification"""

    def __init__(self,
                 embedding_columns: List[str],
                 minimum_number_of_products: int,
                 entropy_threshold: float,
                 clustering_model: Any,
                 ):

        self.embedding_columns = embedding_columns
        self.minimum_number_of_products = minimum_number_of_products
        self.entropy_threshold = entropy_threshold
        self.clustering_model = clustering_model

    def predict(self, context: PythonModelContext, model_input: pd.DataFrame):
        _model_input = model_input.copy()

        return make_unsupervised_intent_classification(_model_input,
                                                       self.embedding_columns,
                                                       self.minimum_number_of_products,
                                                       self.entropy_threshold,
                                                       self.clustering_model)


class SupervisedIntentClassificationPreprocessingWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper class to keep all references to preprocess data for Supervised Intent Classification"""

    def __init__(self,
                 basic_features: List[str] = None,
                 artisanal_features: List[str] = None,
                 embeddings_features: List[str] = None
                 ):

        self.basic_features = basic_features or []
        self.artisanal_features = artisanal_features or []
        self.embeddings_features = embeddings_features or []

    def predict(self, context: PythonModelContext, model_input: pd.DataFrame) -> pd.DataFrame:

        return preprocess_for_query_intent_classification(model_input,
                                                          basic_features=[],
                                                          artisanal_features=self.artisanal_features,
                                                          embeddings_features=self.embeddings_features)
