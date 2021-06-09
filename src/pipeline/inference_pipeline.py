import os
from typing import List, Tuple, Any
from functools import lru_cache

import numpy as np
import pandas as pd
import mlflow

from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import LabelEncoder
from ..logging import LoggerFactory
from ..entities import Product
from .. import settings
from ..pipeline.training_pipeline import (compute_embeddings_frame,
                                          create_feature_matrix,
                                          get_qualified_queries,
                                          compute_frame_column_entropy)

logger = LoggerFactory.get_logger(__name__)

@lru_cache(maxsize=10)
def load_model_resources(run_id: str) -> Tuple[PyFuncModel, Any, LabelEncoder]:

    # Retrieves Mlflow Run Data
    mlflow_client = MlflowClient(tracking_uri=settings.TRACKING_URI)
    model_run = mlflow_client.get_run(run_id)
    artifact_uri = model_run.info.artifact_uri

    # Load the set of functions and parameters to preprocess data
    preprocessing_model_path = os.path.join(model_run.info.artifact_uri
                                            .replace(model_run.info.run_id,
                                                     model_run.data.tags.get('mlflow.parentRunId')),
                                            'log',
                                            'preprocessing_model')
    preprocessing_model = mlflow.pyfunc.load_model(preprocessing_model_path)

    # Load the model
    model = mlflow.sklearn.load_model(f'{artifact_uri}/model')

    # Load the label encoder if it exists
    parent_path = artifact_uri.replace(model_run.info.run_id, model_run.data.tags.get('mlflow.parentRunId'))
    label_encoder_path = os.path.join(parent_path, 'label_encoder')
    label_encoder_model = mlflow.sklearn.load_model(label_encoder_path)

    return preprocessing_model, model, label_encoder_model


def make_batch_predictions(products: List[Product]) -> List[Product]:

    # Retrieve the model's reference resources from Mlflow
    preprocessing, model, label_encoder = load_model_resources(settings.CATEGORY_CLASSIFICATION_RUN_ID)
    # Load products as a data frame for batch prediction
    products_frame = pd.DataFrame(products)
    # Preprocess features
    features = preprocessing.predict(products_frame)
    # Make predictions
    predictions = model.predict(features)
    # Assign the prediction as categories
    products_frame['category'] = label_encoder.inverse_transform(predictions)
    # Cast products from DataFrame records to dicts and then dataclass again
    return [Product(**item)
            for item in products_frame.to_dict(orient='records')]


def predict_product_cluster(base_frame: pd.DataFrame, embedding_columns: List[str], clustering_model: Any) -> np.array:
    embeddings_columns_names = [f'{item}_embedding' for item in embedding_columns]
    embeddings_frame = pd.concat([base_frame, compute_embeddings_frame(base_frame, embedding_columns)], axis=1)

    # Create features array based on embeddings
    X_clustering = create_feature_matrix(embeddings_frame,
                                         feature_columns=[],
                                         embeddings_columns=embeddings_columns_names)

    # Define the cluster of each qualified record
    return clustering_model.predict(X_clustering)


def make_unsupervised_intent_classification(base_frame: pd.DataFrame,
                                            embedding_columns: List[str],
                                            minimum_number_of_products: int,
                                            entropy_threshold: float,
                                            clustering_model: Any
                                            ) -> np.array:
    """Define the query intention based on the entropy analysis of the products with interactions withing the results

       Output:
            -1: Undefined: The query does not have the minimum number of products with interactions
            0: Exploration -- The query was created to see a wide range of product types
            1: Focus -- The query was created to find a specific product
    """

    # Assign a cluster for each record
    internal_frame = base_frame.copy()
    internal_frame['cluster'] = predict_product_cluster(internal_frame, embedding_columns, clustering_model)

    # Select qualified queries based on the products with interactions
    qualified_queries = get_qualified_queries(internal_frame, minimum_number_of_products)

    frame_slice = internal_frame.loc[lambda f: f['query'].isin(qualified_queries)]

    internal_frame = (internal_frame
                      .set_index('query')
                      .merge(compute_frame_column_entropy(frame_slice, 'query', 'cluster').rename('entropy'),
                             on='query', how='left')
                      .fillna({'entropy': -1})
                      .reset_index()
                      )

    return (internal_frame['entropy']
            .apply(lambda e: -1 if e < 0 else int(e <= entropy_threshold))
            )


def make_supervised_intent_classification(queries: List[str]) -> List[str]:
    # Retrieve the model's reference resources from Mlflow
    preprocessing, model, label_encoder = load_model_resources(settings.SUPERVISED_INTENT_CLASSIFICATION_RUN_ID)
    # Load products as a data frame for batch prediction
    queries_frame = pd.DataFrame({'query': queries})
    # Preprocess features
    features = preprocessing.predict(queries_frame)
    # Make predictions
    predictions = model.predict(features)

    # Return the intent of each query as a list of strings
    return label_encoder.inverse_transform(predictions)
