import os
from typing import List, Tuple, Any
from functools import lru_cache

import pandas as pd
import mlflow
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient

from sklearn.preprocessing import LabelEncoder
from ..entities import Product
from .. import settings


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

    # Load the label encoder
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
