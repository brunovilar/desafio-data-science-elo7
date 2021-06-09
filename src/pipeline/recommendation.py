from enum import Enum
from pathlib import Path
from typing import List
import fasttext
import funcy as fp
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .. import settings
from ..logging import LoggerFactory
from ..entities import Product
from ..pipeline.training_pipeline import compute_embeddings_frame
from ..pipeline.inference_pipeline import load_model_resources, make_supervised_intent_classification

logger = LoggerFactory.get_logger(__name__)

CATEGORY_PROBABILITY_THRESHOLD = .25


class RecommendationWeights(Enum):
    """Defines the how much each parameter will influence the recommendations"""
    TITLE = 1.0  # Similarity with title embedding
    TAGS = 0.75  # Similarity with concatenated tags embedding
    CATEGORY = 0.5  # Probability of a query being of a category
    ORDER_PER_VIEW = 0.25  # How efficient an order


class QueryIntent(Enum):
    FOCUS = 'Foco'
    EXPLORATION = 'Exploração'


def generate_product_recommendation_dataset() -> pd.DataFrame():
    """Creates a product dataset with the elements necessary filter and to perform recommendations. """

    dataset_file_path = Path(settings.DATA_PATH).joinpath('processed', 'product_recommendation.feather')

    columns_to_read = ['product_id', 'title', 'concatenated_tags', 'category', 'view_counts', 'order_counts']
    base_frame = (pd
                  .read_csv(Path(settings.DATA_PATH).joinpath('interim', 'training.csv'),
                            usecols=columns_to_read)
                  .drop_duplicates('product_id')
                  )

    embeddings_columns = ['title', 'concatenated_tags']
    base_frame = pd.concat([base_frame, compute_embeddings_frame(base_frame, embeddings_columns)], axis=1)
    base_frame.reset_index(drop=True).to_feather(str(dataset_file_path))

    logger.info('Product Recommendation Dataset was generated.', extra={'props': {'dataset_length': len(base_frame)}})


def compute_embedding_columns_similarity(base_embeddings_frame: pd.DataFrame,
                                         reference_embedding: np.array,
                                         columns: List[str]) -> pd.DataFrame:
    """Adds, for each column, the similarity between its elements and the reference embedding"""

    similarity_frame = base_embeddings_frame[[]].copy()
    reference_embedding = np.expand_dims(reference_embedding, axis=0)

    for column in columns:
        column_embeddings = np.stack(base_embeddings_frame[f'{column}_embedding'].to_numpy(), axis=0)
        similarity_frame[f'{column}_similarity'] = np.concatenate(
            cosine_similarity(reference_embedding, column_embeddings), axis=0)

    return similarity_frame


def recommend_products(base_frame: pd.DataFrame, query: str, items_to_retrieve: int = 10) -> pd.DataFrame:

    ft_model = fasttext.load_model(str(Path(settings.MODELS_PATH).joinpath(settings.EMBEDDINGS_MODEL)))
    query_embedding = ft_model.get_sentence_vector(query)

    search_frame = base_frame.copy()
    search_frame = pd.concat([search_frame,
                              compute_embedding_columns_similarity(search_frame, query_embedding,
                                                                   ['title', 'concatenated_tags'])]
                             , axis=1)

    columns_to_drop = [item for item in search_frame.columns if item.endswith('_embedding')]

    return (search_frame
            .fillna({'order_counts': 0,
                     'view_counts': 0})
            .assign(orders_per_views=lambda f: ((f['order_counts'] + 1) / (f['view_counts'] + 1))
                    .apply(lambda x: min(1.0, x)))
            .assign(score=lambda f: (f['title_similarity'] * RecommendationWeights.TITLE.value) +
                                    (f['concatenated_tags_similarity'] * RecommendationWeights.TAGS.value) +
                                    (f['category_prob'] * RecommendationWeights.CATEGORY.value) +
                                    (f['orders_per_views'] * RecommendationWeights.ORDER_PER_VIEW.value)
                    )
            .sort_values(by='score', ascending=False)
            .head(items_to_retrieve)
            .drop(columns=columns_to_drop, inplace=False)
            )


def make_recommendations_for_query(query: str) -> List[Product]:
    # Identify query intent
    query_product = Product(title=query, price=None, concatenated_tags=query)
    query_intent = fp.first(make_supervised_intent_classification([query]))

    # Compute the probability of the query being of each category
    category_preprocessing_model, category_model, category_label_encoder_model = \
        load_model_resources(settings.CATEGORY_CLASSIFICATION_RUN_ID)
    preprocessed_products = category_preprocessing_model.predict(pd.DataFrame([query_product]))
    predictions = category_model.predict_proba(preprocessed_products)

    # Get categories that have probability higher than a threshold (exploration) ou the highest probability (focus)
    categories = np.array(category_label_encoder_model.classes_)
    if query_intent != QueryIntent.FOCUS.value:
        selected_categories = set(categories[predictions[0] >= CATEGORY_PROBABILITY_THRESHOLD])
    else:
        selected_categories = set([categories[predictions[0].argmax()]])

    # Create a dictionary with the probability of each selected category
    category_prob_dict = {category: prob
                          for prob, category in zip(predictions[0], categories)
                          if category in selected_categories
                          }

    # Load products dataset
    dataset_file_path = Path(settings.DATA_PATH).joinpath('processed', 'product_recommendation.feather')
    if not dataset_file_path.exists():
        generate_product_recommendation_dataset()
    base_frame = pd.read_feather(str(dataset_file_path))

    # Preprocess dataset
    base_frame = (base_frame
                  .loc[lambda f: f['category'].isin(selected_categories)]
                  .assign(category_prob=lambda f: f['category'].map(category_prob_dict))
                  )

    # Make recommendations
    recommendations_frame = recommend_products(base_frame, query)

    # Format and return output
    for item in recommendations_frame[['product_id', 'title']].to_dict(orient='records'):
        yield Product(**item)

    # Log recommendation details
    log_object = {
        'category': 'action_result',
        'query': query,
        'query_intent': query_intent,
        'selected_categories': list(selected_categories),
        'categories_probs': {category: prob
                             for prob, category in zip(predictions[0], categories)},
        'dataset_length': len(recommendations_frame),
        'recommendations': recommendations_frame[['product_id', 'title', 'score']].to_dict(orient='records')
    }
    logger.info('Recommendations for Query', extra={'props': log_object})
