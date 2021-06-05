import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax
import fasttext
from functools import partial
from typing import List, Tuple, Dict, Set
from .. import settings
from ..utils.experiments import compute_entropy
from ..utils.text import extract_artisanal_text_features


def compute_embeddings_frame(base_frame: pd.DataFrame, columns: List[str],
                             ft_model_ref: str = 'cc.pt.300.bin') -> pd.DataFrame:
    """For each column, creates an embedding representation of its elements"""

    ft_model = fasttext.load_model(os.path.join(settings.MODELS_PATH, ft_model_ref))
    embeddings_frame = pd.DataFrame()
    for column in columns:
        embeddings_frame[f'{column}_embedding'] = (
            base_frame
            .assign(**{f'{column}': lambda f: f[column].apply(lambda v: '' if pd.isna(v) else v)})
            [column]
            .str
            .lower()
            .apply(ft_model.get_sentence_vector)
        )
    return embeddings_frame


def compute_average_embeddings(base_frame: pd.DataFrame, columns: List[str], categories: List[str]) -> \
        Dict[str, np.array]:
    """Creates a dictionary with category as key and average of elements embeddings as value"""

    category_embeddings_dict = {}
    for category in categories:
        for column in columns:
            embeddings = np.stack(base_frame
                                  .loc[lambda f: f['category'] == category]
                                  [f'{column}_embedding']
                                  .to_numpy(),
                                  axis=0)
            category_embeddings_dict[(category, column)] = np.mean(embeddings, axis=0)

    return category_embeddings_dict


def compute_column_category_similarity(base_embeddings_frame: pd.DataFrame,
                                       categories_embeddings: dict,
                                       column: str,
                                       apply_softmax: bool = False) -> pd.DataFrame:
    """Adds, for each category, the similarity between its elements and the columns embeddings"""

    similarity_frame = base_embeddings_frame[[]].copy()
    categories = sorted(set([category for category, _ in categories_embeddings.keys()]))
    similarity_columns = [f'similarity_{column}_{category.lower().replace(" ", "_")}' for category in categories]

    similarities_list = []
    for category in categories:
        column_embeddings = np.stack(base_embeddings_frame[f'{column}_embedding'].to_numpy(), axis=0)
        category_embedding = categories_embeddings[(category, column)]
        category_embedding = np.expand_dims(category_embedding, axis=0)

        similarities_list.append(cosine_similarity(category_embedding, column_embeddings))

    if apply_softmax:
        similarity_frame[similarity_columns] = softmax(np.concatenate(similarities_list, axis=0), axis=0).T
    else:
        similarity_frame[similarity_columns] = np.concatenate(similarities_list, axis=0).T

    return similarity_frame


def compute_stats_for_numeric_values(base_frame: pd.DataFrame, columns: List[str]) -> Dict[Tuple[str, str], float]:
    """Creates a nested dictionary with column name, statistics and value"""

    return (base_frame
            [columns]
            .agg({np.median,
                  np.mean,
                  np.std,
                  np.min,
                  np.max
                  })
            .to_dict()
            )


def fill_missing_numeric_values(base_frame: pd.DataFrame, numeric_stats_dict: dict,
                                statistics: str = 'median') -> pd.DataFrame:
    """Fills missing values from columns based on a dictionary with the columns statistics"""

    filled_frame = base_frame.copy()
    for column in numeric_stats_dict.keys():
        filled_frame[column] = filled_frame[column].apply(lambda v: numeric_stats_dict[column][statistics])

    return filled_frame


def create_feature_matrix(base_frame: pd.DataFrame, feature_columns: List[str],
                          embeddings_columns: List[str]) -> np.array:
    """Extracts and formats features as a numeric matrix"""

    embeddings_list = []
    for column in embeddings_columns:
        embedding_column = np.stack(base_frame[column].to_numpy(), axis=0)
        embeddings_list.append(embedding_column)

    basic_columns_list = [np.expand_dims(base_frame[column].to_numpy(), axis=1)
                          for column in feature_columns]

    return np.concatenate(embeddings_list + basic_columns_list, axis=1)


def create_preprocessing_resources(base_frame: pd.DataFrame,
                                   embeddings_columns: List[str],
                                   numeric_columns: List[str],
                                   text_preprocessing_fn: partial) -> (List[str], Dict, Dict):
    """Creates dictionaries and reference lists to preprocess data for training and test"""

    categories = sorted(base_frame['category'].unique().tolist())
    columns_to_copy = ['category'] + embeddings_columns + numeric_columns

    # Creates embeddings for textual columns
    features_frame = (base_frame
                      [columns_to_copy]
                      .copy()
                      .fillna({c: '' for c in embeddings_columns})
                      )

    for feature in embeddings_columns:
        features_frame[feature] = features_frame[feature].apply(text_preprocessing_fn)

    embeddings_frame = compute_embeddings_frame(base_frame, embeddings_columns)
    features_frame = pd.concat([features_frame, embeddings_frame], axis=1)

    # Creates a dictionary with the average of the category embeddings
    category_embeddings_dict = compute_average_embeddings(features_frame, embeddings_columns, categories)

    # Creates a dictionary with statistics from numeric values to impute missing values
    numerics_stats_dict = compute_stats_for_numeric_values(base_frame, numeric_columns)

    return categories, category_embeddings_dict, numerics_stats_dict


def combine_columns(base_frame: pd.DataFrame, columns: List[Tuple[str, str]], sep: str=None) -> pd.DataFrame():
    """Combines two columns into a new one"""
    sep = sep or ''

    for column_a, column_b in columns:
        base_frame = (base_frame
                      .assign(**{f'{column_a}_and_{column_b}': lambda f: f[column_a] + sep + f[column_b]})
                     )
    return base_frame


def preprocess_features(base_frame: pd.DataFrame,
                        category_embeddings: Dict,
                        numeric_stats: Dict,
                        numeric_features: List[str],
                        text_features: List[str],
                        similarity_features: List[str],
                        columns_to_combine: List[Tuple[str, str]],
                        text_preprocessing_fn: partial
                        ) -> pd.DataFrame:
    """Preprocesses features for training, validation and test"""
    combination_columns = [f'{c_a}_and_{c_b}' for c_a, c_b in columns_to_combine]
    text_features = text_features + combination_columns

    columns_to_copy = ['category'] + numeric_features + text_features
    features_frame = (base_frame
                      .pipe(combine_columns, columns_to_combine)
                      [columns_to_copy]
                      .copy()
                      .fillna({c: '' for c in text_features})
                      )

    for feature in text_features:
        features_frame[feature] = features_frame[feature].apply(text_preprocessing_fn)

    # Computes embeddings for textual columns
    embeddings_frame = compute_embeddings_frame(features_frame, text_features)
    features_frame = pd.concat([features_frame.drop(columns=text_features), embeddings_frame], axis=1)

    # Adds similarity features
    for feature in similarity_features:
        category_similarity_frame = compute_column_category_similarity(features_frame, category_embeddings,
                                                                       feature)
        features_frame = pd.concat([features_frame, category_similarity_frame], axis=1)

    # Fills missing values
    features_frame = fill_missing_numeric_values(features_frame, numeric_stats)

    # Standardize numeric values
    for column in numeric_features:
        features_frame[column] = (features_frame[column] - numeric_stats[column]['mean']) / numeric_stats[column]['std']

    return features_frame


def count_frame_items(base_frame: pd.DataFrame, group_column: str, count_column: str, drop_duplicates=True) -> pd.Series:

    frame_view = base_frame[[group_column, count_column]]

    if drop_duplicates:
        frame_view = frame_view.drop_duplicates()

    return (frame_view
            .assign(records=1)
            .groupby(group_column)
            .sum()
            ['records'])


def compute_frame_column_entropy(base_frame: pd.DataFrame, group_column: str, count_column: str) -> pd.Series:
    return (base_frame
            [[group_column, count_column]]
            .groupby(group_column)
            .apply(lambda f: compute_entropy(f[count_column].to_numpy())))


def get_qualified_queries(base_frame: pd.DataFrame, minimum_number_of_products: int) -> Set[str]:
    return set(base_frame
               [['query']]
               .assign(products=1)
               .groupby('query')
               .sum()
               .reset_index()
               .loc[lambda f: f['products'] >= minimum_number_of_products]
               ['query']
               .unique()
               .tolist())


def preprocess_for_query_intent_classification(base_frame: pd.DataFrame,
                                               basic_features: List[str],
                                               artisanal_features: List[str],
                                               embeddings_features: List[str]
                                               ) -> pd.DataFrame:

    features_frame = base_frame.copy()

    # Create embeddings for each feature in embeddings_features
    features_frame = pd.concat([features_frame,
                                compute_embeddings_frame(features_frame, embeddings_features)], axis=1)

    # Create embeddings_columns with the new columns created with embeddings
    embeddings_columns = [f'{feature}_embedding' for feature in embeddings_features]

    word_pattern = re.compile(r'\W')
    for feature in artisanal_features:
        features_frame = (
            features_frame
            .assign(word_level_feature=
                    lambda f: f[feature].apply(lambda q: extract_artisanal_text_features(q, word_pattern)))
            .assign(char_level_feature=lambda f: f[feature].apply(extract_artisanal_text_features))
            .rename(columns={c: f'{feature}_{c}' for c in ['word_level_feature', 'char_level_feature']})
        )
        # Extend embeddings columns with artisanal features
        embeddings_columns.extend([f'{feature}_{c}' for c in ['word_level_feature', 'char_level_feature']])

    return create_feature_matrix(features_frame, basic_features, embeddings_columns=embeddings_columns)
