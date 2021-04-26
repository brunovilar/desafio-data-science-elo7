import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax
import fasttext
from functools import partial
from typing import List, Tuple, Dict
from .. import settings


def compute_embeddings_frame(base_frame: pd.DataFrame, columns: List[str],
                             ft_model_ref: str = 'cc.pt.300.bin') -> pd.DataFrame:
    """Para cada coluna indicada, cria uma coluna com a representação de embeddings de cada elemento."""

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
    """Gera um dicionário contendo como chave categorias e como valor a média dos embeddings de todos os elementos
    do data frame para a categoria.
    """

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
    """Adiciona, para cada categoria, qual a similaridade entre os elementos dela e os embeddings da coluna indicada"""

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
    """Gera um dicionário aninhado contendo nome da coluna, estatística e valor"""

    return (base_frame
            [columns]
            .agg({np.median,
                  np.mean,
                  np.std
                  })
            .to_dict()
            )


def fill_missing_numeric_values(base_frame: pd.DataFrame, numeric_stats_dict: dict,
                                statistics: str = 'median') -> pd.DataFrame:
    """Função para preencher valores ausentes com dicionário contendo estatísticas de colunas"""

    filled_frame = base_frame.copy()
    for column in numeric_stats_dict.keys():
        filled_frame[column] = filled_frame[column].apply(lambda v: numeric_stats_dict['weight'][statistics])

    return filled_frame


def create_feature_matrix(base_frame: pd.DataFrame, feature_columns: List[str],
                          embeddings_columns: List[str]) -> np.array:
    """Função para extrair e formatar features em uma matriz para treinamento/inferência"""

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
    """Função para criar dicionários e listas de referência para processar dados de treinamento e de teste"""

    categories = sorted(base_frame['category'].unique().tolist())
    columns_to_copy = ['category'] + embeddings_columns + numeric_columns

    # Limpar e gerar embeddings de colunas textuais selecionadas
    features_frame = (base_frame
                      [columns_to_copy]
                      .copy()
                      .fillna({c: '' for c in embeddings_columns})
                      )

    for feature in embeddings_columns:
        features_frame[feature] = features_frame[feature].apply(text_preprocessing_fn)

    embeddings_frame = compute_embeddings_frame(base_frame, embeddings_columns)
    features_frame = pd.concat([features_frame, embeddings_frame], axis=1)

    # Gerar dicionário com os embeddings médios de cada categoria dos dados de treinamento
    category_embeddings_dict = compute_average_embeddings(features_frame, embeddings_columns, categories)

    # Gerar dicionário com estatísticas de valores numéricos para fazer imputação
    numerics_stats_dict = compute_stats_for_numeric_values(base_frame, numeric_columns)

    return categories, category_embeddings_dict, numerics_stats_dict


def preprocess_features(base_frame: pd.DataFrame,
                        categories: List[str],
                        category_embeddings: Dict,
                        numeric_stats: Dict,
                        numeric_features: List[str],
                        text_features: List[str],
                        similarity_features: List[str],
                        text_preprocessing_fn: partial
                        ) -> pd.DataFrame:
    """Função para fazer pre-processamento das features para treinamento, validação e teste"""

    columns_to_copy = ['category'] + numeric_features + text_features
    features_frame = (base_frame
                      [columns_to_copy]
                      .copy()
                      .fillna({c: '' for c in text_features})
                      )

    for feature in text_features:
        features_frame[feature] = features_frame[feature].apply(text_preprocessing_fn)

    # Calcular embeddings para colunas textuais
    embeddings_frame = compute_embeddings_frame(features_frame, text_features)
    features_frame = pd.concat([features_frame.drop(columns=text_features), embeddings_frame], axis=1)

    # Adicionar similaridade
    for feature in similarity_features:
        category_similarity_frame = compute_column_category_similarity(features_frame, category_embeddings,
                                                                       feature)
        features_frame = pd.concat([features_frame, category_similarity_frame], axis=1)

    # Preencher valores numéricos
    features_frame = fill_missing_numeric_values(features_frame, numeric_stats)

    return features_frame
