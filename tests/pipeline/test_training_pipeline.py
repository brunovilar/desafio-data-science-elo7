import pytest
import numpy as np
import pandas as pd
from itertools import product
from typing import List
from collections.abc import Iterable
from src.pipeline.training_pipeline import (compute_embeddings_frame,
                                            compute_average_embeddings,
                                            compute_stats_for_numeric_values)


all_text_columns = ['column_empty', 'column_letter',
                    'column_word', 'column_sentence']
some_text_columns = ['column_empty', 'column_sentence', 'column_letter']
single_text_column = ['column_empty']

all_numeric_columns = ['direct_zero_average', 'indirect_zero_average',
                       'positive_average', 'negative_average', 'same_positive']

no_column = []


@pytest.fixture
def text_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {'column_empty': ['', None, np.nan],
         'column_letter': ['a', 'b', 'c'],
         'column_word': ['first', 'second', 'third'],
         'column_sentence': ['first sentence', 'second sentence', 'third sentence'],
         'category': ['gift', 'grossery', 'gift']
         }
    )


@pytest.fixture
def numeric_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {'direct_zero_average_embedding': [0, 0, 0, 0],
         'indirect_zero_average_embedding': [-50, 100, 50, -100],
         'positive_average_embedding': [100, 50, 150, 140],
         'negative_average_embedding': [-100, 50, 150, -140],
         'same_positive_embedding': [15, 15, 15, 15],
         'compound_embedding': [[5, -15], [15, -5], [8, -3], [12, -17]],
         'category': ['cat_a', 'cat_a', 'cat_a', 'cat_a']
         }
    )


@pytest.fixture
def text_with_embeddings_frame():
    frame = text_frame()
    return compute_embeddings_frame(frame, all_text_columns)


@pytest.mark.parametrize("columns_to_encode", [all_text_columns, some_text_columns, single_text_column, no_column])
def test_compute_embeddings_frame(text_frame: pd.DataFrame, columns_to_encode: List[str]):
    result_frame = compute_embeddings_frame(
        text_frame, columns=columns_to_encode)

    # Check columns order and quantity consistency
    result_columns = result_frame.columns.tolist()
    assert len(result_columns) == len(
        columns_to_encode), 'The expected number of embeddings columns is wrong'
    assert columns_to_encode == [c.replace(
        '_embedding', '') for c in result_columns], 'The expected columns are different'

    # Check columns content
    for column in columns_to_encode:
        if column != 'column_empty':
            assert all(result_frame[f'{column}_embedding'].apply(
                lambda e: np.sum(e, axis=0) != 0)), 'The non-empty embeddings is zero'
        else:
            assert all(result_frame[f'{column}_embedding'].apply(
                lambda e: np.sum(e, axis=0) == 0)), 'The empty embeddings is not zero'

        assert all(result_frame[f'{column}_embedding'].apply(
            lambda e: e.shape[0] > 1)), 'The embeddings shape is wrong'

    # Check the absense of the column
    assert 'category' not in result_columns and f'category_embedding' not in result_columns


@pytest.mark.parametrize("columns", 
                         [all_numeric_columns + ['compound'], no_column])
@pytest.mark.parametrize("categories", [['cat_a'], []])
def test_compute_average_embeddings(numeric_frame, columns, categories):

    expected_answers = {'direct_zero_average': 0,
                        'indirect_zero_average': 0,
                        'positive_average': 110,
                        'negative_average': -10,
                        'same_positive': 15,
                        'compound': [10, -10]
                        }

    average_dict = compute_average_embeddings(
        numeric_frame, columns, categories)

    assert len(average_dict) == (len(columns) * len(categories)), \
        "Unexpected number of result elements"

    for (category, column), value in average_dict.items():
        assert category in categories, 'Category used was not expected'
        assert column in expected_answers, 'Column used is was not expected'

        if isinstance(value, Iterable):
            assert all(value == expected_answers.get(column)), \
            'Wrong value computed'
        else:
            assert value == expected_answers.get(column), 'Wrong value computed'


@pytest.mark.parametrize("columns", [all_numeric_columns])
@pytest.mark.parametrize("stats", [['mean'], ['mean', 'min']])
def test_compute_stats_for_numeric_values(numeric_frame, columns, stats):

    expected_answers = {'direct_zero_average': {'mean': 0, 'min': 0},
                        'indirect_zero_average': {'mean': 0, 'min': -100},
                        'positive_average': {'mean': 110, 'min': 50},
                        'negative_average': {'mean': -10, 'min': -140},
                        'same_positive': {'mean': 15, 'min': 15}
                        }

    columns_to_rename = {c: c.replace('_embedding', '')
                         for c in numeric_frame.columns}
    numeric_frame = numeric_frame.rename(columns=columns_to_rename)

    stats_dict = compute_stats_for_numeric_values(numeric_frame, columns)

    assert len(stats_dict) == len(columns), \
        "Unexpected number of result elements"

    for (column, internal_dict) in stats_dict.items():
        assert column in expected_answers, 'Column used is was not expected'
        column_stats = expected_answers.get(column)
        for expected_stat, expected_value in column_stats.items():
            assert expected_value == internal_dict.get(expected_stat),\
                 'Wrong value computed'
