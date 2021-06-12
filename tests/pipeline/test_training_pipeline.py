import pytest
import numpy as np
import pandas as pd
from typing import List
from src.pipeline.training_pipeline import (compute_embeddings_frame,
                                            compute_average_embeddings)


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
def embeddings_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {'direct_zero_average_embedding': [0, 0, 0, 0],
         'indirect_zero_average_embedding': [-50, 100, 50, -100],
         'positive_average_embedding': [100, 50, 150, 140],
         'negative_average_embedding': [-100, 50, 150, -140],
         'same_positive_embedding': [15, 15, 15, 15],
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


@pytest.mark.parametrize("columns", [all_numeric_columns, no_column])
@pytest.mark.parametrize("categories", [['cat_a'], []])
def test_compute_average_embeddings(embeddings_frame, columns, categories):

    expected_answers = {'direct_zero_average': 0,
                        'indirect_zero_average': 0,
                        'positive_average': 110,
                        'negative_average': -10,
                        'same_positive': 15
                        }

    average_dict = compute_average_embeddings(
        embeddings_frame, columns, categories)

    assert len(average_dict) == (len(columns) * len(categories)), \
        "Unexpected number of result elements"

    for (category, column), value in average_dict.items():
        assert category in categories, 'Category used was not expected'
        assert column in expected_answers, 'Column used is was not expected'
        assert value == expected_answers.get(column), 'Wrong value computed'
