import re
import numpy as np
import pandas as pd
from sklearn import metrics
from typing import List, Dict, Set, Any, Tuple

TAG_SPLIT_PATTERN = re.compile(r"(\s|[-/,;.()])")


def set_dataset_split(frame: pd.DataFrame, cut_off_period: str) -> pd.DataFrame:
    split_frame = (
        frame
        .assign(period=lambda f: f['creation_date'].apply(lambda x: str(x)[:7]))
        .assign(group=lambda f: f['period'].apply(lambda period: 'training' if period <= cut_off_period else 'test'))
    )

    return split_frame


def extract_tokens(frame: pd.DataFrame, raw_column_name: str, split_pattern: re.Pattern) -> Set:
    tags = (frame
            .assign(tag=lambda f: f[raw_column_name].apply(lambda x: split_pattern.split(str(x))))
            .explode('tag')
            .loc[lambda f: f['tag'].apply(lambda x: len(x.strip()) > 0)]
            ['tag']
            .tolist())

    return tags


def compute_multiclass_classification_metrics(y_train: np.array, y_preds: np.array, average=None) -> Dict:

    return {
        'acc': metrics.accuracy_score(y_train, y_preds),
        'precision': metrics.precision_score(y_train, y_preds, average=average),
        'recall': metrics.recall_score(y_train, y_preds, average=average),
        'f1': metrics.f1_score(y_train, y_preds, average=average),
    }


def compute_binary_classification_metrics(y_train: np.array, y_preds: np.array) -> Dict:
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_preds)
    return {
        'acc': metrics.accuracy_score(y_train, y_preds),
        'auc': metrics.auc(fpr, tpr),
        'precision': metrics.precision_score(y_train, y_preds),
        'recall': metrics.recall_score(y_train, y_preds),
        'f1': metrics.f1_score(y_train, y_preds),
        'filtering': sum(y_preds) / len(y_preds)
    }


def format_compound_value(value: Any) -> List:

    basic_types = [int, float, bool]

    if any([isinstance(value, item_type) for item_type in basic_types]):
        return value, False
    else:
        if not isinstance(value, str) and isinstance(value, Iterable):
            formatted_value = ','.join([str(item) for item in value])
        else:
            formatted_value = str(value).replace('\n', '')

        return formatted_value, len(formatted_value) > 250


def format_nested_parameters(param_dict: Dict, param_name: str) -> Tuple[List[Any], List[Any]]:

    preprocessed_params = [(key, *format_compound_value(value)) for key, value in param_dict.items()]

    simple_params = {f'{param_name}__{key}': value if not is_longer else f'Snippet: {value[:235]} (...)'
                     for key, value, is_longer in preprocessed_params}

    complex_params = {f'{param_name}__{key}': value
                      for key, value, is_longer in preprocessed_params
                      if is_longer}

    return simple_params, complex_params
