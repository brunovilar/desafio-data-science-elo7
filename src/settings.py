import os
from pathlib import Path

SRC_PATH = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
RAW_DATASET_PATH = 'https://elo7-datasets.s3.amazonaws.com/data_scientist_position/elo7_recruitment_dataset.csv'
DATA_PATH = os.path.join(os.pardir, 'data')
MODELS_PATH = Path(SRC_PATH).parent.joinpath('models')

TRACKING_URI = os.path.join('notebooks', 'mlruns')
LOGS_ARTIFACTS_PATH = os.path.join(os.pardir, 'data', 'log')
WORDS_EMBEDDINGS_MODEL_SOURCE = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.bin.gz'

CATEGORY_CLASSIFICATION_RUN_ID = '2f2ef8c238084aa0b850353d2fa98e57'  # Identifies the 'production' model
