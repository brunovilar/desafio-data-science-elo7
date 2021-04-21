import os

RAW_DATASET_PATH = 'https://elo7-datasets.s3.amazonaws.com/data_scientist_position/elo7_recruitment_dataset.csv'
DATA_PATH = os.path.join(os.pardir, 'data')
MODELS_PATH = os.path.join(os.pardir, 'models')

LOGS_ARTIFACTS_PATH = os.path.join(os.pardir, 'data', 'log')
WORDS_EMBEDDINGS_MODEL_SOURCE = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.bin.gz'
