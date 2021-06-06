from pathlib import Path

SRC_PATH = str(Path.cwd().joinpath(Path(__file__).parent.expanduser()))

RAW_DATASET_PATH = 'https://elo7-datasets.s3.amazonaws.com/data_scientist_position/elo7_recruitment_dataset.csv'
DATA_PATH = str(Path(SRC_PATH).parent.joinpath('data'))
STOPWORDS_PATH = Path(SRC_PATH).parent.joinpath('resources', 'stopwords.txt')
WORDS_EMBEDDINGS_MODEL_SOURCE = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.bin.gz'

# Word2Vec Embeddings
EMBEDDINGS_MODEL = 'cc.pt.300.bin'

# Model Tracking
MODELS_PATH = str(Path(SRC_PATH).parent.joinpath('models'))
TRACKING_URI = str(Path(SRC_PATH).parent.joinpath('notebooks', 'mlruns'))
LOGS_ARTIFACTS_PATH = str(Path(SRC_PATH).parent.joinpath('data', 'log'))

# Identifies the 'production' models
CATEGORY_CLASSIFICATION_RUN_ID = '1a066c58ad7d4f92ac0c1adfbae31d57'
USUPERVISED_INTENT_CLASSIFICATION_RUN_ID = '01915c9ca4c8440d899a552860bdae1d'
SUPERVISED_INTENT_CLASSIFICATION_RUN_ID = '726952678b9f4cb2949b9707fa66a273'
