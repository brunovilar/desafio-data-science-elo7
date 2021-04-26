import mlflow.pyfunc


class PreprocessingWrapper(mlflow.pyfunc.PythonModel):
    """Classe que atua como um pacote para manter referências utilizadas para pré-processar dados de treinamento"""

    def __init__(self,
                 partial_clean_fn,
                 preprocess_fn,
                 numeric_columns_to_impute,
                 text_columns_to_encode,
                 similarity_features,
                 categories,
                 category_embeddings,
                 numeric_stats,
                 matrix_creation_fn,
                 basic_features,
                 embeddings_features
                 ):

        self.partial_clean_fn = partial_clean_fn
        self.preprocess_fn = preprocess_fn
        self.numeric_columns_to_impute = numeric_columns_to_impute
        self.text_columns_to_encode = text_columns_to_encode
        self.similarity_features = similarity_features
        self.categories = categories
        self.category_embeddings = category_embeddings
        self.numeric_stats = numeric_stats
        self.matrix_creation_fn = matrix_creation_fn
        self.basic_features = basic_features
        self.embeddings_features = embeddings_features

    def predict(self, context, model_input):
        _model_input = model_input.copy()

        features = self.preprocess_fn(_model_input,
                                      self.categories,
                                      self.category_embeddings,
                                      self.numeric_stats,
                                      self.numeric_columns_to_impute,
                                      self.text_columns_to_encode,
                                      self.similarity_features,
                                      self.partial_clean_fn)

        return self.matrix_creation_fn(features, self.basic_features, self.embeddings_features)
