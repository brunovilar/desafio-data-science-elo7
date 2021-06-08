import sys
import json
import argparse
import funcy as fp
from pathlib import Path
from pydantic import validate_arguments, ValidationError, Field, Json
from pydantic.typing import Annotated

sys.path.append(str(Path.cwd()))

from src.entities import Product
from src.pipeline.inference_pipeline import make_batch_predictions, make_supervised_intent_classification
from src.pipeline.recommendation import make_recommendations_for_query


@validate_arguments()
def classify_product(product: Annotated[Json, {"format": "json-string"}]) -> None:
    product_obj = Product(**product)
    product_obj = fp.first(make_batch_predictions([product_obj]))
    print(product_obj.category)


@validate_arguments()
def classify_query(query: str) -> None:
    query_intent = fp.first(make_supervised_intent_classification([query]))
    print(query_intent)


@validate_arguments()
def recommend(query: str) -> None:
    recommended_products = make_recommendations_for_query(query)
    for product in recommended_products:
        print(f' - Product ID: {product.product_id} | Title: {product.title}')


def main():

    task_function_map = {
        "classify": classify_product,
        "classify_query": classify_query,
        "recommend": recommend
    }

    parser = argparse.ArgumentParser(add_help=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--category", dest='classify', default=None, type=str, help="Classify a product into a category")
    group.add_argument("--intent", dest='classify_query', default=None, type=str, help="Define the intent of a query")
    group.add_argument("--recommendation", dest='recommend', type=str, help="Make recommendations for a query")

    parsed_args = parser.parse_args().__dict__
    task, parameter = fp.first([(key, value)
                                for key, value in parsed_args.items()
                                if value
                                ])
    task_fn = task_function_map.get(task)

    if task_fn:
        try:
            task_fn(parameter)

        except ValidationError as e:
            validation_issues = ' '.join([
                f'{error["loc"][0]}: {error["msg"]}'
                for error in e.errors()
            ])
            print(f'Problemas encontrados nos dados de entrada: {validation_issues}')
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
