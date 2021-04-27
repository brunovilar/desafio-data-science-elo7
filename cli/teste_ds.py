import os
import sys
import json
import argparse
import funcy as fp

SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, os.pardir)))

from src.entities import Product
from src.pipeline.inference_pipeline import make_batch_predictions


def classify_product(product: str) -> None:
    print(f'Classify Product: {product}')
    product_obj = Product(**json.loads(product))
    product_obj = fp.first(make_batch_predictions([product_obj]))
    print(product_obj.category)


def classify_query(query: str) -> None:
    print(f'Classify Query: {query}')


def recommend(query: str) -> None:
    print(f'Recommend for Query: {query}')


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
        task_fn(parameter)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
