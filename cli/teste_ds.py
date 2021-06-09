import sys
import json
import argparse
import funcy as fp
from pathlib import Path
from pydantic import validate_arguments, ValidationError, Field, Json
from pydantic.typing import Annotated

sys.path.append(str(Path.cwd()))
from src.logging import LoggerFactory
from src.entities import Product
from src.pipeline.inference_pipeline import make_batch_predictions, make_supervised_intent_classification
from src.pipeline.recommendation import make_recommendations_for_query

logger = LoggerFactory.get_logger(__name__)


@validate_arguments()
def classify_product(product: Annotated[Json, {"format": "json-string"}]) -> None:
    logger.info(f'Classify Product', extra={'props': {'category':'action_call',
                                                      'input': product}})
    product_obj = Product(**product)
    product_obj = fp.first(make_batch_predictions([product_obj]))
    print(product_obj.category)
    logger.info(f'Classified Product', extra={'props': {'category':'action_result',
                                                        'input_product': product,
                                                        'output_product': product_obj.to_dict()}})


@validate_arguments()
def classify_query(query: str) -> None:
    logger.info(f'Classify Query', extra={'props': {'category':'action_call',
                                                    'input': query}})
    query_intent = fp.first(make_supervised_intent_classification([query]))
    print(query_intent)
    logger.info(f'Classified Query', extra={'props': {'category':'action_result',
                                                      'input': query,
                                                      'query_intent': query_intent}})


@validate_arguments()
def recommend(query: str) -> None:
    logger.info(f'Recommend for Query', extra={'props': {'category':'action_call',
                                                         'input': query}})
    for product in make_recommendations_for_query(query):
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
            logger.exception('Input Validation Exception')
            print(f'Problemas encontrados nos dados de entrada: {validation_issues}')
        except:
            logger.exception('Unexpected Exception')
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
