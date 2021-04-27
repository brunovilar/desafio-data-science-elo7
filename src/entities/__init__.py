from dataclasses import dataclass
from datetime import date


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=False, frozen=False)
class Product(object):
    title: str
    concatenated_tags: str
    price: float
    weight: float = None
    express_delivery: bool = None
    minimum_quantity: float = None
    category: str = None
    creation_date: date = None
