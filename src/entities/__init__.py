from pydantic import validator
from pydantic.dataclasses import dataclass
from datetime import date

CATEGORIES = ['Bebê', 'Bijuterias e Jóias', 'Decoração', 'Lembrancinhas', 'Outros', 'Papel e Cia']


class PydanticConfig:
    validate_assignment = True
    error_msg_templates = {
        'type_error.float': 'Tipo incorreto de dados. Deveria ser número.',
        'type_error.int': 'Tipo incorreto de dados. Deveria ser número (inteiro).'
    }


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=False, frozen=False, config=PydanticConfig)
class Product(object):
    title: str = None
    concatenated_tags: str = None
    price: float = None
    weight: float = None
    express_delivery: bool = None
    minimum_quantity: int = None
    category: str = None
    creation_date: date = None
    product_id: int = None

    @validator('title')
    def required_field(cls, v):
        assert v is not None, 'O campo é obrigatório.'
        return v

    @validator('title')
    def non_empty_sentence(cls, v):
        assert v is None or len(v.strip()) > 0, 'Um conteúdo deve ser definido.'
        return v

    @validator('weight', 'price', 'product_id', 'minimum_quantity')
    def positive_value(cls, v):
        assert v is None or v >= 0, 'O valor de deve ser um número positivo.'
        return v

    @validator('minimum_quantity', 'product_id')
    def integer_value(cls, v):
        assert (v is None
                or isinstance(v, int)
                or (isinstance(v, str) and v.isdecimal())), f'O valor deve ser um número inteiro.'
        return v

    @validator('category')
    def category_values(cls, v):
        assert v is None or v in CATEGORIES, 'Categoria não identificada.'
        return v
