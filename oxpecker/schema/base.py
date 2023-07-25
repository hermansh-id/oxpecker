from abc import ABC
from typing import Any, Dict, List, Literal, TypedDict, Union, cast

from pydantic import BaseModel, PrivateAttr

class BaseSchema(BaseModel, ABC):
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)