from oxpecker.schema.base import BaseSchema
from pydantic import Field

class Document(BaseSchema):
    
    doc_content : str
    metadata: dict = Field(default_factory=dict)