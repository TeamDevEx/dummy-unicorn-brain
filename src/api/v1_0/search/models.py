from pydantic import BaseModel, validator
from typing import Union
from utils.common_imports import is_valid_content_filter

class SearchQuery(BaseModel):
    collection_name: str
    query: str
    max_num_results: int = 4
    max_distance: float = 1.0 # 1.0 is the maximum distance for OpenAI embeddings. This might change if we change the embedding model.
    content_filter: Union[str, None] = None

    @validator('max_num_results')
    def validate_max_num_results(cls, value):
        if not 1 <= value <= 16384:
            raise ValueError('max_num_results must be between 1 and 16384')
        return value
    
    @validator('max_distance')
    def validate_max_distance(cls, value):
        if not 0 <= value <= 1:
            raise ValueError('max_distance must be between 0 and 1')
        return value
    
    @validator('content_filter')
    def validate_content_filter(cls, value):
        if value is not None and not is_valid_content_filter(value):
            raise ValueError('content_filter must be a valid filter')
        return value