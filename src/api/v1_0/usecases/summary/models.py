# add src for local imports
import os, sys
absolute_path = os.path.dirname(__file__)
relative_path = "../../../../"
full_path = os.path.realpath(os.path.join(absolute_path, relative_path))
sys.path.append(full_path)

# python imports
from typing import Union, Dict, Optional
import json

# imports
from pydantic import BaseModel, validator, Field
from utils.common_imports import is_valid_content_filter
from utils.usecases import onesource_bot_helper

class SummarizationRequest(BaseModel):
    query: Optional[str]
    temperature: float #Default is 0.4 # overrides creativity_level, range 0-1
    collection_name: Optional[str]
    content_filter: Optional[str]
    prompt_type: str = "default"

    @validator('temperature', pre=True, always=True)
    def check_temperature(cls, temperature):
        if temperature is not None and not 0 <= temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        return temperature
    
    @validator('content_filter')
    def validate_content_filter(cls, content_filter):
        if content_filter is not None and not is_valid_content_filter(content_filter):
            raise ValueError("Invalid content filter. Parentheses are not balanced.")
        return content_filter
    
#KNOWLEDGE ASSIST TESTING

class KnowledgeAssistRequest(BaseModel):
    query: str
    temperature: float = 0.7
    max_num_results: int = 4
    max_distance: float = 1.0 # 1.0 is the maximum distance for OpenAI embeddings. This might change if we change the embedding model.

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
    
    @validator('temperature', pre=True, always=True)
    def check_temperature(cls, temperature):
        if temperature is not None and not 0 <= temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        return temperature
    