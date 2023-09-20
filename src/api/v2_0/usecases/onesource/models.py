from typing import Union, Dict
from pydantic import BaseModel, validator

# add src for local imports
import os, sys
absolute_path = os.path.dirname(__file__)
relative_path = "../../../../"
full_path = os.path.realpath(os.path.join(absolute_path, relative_path))
sys.path.append(full_path)

from utils.usecases import onesource_bot_helper
from utils.common_imports import is_valid_content_filter

site_id_options = onesource_bot_helper.get_site_ids()
creativity_temperature_map = {
    'creative': 0.7,
    'balanced': 0.4,
    'precise': 0.1,
}

class ChatBotHistoryItem(BaseModel):
    role: str # options: "Human" "Assistant"
    content: str

    @validator('role')
    def check_role(cls, role):
        if role not in ["Human", "Assistant"]:
            raise ValueError("Invalid role, options are 'Human' and 'Assistant'.")
        return role
    
    @validator('content')
    def check_content(cls, content):
        if len(content) <= 3:
            raise ValueError("content must be longer than 3 characters")
        return content

class OneSourceAPIRequest(BaseModel):
    site_ids: list[int] = site_id_options
    status_filter: str = 'current' # options: "current" "non current" "expired" "all" "automated"
    content_filter: Union[str, None] = None
    creativity_level: Union[str, None] = None # options: "creative" "balanced" "precise"
    temperature: Union[float, None] = 0.4 # Default is 0.4 # overrides creativity_level, range 0-1
    chat_history: Union[list[ChatBotHistoryItem], None]
    query: str
    stream: bool = False # should we stream the response or not?
    return_docs: bool = True # should we return "docs" as well, or not?
    roles: list[int] # user's roles
    types: list[int] # content types
    x_id: str # user's XID/TID
    meta_tags: Union[Dict[str, str], None]

    @validator('status_filter')
    def check_status_filter(cls, status_filter):
        if status_filter not in ["current", "non current", "expired", "all", "automated"]:
            raise ValueError("Invalid status_filter, values must be one of ['current', 'non current', 'expired', 'all']")
        return status_filter
    
    @validator('content_filter')
    def validate_content_filter(cls, content_filter):
        if content_filter is not None and not is_valid_content_filter(content_filter):
            raise ValueError("Invalid content filter. Parentheses are not balanced.")
        return content_filter

    @validator('site_ids')
    def check_site_ids(cls, site_ids):
        if not set(site_ids).issubset(set(site_id_options)):
            raise ValueError(f"Invalid site_ids, values must be in {str(site_id_options)}")
        if len(site_ids) == 0:
            raise ValueError("site_ids must be non-empty")
        return site_ids

    @validator('query')
    def check_query(cls, query):
        if len(query) <= 3:
            raise ValueError("Query must be longer than 3 characters")
        return query

    @validator('temperature', pre=True, always=True)
    def check_temperature(cls, temperature, values):
        creativity_level = values.get('creativity_level')
        if (creativity_level is not None) and (temperature is not None):
            raise ValueError("Provide either creativity_level or temperature, not both")
        if (creativity_level is None) and (temperature is None):
            creativity_level = "balanced" # default config
        if temperature is not None and not 0 <= temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        if creativity_level:
            temperature = creativity_temperature_map[creativity_level]
        return temperature

