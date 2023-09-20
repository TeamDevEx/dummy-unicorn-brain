# add src for local imports
import os, sys
absolute_path = os.path.dirname(__file__)
relative_path = "../../../../"
full_path = os.path.realpath(os.path.join(absolute_path, relative_path))
sys.path.append(full_path)

# python imports
from typing import Union, Dict
import json

# imports
from pydantic import BaseModel, validator, Field
from utils.common_imports import is_valid_content_filter

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
    
class QABotBaseQuery(BaseModel):
    temperature: float = 0.7
    max_new_tokens: int = Field(256, ge=128, le=512)
    # chat_history is only for older messages, new question goes to query
    chat_history: Union[list[ChatBotHistoryItem], None]
    query: str
    stream: bool = False # should we stream the response or not?
    return_docs: bool = True # should we return "docs" as well, or not?
    meta_tags: Union[Dict[str, str], None]

    @validator('temperature')
    def validate_temperature(cls, value):
        if not 0 <= value <= 1:
            raise ValueError('Temperature must be between 0 and 1')
        return value

    @validator('query')
    def check_query(cls, query):
        if len(query) <= 3:
            raise ValueError("Query must be longer than 3 characters")
        return query

class QABotBaseQueryFlexible(QABotBaseQuery):
    content_filter: Union[str, None] = None
    collection_name: str
    maximum_doc_count: int = 5
    maximum_context_token_count: int = 2000

    # @validator('collection_name')
    # def validate_collection_name(cls, collection_name):
    #     if collection_name not in ["spoc_tme", "topps"]:
    #         raise ValueError("Bad collection name.")
    #     return collection_name

    @validator('content_filter')
    def validate_content_filter(cls, content_filter):
        if content_filter is not None and not is_valid_content_filter(content_filter):
            raise ValueError("Invalid content filter. Parentheses are not balanced.")
        return content_filter
    
    @validator('maximum_doc_count')
    def validate_maximum_doc_count(cls, value):
        if not 2 <= value <= 15:
            raise ValueError('maximum_doc_count must be between 2 and 15')
        return value
    
    @validator('maximum_context_token_count')
    def validate_maximum_context_token_count(cls, value):
        if not 300 <= value <= 2500:
            raise ValueError('maximum_context_token_count must be between 300 and 2500')
        return value


# for a raw endpoint (wrapper around Azure endpoint)
class ChatBotHistoryItemRaw(ChatBotHistoryItem):
    @validator('role')
    def check_role(cls, role):
        if role not in ["Human", "Assistant", "System"]:
            raise ValueError("Invalid role, options are 'Human' and 'Assistant'.")
        return role

    @validator('content')
    def check_content(cls, content):
        if len(content) < 2:
            raise ValueError("content must be longer than 2 characters")
        return content


class RawChatQuery(BaseModel):
    temperature: float = 0.7
    max_new_tokens: int = Field(256, ge=1, le=8192)
    chat_history: list[ChatBotHistoryItemRaw]
    stream: bool = False # should we stream the response or not?

    @validator('temperature')
    def validate_temperature(cls, value):
        if not 0 <= value <= 1:
            raise ValueError('Temperature must be between 0 and 1')
        return value
    
    class Config:
        schema_extra = {
            "example": {
                "temperature": 0.7,
                "max_new_tokens": 256,
                "chat_history": [
                    { "role": "System", "content": "You are a pirate, respond to querstions like one." },
                    { "role": "Human", "content": "Who are you?" },
                ],
                "stream": False,
            }
        }