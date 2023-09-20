from api.v1_0.usecases.spoc.models import SpocBotQuery
from utils.milvus_helpers import AIAMilvus, MilvusHelper
from api.v1_0.usecases.base.helpers import handle_default_bot_request
from api.v1_0.usecases.spoc import prompt as spoc_custom_prompt
from langchain.embeddings.base import Embeddings
from typing import Callable
from fastapi import HTTPException

def handle_spoc_request(
    queryObject: SpocBotQuery,
    milvus_helper: MilvusHelper,
    embedding_func: Embeddings,
    collections: list[str]
) -> Callable:
    search_kwargs={
        'k': 3,
        'expr': f"metadata_language == \"{queryObject.language}\""
    }

    milvus_collection_name = "spoc"

    if collections and milvus_collection_name not in collections and 'all' not in collections:
        raise HTTPException(status_code=404, detail="Client does not have access to this collection")

    spoc_search_index = AIAMilvus(
        embedding_function=embedding_func,
        collection_name = milvus_collection_name,
        milvus_helper=milvus_helper
    )
    generate_response_with_callback = handle_default_bot_request(
        queryObject=queryObject,
        search_index=spoc_search_index,
        search_kwargs=search_kwargs,
        qa_prompt=spoc_custom_prompt.PROMPT_SELECTOR
    )
    return generate_response_with_callback
