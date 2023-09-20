from api.v2_0.llm.ask.models import QABotBaseQueryFlexible
from utils.milvus_helpers import AIAMilvus, MilvusHelper
from api.v2_0.llm.ask.helpers import handle_default_bot_request
from api.v2_0.usecases.generic import prompt as generic_custom_prompt
from langchain.embeddings.base import Embeddings
from typing import Callable
from fastapi import HTTPException

def handle_generic_request(
    queryObject: QABotBaseQueryFlexible,
    milvus_helper: MilvusHelper,
    embedding_func: Embeddings,
    collections: list[str]
) -> Callable:
    search_kwargs={
        'k': queryObject.maximum_doc_count,
        'expr': queryObject.content_filter,
    }

    if collections and queryObject.collection_name not in collections and 'all' not in collections:
        raise HTTPException(status_code=404, detail="Client does not have access to this collection")
    
    search_index = AIAMilvus(
        embedding_function=embedding_func,
        collection_name = queryObject.collection_name,
        milvus_helper=milvus_helper
    )
    generate_response_with_callback = handle_default_bot_request(
        queryObject=queryObject,
        search_index=search_index,
        search_kwargs=search_kwargs,
        qa_prompt=generic_custom_prompt.PROMPT_SELECTOR,
        condense_question_prompt=generic_custom_prompt.CONDENSE_QUESTION_PROMPT,
        max_context_tokens_limit=queryObject.maximum_context_token_count
    )
    return generate_response_with_callback
