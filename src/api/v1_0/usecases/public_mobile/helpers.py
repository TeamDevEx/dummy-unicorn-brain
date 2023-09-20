from api.v1_0.usecases.public_mobile.models import PublicMobileBotQuery
from utils.milvus_helpers import AIAMilvus, MilvusHelper
from api.v1_0.usecases.base.helpers import handle_default_bot_request
from api.v1_0.usecases.public_mobile.prompt import CONDENSE_QUESTION_PROMPT as PM_CONDENSE_QUESTION_PROMPT
from api.v1_0.usecases.public_mobile.prompt import CHAT_PROMPT as PM_CHAT_PROMPT
from langchain.embeddings.base import Embeddings
from typing import Callable
from fastapi import HTTPException

def handle_public_mobile_request(
    queryObject: PublicMobileBotQuery,
    milvus_helper: MilvusHelper,
    embedding_func: Embeddings,
    collections: list[str]
) -> Callable:
    search_kwargs={
        'k': 3,
    }

    milvus_collection_name = "public_mobile"

    if collections and milvus_collection_name not in collections and 'all' not in collections:
        raise HTTPException(status_code=404, detail="Client does not have access to this collection")

    public_mobile_search_index = AIAMilvus(
        embedding_function=embedding_func,
        collection_name = milvus_collection_name,
        milvus_helper=milvus_helper
    )
    generate_response_with_callback = handle_default_bot_request(
        queryObject=queryObject,
        search_index=public_mobile_search_index,
        search_kwargs=search_kwargs,
        condense_question_prompt=PM_CONDENSE_QUESTION_PROMPT,
        qa_prompt=PM_CHAT_PROMPT
    )
    return generate_response_with_callback
