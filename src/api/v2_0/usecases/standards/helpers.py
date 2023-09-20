from api.v2_0.usecases.standards.models import StandardsBotQuery
from utils.milvus_helpers import AIAMilvus, MilvusHelper
from api.v2_0.llm.ask.helpers import handle_default_bot_request
from api.v2_0.usecases.standards import prompt as standards_custom_prompt
from langchain.embeddings.base import Embeddings
from typing import Callable

def handle_standards_request(
    queryObject: StandardsBotQuery,
    milvus_helper: MilvusHelper,
    embedding_func: Embeddings,
) -> Callable:
    search_kwargs={
        'k': 3,
    }

    standards_search_index = AIAMilvus(
        embedding_function=embedding_func,
        collection_name = "topps",
        milvus_helper=milvus_helper
    )
    generate_response_with_callback = handle_default_bot_request(
        queryObject=queryObject,
        search_index=standards_search_index,
        search_kwargs=search_kwargs,
        qa_prompt=standards_custom_prompt.PROMPT_SELECTOR,
    )
    return generate_response_with_callback