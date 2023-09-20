from api.v1_0.usecases.pnc.models import PnCBotQuery
from utils.milvus_helpers import AIAMilvus, MilvusHelper
from api.v1_0.usecases.base.helpers import handle_default_bot_request
from api.v1_0.usecases.pnc.prompt import CONDENSE_QUESTION_PROMPT as PNC_CONDENSE_QUESTION_PROMPT
from api.v1_0.usecases.pnc.prompt import CHAT_PROMPT as PNC_CHAT_PROMPT
from langchain.embeddings.base import Embeddings
from typing import Callable

def handle_pnc_request(
    queryObject: PnCBotQuery,
    milvus_helper: MilvusHelper,
    embedding_func: Embeddings,
) -> Callable:
    search_kwargs={
        'k': 3,
    }

    pnc_search_index = AIAMilvus(
        embedding_function=embedding_func,
        collection_name = "pnc_recursiveSearch",
        milvus_helper=milvus_helper
    )
    generate_response_with_callback = handle_default_bot_request(
        queryObject=queryObject,
        search_index=pnc_search_index,
        search_kwargs=search_kwargs,
        condense_question_prompt=PNC_CONDENSE_QUESTION_PROMPT,
        qa_prompt=PNC_CHAT_PROMPT
    )
    return generate_response_with_callback
