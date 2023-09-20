from api.v2_0.usecases.milo.models import MiloBotQuery
from utils.milvus_helpers import AIAMilvus, MilvusHelper
from api.v2_0.llm.ask.helpers import handle_default_bot_request
from api.v2_0.usecases.milo.prompt import CONDENSE_QUESTION_PROMPT as MILO_CONDENSE_QUESTION_PROMPT
from api.v2_0.usecases.milo.prompt import CHAT_PROMPT as MILO_CHAT_PROMPT
from langchain.embeddings.base import Embeddings
from typing import Callable
from datetime import datetime
from fastapi import HTTPException

def handle_milo_request(
    queryObject: MiloBotQuery,
    milvus_helper: MilvusHelper,
    embedding_func: Embeddings,
    collections: list[str]
) -> Callable:
    
    milvus_collection_name = "milo_latest"
    
    if collections and milvus_collection_name not in collections and 'all' not in collections:
        raise HTTPException(status_code=404, detail="Client does not have access to this collection")
    
    province_expr_list = [f"metadata_province_{province} == true" for province in queryObject.provinces]
    brand_expr_list = [f"metadata_brand_{brand} == true" for brand in queryObject.brands]
    province_expr_filter = " || ".join(province_expr_list)
    brand_expr_filter = " || ".join(brand_expr_list)
    expr_filter = f"({province_expr_filter}) && ({brand_expr_filter})"
    
    if queryObject.status in ['current', 'expired']:
        expr_filter = f"(metadata_status == \"{queryObject.status}\") && ({expr_filter})"
    

    if queryObject.publish_date:
        date_filter_val = datetime.strptime(queryObject.publish_date, '%Y-%m-%d').timestamp()
        expr_filter = f"(metadata_publish_date >= {date_filter_val}) && ({expr_filter})"

    # print(f"milo expr_filter: {expr_filter}")
    
    search_kwargs={
        'expr': expr_filter,
        'k': 3,
    }

    milo_search_index = AIAMilvus(
        embedding_function=embedding_func,
        collection_name = milvus_collection_name,
        milvus_helper=milvus_helper
    )
    generate_response_with_callback = handle_default_bot_request(
        queryObject=queryObject,
        search_index=milo_search_index,
        search_kwargs=search_kwargs,
        condense_question_prompt=MILO_CONDENSE_QUESTION_PROMPT,
        qa_prompt=MILO_CHAT_PROMPT
    )
    return generate_response_with_callback
