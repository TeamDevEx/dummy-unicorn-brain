from api.v2_0.usecases.t_com.models import TcomBotQuery
from utils.milvus_helpers import AIAMilvus, MilvusHelper
from api.v2_0.llm.ask.helpers import handle_default_bot_request, get_default_chat_model_config
from langchain.embeddings.base import Embeddings
from typing import Callable
from utils.bot_util_lc import get_today_str

from api.v2_0.usecases.t_com.prompt import CONDENSE_QUESTION_PROMPT as TCOM_SUPPORT_CONDENSE_QUESTION_PROMPT
from api.v2_0.usecases.t_com.prompt import QA_PROMPT as TCOM_SUPPORT_QA_PROMPT
from fastapi import HTTPException



def handle_tcom_support_request(
        queryObject: TcomBotQuery,
        milvus_helper: MilvusHelper,
        embedding_func: Embeddings,
        collections: list[str]
) -> Callable:
    search_kwargs={
        'k': 5,
        # Potential metadata filters can be on products / intent
    }

    milvus_collection_name = "tcom_support_articles"

    if collections and milvus_collection_name not in collections and 'all' not in collections:
        raise HTTPException(status_code=404, detail="Client does not have access to this collection")

    tcom_search_index = AIAMilvus(
        embedding_function=embedding_func,
        collection_name = milvus_collection_name,
        milvus_helper=milvus_helper
    )

    # Set temperature for main llm + callback
    main_llm_config = {
                        **get_default_chat_model_config(),
                        "max_tokens": 512, # Response max token
                        "temperature" : queryObject.temperature,
                        "request_timeout" : 10,                        
        }
    condense_llm_config = {
            **get_default_chat_model_config(),
            "max_tokens": 256,
            "request_timeout" : 10,
            "temperature" : queryObject.temperature
        }
    
    # Additional inputs to qa_prompt for qa chain 
    qa_chain_inputs_override = {"date": get_today_str()}

    res_object_override = {"model": main_llm_config['deployment_name']}

    generate_response_with_callback = handle_default_bot_request(
                                                                    queryObject=queryObject,
                                                                    search_index=tcom_search_index,
                                                                    search_kwargs=search_kwargs,
                                                                    condense_llm_config=condense_llm_config,
                                                                    main_llm_config = main_llm_config,
                                                                    condense_question_prompt=TCOM_SUPPORT_CONDENSE_QUESTION_PROMPT,
                                                                    qa_prompt=TCOM_SUPPORT_QA_PROMPT,
                                                                    qa_chain_inputs_override=qa_chain_inputs_override,
                                                                    res_object_override = res_object_override
                                                                )

    return generate_response_with_callback
