import os, sys
import csv
import json
from datetime import datetime
import uuid
from typing import Callable
from google.cloud import firestore

# add src for local imports
absolute_path = os.path.dirname(__file__)
relative_path = "../../../../"
full_path = os.path.realpath(os.path.join(absolute_path, relative_path))
sys.path.append(full_path)

# import bot_interface modules
from utils.common_imports import *
# setup_http_proxy() # Note: try to use env variables if possible
# load .env
load_dot_env() # Note: Use Config file for loading environment variables

# setup logging
import logging
logging.basicConfig(level=logging.INFO)

from langchain.embeddings.base import Embeddings

from utils import milvus_helpers
from utils.usecases import onesource_bot_helper

from api.v1_0.usecases.onesource.models import OneSourceAPIRequest
from api.v1_0.usecases.base.helpers import handle_default_bot_request, get_default_chat_model_config
from utils.bot_util_lc import get_today_str
from api.v1_0.usecases.onesource.prompt import CONDENSE_QUESTION_PROMPT as ONESOURCE_CONDENSE_QUESTION_PROMPT
from api.v1_0.usecases.onesource.prompt import chat_prompt as ONESOURCE_CHAT_PROMPT 
from typing import Dict
from fastapi import HTTPException



def handle_onesource_request(
        queryObject: OneSourceAPIRequest,
        milvus_helper: milvus_helpers.MilvusHelper,
        embedding_func: Embeddings,
        collections: list[str],
        firestore_doc_ref: Dict = {} ,
) -> Callable:
    
    # Add reading from firestore to get latest collection_name here
    milvus_collection_name = "onesource_latest"

    if collections and milvus_collection_name not in collections and 'all' not in collections:
        raise HTTPException(status_code=404, detail="Client does not have access to this collection")
    
    onesource_search_index = milvus_helpers.AIAMilvus(
                                    embedding_function=embedding_func,
                                    collection_name = firestore_doc_ref['collection_name'],
                                    milvus_helper=milvus_helper                                
                                )
    # Future: Add state variable to keep track of the latest succesful onesource collection
    # Monitor response timing. May need to find aways of improving speed in the future

    expr_filter = onesource_bot_helper.get_milvus_expr_for_site_ids(queryObject.site_ids)

    if queryObject.status_filter in ['current', 'non current', 'expired']:
        expr_filter = f"(metadata_status == \"{queryObject.status_filter}\") && ({expr_filter})"
    
    if queryObject.content_filter:
        expr_filter = f"({queryObject.content_filter}) && ({expr_filter})"
        
    context_config_override = {}
    # Filters in Milvus search
    context_config_override["search_index_kwargs"] = {
        'expr': expr_filter,
        'k' : 8 
    }

    # Set temperature for main llm + callback
    main_llm_config = {
                        **get_default_chat_model_config(),
                        "max_tokens": 256, # Response max token
                        "temperature" : queryObject.temperature,
                        "request_timeout" : 10,                        
        }
    condense_llm_config = {
            **get_default_chat_model_config(),
            "max_tokens": 256,
            "request_timeout" : 10,
        }
    # Add transfrom docs func here to return docs with BU added to page title
    transform_docs_func = lambda docs: onesource_bot_helper.transform_doc_list_with_site_ids(docs, queryObject.site_ids)

    # Additional inputs to qa_prompt for qa chain 
    qa_chain_inputs_override = {"date": get_today_str()}

    # Get last refresh of Milvus DB from firestore for Onesource to return in API
    res_object_override = {"milvus_last_refresh": firestore_doc_ref['data_refresh_time'],
                        
                           "model": main_llm_config['deployment_name']}

    generate_response_with_callback = handle_default_bot_request(
                                    queryObject=queryObject,
                                    search_index=onesource_search_index,
                                    search_kwargs=context_config_override["search_index_kwargs"],
                                    condense_llm_config=condense_llm_config,
                                    main_llm_config = main_llm_config,
                                    transform_docs_func=transform_docs_func,
                                    condense_question_prompt=ONESOURCE_CONDENSE_QUESTION_PROMPT,
                                    qa_prompt=ONESOURCE_CHAT_PROMPT,
                                    qa_chain_inputs_override=qa_chain_inputs_override,
                                    res_object_override = res_object_override
    )

    return generate_response_with_callback

def handle_onesource_tbs_request(
        queryObject: OneSourceAPIRequest,
        milvus_helper: milvus_helpers.MilvusHelper,
        embedding_func: Embeddings,
        collections: list[str],
        firestore_doc_ref: Dict = {} ,
) -> Callable:
    
    # Add reading from firestore to get latest collection_name here

    milvus_collection_name = "tbs_onesource"

    if collections and milvus_collection_name not in collections and 'all' not in collections:
        raise HTTPException(status_code=404, detail="Client does not have access to this collection")
    
    onesource_search_index = milvus_helpers.AIAMilvus(
                                    embedding_function=embedding_func,
                                    collection_name = milvus_collection_name,
                                    milvus_helper=milvus_helper                                
                                )
    # Future: Add state variable to keep track of the latest succesful onesource collection
    # Monitor response timing. May need to find aways of improving speed in the future

    expr_filter = onesource_bot_helper.get_milvus_expr_for_site_ids(queryObject.site_ids)

    if queryObject.status_filter in ['current', 'non current', 'expired']:
        expr_filter = f"(metadata_status == \"{queryObject.status_filter}\") && ({expr_filter})"
    
    if queryObject.content_filter:
        expr_filter = f"({queryObject.content_filter}) && ({expr_filter})"
        
    context_config_override = {}
    # Filters in Milvus search
    context_config_override["search_index_kwargs"] = {
        'expr': expr_filter,
        'k' : 8 
    }

    # Set temperature for main llm + callback
    main_llm_config = {
                        **get_default_chat_model_config(),
                        "max_tokens": 256, # Response max token
                        "temperature" : queryObject.temperature,
                        "request_timeout" : 10,                        
        }
    condense_llm_config = {
            **get_default_chat_model_config(),
            "max_tokens": 256,
            "request_timeout" : 10,
        }
    # Add transfrom docs func here to return docs with BU added to page title
    transform_docs_func = lambda docs: onesource_bot_helper.transform_doc_list_with_site_ids(docs, queryObject.site_ids)

    # Additional inputs to qa_prompt for qa chain 
    qa_chain_inputs_override = {"date": get_today_str()}

    # Get last refresh of Milvus DB from firestore for Onesource to return in API
    res_object_override = {"model": main_llm_config['deployment_name']}

    generate_response_with_callback = handle_default_bot_request(
                                    queryObject=queryObject,
                                    search_index=onesource_search_index,
                                    search_kwargs=context_config_override["search_index_kwargs"],
                                    condense_llm_config=condense_llm_config,
                                    main_llm_config = main_llm_config,
                                    transform_docs_func=transform_docs_func,
                                    condense_question_prompt=ONESOURCE_CONDENSE_QUESTION_PROMPT,
                                    qa_prompt=ONESOURCE_CHAT_PROMPT,
                                    qa_chain_inputs_override=qa_chain_inputs_override,
                                    res_object_override = res_object_override
    )

    return generate_response_with_callback 
