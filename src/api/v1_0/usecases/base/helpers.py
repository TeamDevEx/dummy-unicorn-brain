# add src for local imports
import os, sys
absolute_path = os.path.dirname(__file__)
relative_path = "../../../../"
full_path = os.path.realpath(os.path.join(absolute_path, relative_path))
sys.path.append(full_path)

from api.v1_0.usecases.base.models import QABotBaseQuery, ChatBotHistoryItem, RawChatQuery, ChatBotHistoryItemRaw
from utils.bot_util_lc import FunctionCallbackHandler
from utils.config import Config
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from api.v1_0.usecases.base.utility import CutomConversationalRetrievalChain,get_token_usage, num_tokens_from_messages
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain import OpenAI
from langchain.llms import AzureOpenAI


from langchain.vectorstores.base import VectorStore
from langchain.callbacks.base import BaseCallbackManager
from langchain.schema import LLMResult
from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLLM
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.prompts.base import BasePromptTemplate
from typing import Dict, Union, Optional

from fastapi import HTTPException
import re 
import time
import uuid
from datetime import datetime

def get_default_embedding_func():
    from langchain.embeddings.openai import OpenAIEmbeddings

    if Config.fetch('azure-openai-api-type') == 'azure':
        return OpenAIEmbeddings(
            openai_api_version=Config.fetch('azure-openai-api-version'),
            openai_api_type=Config.fetch('azure-openai-api-type'),
            openai_api_key=Config.fetch('azure-openai-api-key'),
            openai_api_base=Config.fetch('azure-openai-api-base'),
            deployment=Config.fetch('azure-embedding-deployment-name'),
        )
    
    embedding_func = OpenAIEmbeddings(openai_api_key=Config.fetch('openai-api-key'))
    return embedding_func

def get_default_chat_model():
    if Config.fetch('azure-openai-api-type') == 'azure':
        return AzureChatOpenAI
    return ChatOpenAI

def get_default_model():
    if Config.fetch('azure-openai-api-type') == 'azure':
        return AzureOpenAI
    return OpenAI

def get_default_chat_model_config():
    if Config.fetch('azure-openai-api-type') == 'azure':
        return {
            'openai_api_base': Config.fetch('azure-openai-api-base'),
            'openai_api_version': Config.fetch('azure-openai-api-version'),
            'deployment_name': Config.fetch('azure-deployment-name'),
            'openai_api_key': Config.fetch('azure-openai-api-key'),
            'openai_api_type': Config.fetch('azure-openai-api-type'),
        }
    return {
        "model_name": os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo"),
        "openai_api_key": Config.fetch('openai-api-key'),
    }

from openai.error import OpenAIError
from pymilvus.exceptions import MilvusException

def llm_milvus_error_handler(response_object):
    def error_decorator(func):
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result
            except OpenAIError as err:
                response_object['error'] = {
                    'code': err.code,
                    'detail': err.user_message
                }
                return response_object
            except MilvusException as err:
                response_object['error'] = {
                    'code': err._code,
                    'detail': err._message
                }
                return response_object
            except Exception as err:
                response_object['error'] = {
                    'code': err.__cause__,
                    'detail': err.__context__
                }
        return wrapper
    return error_decorator

def answer_unknown_string(input_string):

    processed_string = re.sub(r'\s+', '', input_string.lower())
    processed_string = re.sub(r'[^a-zA-Z0-9]', '', processed_string)

    answer_unknown = (
        re.search(r'idontknow', processed_string) is not None
        or re.search(r'imsorry', processed_string) is not None
        or re.search(r'imnotsure', processed_string) is not None
    )

    return answer_unknown


def handle_raw_bot_request(
    queryObject: RawChatQuery,
    llm_class: Union[BaseLLM, BaseChatModel] = get_default_chat_model(),
    llm_config: Dict = {
        **get_default_chat_model_config(),
    },
):
    res_object_base = {
        'chat_history': queryObject.chat_history,
        'done': False,
    }
    @llm_milvus_error_handler(response_object=res_object_base)  
    def generate_response_with_callback(callback):
        request_id = uuid.uuid4().fields[-1]
        request_time = datetime.utcnow()
        start_time = time.time()

        res_object_base.update({
            'request_id': request_id,
            'request_time': request_time,
        })

        def custom_callback(response):
            res_object = {
                **res_object_base,
                'response': response,
                'elapsed': time.time() - start_time,
            }
            if callback:
                callback(res_object)
            return res_object

        # prepare streaming config
        cb_handler = FunctionCallbackHandler(
            llm_progress_cb=custom_callback
        )
        STREAM_CONFIG = {
            "streaming": True,
            "verbose": True,
            "callback_manager": BaseCallbackManager([cb_handler])
        }

        # base model config
        BASE_CONFIG = {
            "temperature": queryObject.temperature,
            "max_tokens": queryObject.max_new_tokens
        }
        
        # main LLM
        MAIN_LLM_CONFIG = { **BASE_CONFIG, **llm_config }
        if queryObject.stream:
            MAIN_LLM_CONFIG = { **MAIN_LLM_CONFIG, **STREAM_CONFIG }
        chat_llm = llm_class(**MAIN_LLM_CONFIG)

        def convert_history_item(item: ChatBotHistoryItemRaw):
            role_class_map = {
                "Human": HumanMessage,
                "System": SystemMessage,
                "Assistant": AIMessage,
            }
            if item.role not in role_class_map:
                raise Exception("Bad role given!")
    
                # e.g., can be HumanMessage(content=...)
            return role_class_map[item.role](content=item.content)
                        
        res = chat_llm([
            convert_history_item(item) for item in queryObject.chat_history
        ])

        # return for non-streaming, also call the callback for one final update
        return custom_callback(res.content)

    return generate_response_with_callback

def render_prompt_template(prompt, llm):
    prompt_rendered = None
    if prompt is not None:
        if isinstance(prompt, ConditionalPromptSelector):
            prompt_rendered = prompt.get_prompt(llm)
        elif isinstance(prompt, BasePromptTemplate):
            # prompt can be passed in, no need for change
            prompt_rendered = prompt
        else:
            raise Exception('Bad prompt object!')
        
    return prompt_rendered

def handle_default_bot_request(
    queryObject: QABotBaseQuery,
    search_index: VectorStore,
    search_kwargs: Dict = { 'k': 3 },
    main_llm_class: Union[BaseLLM, BaseChatModel] = get_default_chat_model(),
    main_llm_config: Dict = {
        **get_default_chat_model_config(),
        "max_tokens": 256,
    },
    condense_llm_class: Union[BaseLLM, BaseChatModel] = get_default_chat_model(),
    condense_llm_config: Dict = {
        **get_default_chat_model_config(),
        "max_tokens": 256,
    },
    chain_config_override: Dict = {},
    qa_prompt: Optional[Union[ConditionalPromptSelector, BasePromptTemplate]] = None,
    condense_question_prompt: Optional[Union[ConditionalPromptSelector, BasePromptTemplate]] = None,
    max_context_tokens_limit: int = 3000,
    transform_docs_func = None,
    qa_chain_inputs_override: Dict = {},
    res_object_override: Dict = {}
):

    res_object_base = {
        'query': queryObject.query,
        'query_converted': queryObject.query, # default value, if chat_history is present will be replaced
        'qie_status': False,
        'chat_history': queryObject.chat_history,
        'done': False,
        'temp': queryObject.temperature,
        'answer_unknown': False,
        **res_object_override
    }

    @llm_milvus_error_handler(response_object=res_object_base)
    def generate_response_with_callback(callback):
        request_id = uuid.uuid4().fields[-1]
        request_time = datetime.utcnow()
        start_time = time.time()

        res_object_base.update({
            'request_id': request_id,
            'request_time': request_time,
        })

        if queryObject.chat_history == [] or queryObject.chat_history is None:
            # Initialize chat_history to run Condense LLM
            queryObject.chat_history = [ChatBotHistoryItem(**{"role" : "Human", "content" : "Hello Bot"}),
                                        ChatBotHistoryItem(**{"role" : "Assistant", "content" : "Hello User"})]

        def custom_callback(response):
            res_object = {
                **res_object_base,
                'response': response,
                'answer_unknown': answer_unknown_string(response),
                'elapsed': time.time() - start_time,
            }

            if callback:
                callback(res_object)
            return res_object
        
        # prepare streaming config
        cb_handler = FunctionCallbackHandler(
            llm_progress_cb=custom_callback,
        )

        STREAM_CONFIG = {
            "streaming": True,
            "verbose": True,
            "callback_manager": BaseCallbackManager([cb_handler])
        }

        # base model config
        BASE_CONFIG = {
            "temperature": queryObject.temperature,
        }
        
        # main LLM
        MAIN_LLM_CONFIG = { **BASE_CONFIG, **main_llm_config, }
        if queryObject.stream:
            MAIN_LLM_CONFIG = { **MAIN_LLM_CONFIG, **STREAM_CONFIG }
        chat_llm = main_llm_class(**MAIN_LLM_CONFIG)
        
        # condense question LLM
        def on_condense_callback(result: LLMResult):
            response_text = result.generations[0][0].text
            res_object_base.update({
                'query_converted': response_text,
                'qie_status': True
            })
                       
        condense_cb_handler = FunctionCallbackHandler(
            llm_end_cb=on_condense_callback,
        )
        CONDENSE_CONFIG = {
            **BASE_CONFIG,
            **condense_llm_config,
            "verbose": True,
            "callback_manager": BaseCallbackManager([condense_cb_handler])
        }
        condense_chat_llm = condense_llm_class(**CONDENSE_CONFIG)

        def get_chat_history(inputs: list[ChatBotHistoryItem]) -> str:
            res = []
            if inputs:
                for input_item in inputs:
                    res.append(f">> {input_item.role}: {input_item.content}")

                return "\n".join(res)
            
            return ""
        
        qa_prompt_rendered = render_prompt_template(qa_prompt, chat_llm) # returns None by default, picks up a default from chain_type so can be set to None
        condense_question_prompt_rendered = render_prompt_template(condense_question_prompt, condense_chat_llm)  # returns None by default

        if condense_question_prompt_rendered is None:
            condense_question_prompt_rendered = CONDENSE_QUESTION_PROMPT # see CutomConversationalRetrievalChain

        chain_ext_config = {
            "chain_type": "stuff",
            "max_tokens_limit": max_context_tokens_limit, # for content stuffed into the prompt
            "qa_prompt": qa_prompt_rendered,
            "condense_question_prompt": condense_question_prompt_rendered,
            **chain_config_override,
            # --> other options:
            # combine_docs_chain_kwargs=
            # condense_question_prompt=
            # qa_prompt=
        }
        qa_chain = CutomConversationalRetrievalChain.from_llm(
            llm=chat_llm,
            condense_llm=condense_chat_llm,
            retriever=search_index.as_retriever(
                search_kwargs=search_kwargs
            ),
            return_source_documents=True,
            get_chat_history=get_chat_history,
            **chain_ext_config,
        )

        condense_llm_inputs = {"question": queryObject.query,
                                "chat_history": queryObject.chat_history}

        res = qa_chain({
            **condense_llm_inputs,
            **qa_chain_inputs_override # Allows any additional inputs to qa prompt to be passed to qa_chain
        })

        
        # Calculate token usage - Currently only works for chain_type = stuff
        # Condense llm - prompt token count
        condense_prompt = qa_chain.question_generator.prompt.format(**condense_llm_inputs)
        condense_llm_token_count = num_tokens_from_messages(condense_prompt)

        # Chat llm - prompt token count
        inputs = {
            "question": res_object_base['query_converted'],
            **qa_chain_inputs_override # Allows any additional inputs to qa prompt to be passed to qa_chain
        }

        inputs_ = qa_chain.combine_docs_chain._get_inputs(docs=res['source_documents'],**inputs)
        chat_prompt = qa_chain.combine_docs_chain.llm_chain.prompt.format(**inputs_)
        chat_token_cnt = num_tokens_from_messages(chat_prompt)

        # Chat llm - completion token count
        chat_llm_completion_token_cnt = num_tokens_from_messages(res['answer'])
        
        if transform_docs_func is not None:
            res['source_documents'] = transform_docs_func(res['source_documents'])

        converted_docs = [
            {
                'page_content': doc.page_content,
                'metadata': doc.metadata,
            }
            for doc in res['source_documents']
        ]

        res_object_base.update({
            'done': True,
            'token_usage' : {
                            'condense_llm_prompt' : condense_llm_token_count,
                            'chat_llm_prompt' : chat_token_cnt,
                            'chat_llm_completion' : chat_llm_completion_token_cnt
                        }
        })

        if queryObject.return_docs:
            res_object_base.update({
                'source_documents': converted_docs,
            })

        # return for non-streaming, also call the callback for one final update
        return custom_callback(res['answer'])

    return generate_response_with_callback
