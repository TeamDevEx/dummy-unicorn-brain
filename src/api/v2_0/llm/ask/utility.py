
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from typing import Optional, Any
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.schema import BaseRetriever
from langchain.base_language import BaseLanguageModel
from langchain.prompts.base import BasePromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatVertexAI,ChatOpenAI,AzureChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# our custom retrieval chain allows more customization
class CutomConversationalRetrievalChain(ConversationalRetrievalChain):
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        condense_llm: Optional[BaseLanguageModel],
        retriever: BaseRetriever,
        condense_question_prompt: BasePromptTemplate = CONDENSE_QUESTION_PROMPT,
        qa_prompt: Optional[BasePromptTemplate] = None, # picks up a default from chain_type
        chain_type: str = "stuff",
        **kwargs: Any,
    ) -> BaseConversationalRetrievalChain:
        """Load chain from LLM."""

        if condense_llm is None:
            condense_llm = llm

        doc_chain = load_qa_chain(
            llm,
            chain_type=chain_type,
            prompt=qa_prompt,
        )
        condense_question_chain = LLMChain(llm=condense_llm, prompt=condense_question_prompt)
        return cls(
            retriever=retriever,
            combine_docs_chain=doc_chain,
            question_generator=condense_question_chain,
            **kwargs,
        )


from pydantic import BaseModel
from datetime import datetime
import json

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.dict()
        if isinstance(obj, datetime):   
            return obj.isoformat()
        return super().default(obj)

def custom_json_dumps(obj):
    return json.dumps(obj, cls=CustomJSONEncoder)


# send streaming response to the user
from api.v2_0.llm.ask.models import QABotBaseQuery
from typing import Callable
from api.v2_0.helpers import generator_from_callback
from fastapi.responses import StreamingResponse
from utils.bq_logging import bq_logging 
import logging

def log_to_bq(
        bq: Optional[bq_logging] = None,
        table_id: Optional[str] = None,
        queryObject: Optional[QABotBaseQuery] = None,
        res: Optional[dict] = None
    ):
    
    if bq is not None:
        try:
            row_data = {
                "request_id" : res["request_id"],
                "partition_dt" : datetime.utcnow().date().strftime("%Y-%m-%d"),
                "request" : custom_json_dumps(queryObject),
                "response" : custom_json_dumps(res)
            }
            errors = bq.insert_rows(table_id, [row_data])
            if errors == []:
                logging.info(f"request_id: {res['request_id']} insert into: {table_id}")
            else:
                logging.info(f"request_id: {res['request_id']} Encountered errors while inserting rows: {errors}")
        except Exception as e:
            res = {
                    "request_id" : 0,
                    "error" : str(e)
                   }
            row_data = {
                "request_id" : res["request_id"],
                "partition_dt" : datetime.utcnow().date().strftime("%Y-%m-%d"),
                "request" : custom_json_dumps(queryObject),
                "response" : custom_json_dumps(res)
            }
            logging.info(f"Error inserting into: {table_id}: {str(e)}")

def send_response_from_generate_func(
    queryObject: QABotBaseQuery,
    generate_response_with_callback: Callable,
    table_id: Optional[str] = None,
    bq: Optional[bq_logging] = None, # Only log when deployed, not locally
):
    def send_progress_update():
        try:
            response = generator_from_callback(generate_response_with_callback)
            for res in response:
                yield (custom_json_dumps(res) + "\n")
            log_to_bq(bq, table_id, queryObject, res)
        except UnboundLocalError as e: # Handle sporadic error where UnboundLocalError with res
            res = {"request_id" : 0,
                   "error" : str(e)}
            log_to_bq(bq, table_id, queryObject, res)
            logging.info(f"Error inserting into: {table_id}: {str(e)}")

    if queryObject.stream:
        return StreamingResponse(send_progress_update(), media_type='application/json')
    
    res = generate_response_with_callback(None)
    log_to_bq(bq, table_id, queryObject, res)
    
    return res 

import tiktoken
def num_tokens_from_messages(messages):
    """Return the number of tokens used by a list of messages. : Ref: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb"""
    
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(messages))

def get_token_usage(llm_result,prompt_input_parameters,prompt_type="llm_chat"):
    
    if type(llm_result.llm) in [ChatOpenAI, AzureChatOpenAI]:
        chat_llm_output = llm_result.generate([prompt_input_parameters])
        return chat_llm_output.llm_output["token_usage"]
    else:
        if prompt_type=='llm_chat':
            chat_llm_prompt = llm_result.prompt.format_prompt(**prompt_input_parameters).to_messages()
            prompt_message = chat_llm_prompt[0].content + chat_llm_prompt[1].content
        elif prompt_type=='condense_llm_chat':
            prompt_message  = llm_result.prompt.format_prompt(**prompt_input_parameters).to_string()
        else:
            raise NameError("unknown_prompt_type")
        llm_response = llm_result.generate([prompt_input_parameters]).generations[0][0].text
        prompt_token = num_tokens_from_messages(prompt_message)
        completion_tokens = num_tokens_from_messages(llm_response)
        return {"completion_tokens" :completion_tokens , "prompt_tokens":prompt_token , "total_tokens" : prompt_token+completion_tokens }
