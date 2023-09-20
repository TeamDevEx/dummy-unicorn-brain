from api.v1_0.usecases.summary.models import SummarizationRequest, KnowledgeAssistRequest
from utils.milvus_helpers import AIAMilvus, MilvusHelper
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from api.v1_0.usecases.summary.prompt import default_stuff_prompt, default_map_prompt, default_combine_prompt
from api.v1_0.usecases.summary.prompt import milvus_stuff_prompt, milvus_map_prompt, milvus_combine_prompt
from api.v1_0.usecases.summary.prompt import chat_stuff_prompt, chat_map_prompt, chat_combine_prompt
from api.v1_0.usecases.summary.prompt import ka_stuff_prompt, ka_map_prompt,ka_combine_prompt
from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Dict
from api.v1_0.usecases.base.utility import num_tokens_from_messages 
import time 
import logging 
from fastapi import HTTPException
from utils.config import Config
#Additional Imports for Knowledge Assist
from utils import milvus_helpers
from fastapi import HTTPException

def summarization_chain(query, temperature, stuff_prompt, map_prompt, combine_prompt):
    start_time = time.time()

    model_token_dict = {
                        'telus-gpt-3_5' : 8192,
                        'telus-gpt-3_5-16k' : 16000
                    }
    
    llm = AzureChatOpenAI(**{
                                'openai_api_base': Config.fetch('azure-openai-api-base'),
                                'deployment_name': Config.fetch('azure-deployment-name'),
                                'openai_api_key': Config.fetch('azure-openai-api-key'),
                                'openai_api_version': Config.fetch('azure-openai-api-version')},
                                temperature = temperature, 
                                streaming = False,
                                max_tokens = 1024
                                )
    if Config.fetch('azure-deployment-name') in model_token_dict == True:
        token_thresh = model_token_dict[Config.fetch('azure-deployment-name')]
    else:
        token_thresh = 4000

    if num_tokens_from_messages(str(query)) <= token_thresh: # Change this based on model 4k vs 16k
        summary_type = "Stuff Summary"
        with get_openai_callback() as cb:
            chain = load_summarize_chain(llm, chain_type="stuff", prompt = stuff_prompt)
            summary = chain.run(query)
        token_count=cb
    else:
        summary_type = "Map Reduce Summary"
        with get_openai_callback() as cb:
            chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt = map_prompt, combine_prompt = combine_prompt)
            summary = chain.run(query)
        token_count=cb

    return {
                "query" : query,
                "summary" : summary,
                "summary_type": summary_type,
                "elapsed:" : time.time() - start_time,
                "token count" : token_count
            }   
    
def handle_Summary_Request(
                            queryObject: SummarizationRequest,
                            milvus_helper: MilvusHelper,
                            collections: list[str]
                            ) -> Dict:
    
    if collections and queryObject.collection_name not in collections and 'all' not in collections:
        raise HTTPException(status_code=404, detail="Client does not have access to this collection")
    
    splitter = RecursiveCharacterTextSplitter()
    try:
        #Milvus Summarization
        if queryObject.collection_name and queryObject.content_filter:
            logging.info(f"Querying from {queryObject.collection_name}")
            res = milvus_helper.query_collection( collection_name=queryObject.collection_name,
                                                    expr=queryObject.content_filter,
                                                    output_fields=['pk', 'text'])
            sorted_list = sorted(res, key=lambda x: x['pk'])
            docs = [Document(page_content=r['text']) for r in sorted_list]
            query = docs
        #Input Text Summarization
        else: 
            text = queryObject.query
            string = [Document(page_content=text)]
            query = splitter.split_documents(string)
        #Prompt Type
        if queryObject.prompt_type == "milvus":
            stuff_prompt = milvus_stuff_prompt
            map_prompt = milvus_map_prompt
            combine_prompt = milvus_combine_prompt
        if queryObject.prompt_type == "call_center":
            stuff_prompt = chat_stuff_prompt
            map_prompt = chat_map_prompt
            combine_prompt = chat_combine_prompt
        else:
            queryObject.prompt_type == "default"
            stuff_prompt = default_stuff_prompt
            map_prompt = default_map_prompt
            combine_prompt = default_combine_prompt
        
        summary = summarization_chain(query, queryObject.temperature, stuff_prompt, map_prompt, combine_prompt)

        return summary

    except Exception as e:
        logging.error(f"Error in handle_upload_request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#KNOWLEDGE ASSIST TESTING

def handle_knowledge_assist_request(
        queryObject: KnowledgeAssistRequest,
        milvus_helper: MilvusHelper,
        embedding_func: Embeddings,
        collections: list[str],
        firestore_doc_ref: Dict = {} ,
        ) -> Dict:
    
    milvus_collection_name = "onesource_latest"
    if collections and milvus_collection_name not in collections and 'all' not in collections:
        raise HTTPException(status_code=404, detail="Client does not have access to this collection")
    
    splitter = RecursiveCharacterTextSplitter()
    onesource_search_index = milvus_helpers.AIAMilvus(
                                    embedding_function=embedding_func,
                                    collection_name = firestore_doc_ref['collection_name'],
                                    milvus_helper=milvus_helper                                
                                )
    try: 
        #Process Query
        start_time = time.time()
        text = queryObject.query
        string = [Document(page_content=text)]
        log = splitter.split_documents(string)
        #STILL NEED TO CREATE MAP AND COMBINE PROMPTS FOR KNOWLEDGE ASSIST
        summary = summarization_chain(log, queryObject.temperature, ka_stuff_prompt, ka_map_prompt, ka_combine_prompt)
        #Similarity search component
        search_kwargs={
        'query': summary["summary"],
        'k': queryObject.max_num_results,
        }
        docs_and_scores = onesource_search_index.similarity_search_with_score(**search_kwargs)
    except NameError as err:
        raise HTTPException(status_code=404, detail=err.__str__())
    except ValueError as err:
        raise HTTPException(status_code=400, detail=err.__str__())
    except Exception as err:
        raise HTTPException(status_code=500, detail=err.__str__())
    
    parsed_docs_and_scores = [{"document": doc, "distance": score} for doc, score in docs_and_scores if score < queryObject.max_distance]

    return {
            "Call Summary" : summary,
            "documents" : parsed_docs_and_scores,
            "elapsed:" : time.time() - start_time,
            "token count" : summary.get("token count", {})
        }

