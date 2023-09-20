import pandas as pd 

from langchain.chains.question_answering import load_qa_chain
from api.v1_0.usecases.base.helpers import get_default_chat_model, get_default_embedding_func, get_default_chat_model_config
from api.v1_0.usecases.logs_summary.prompt import *
from api.v1_0.usecases.logs_summary.models import LogsSummaryBotQuery

import os, sys 
import itertools

from tqdm import tqdm

from typing import Optional

from google.cloud import bigquery
from langchain.document_loaders import DataFrameLoader

def process_df(df, query_col):
    """
    Function to help with processing the dataframe into the format that the prompt expects
    """
    df['row_num'] = df.index + 1

    def generate_resp_str(x):
        return f"Query[{x['row_num']}]: {x[query_col]}"
    
    df[query_col + '_str'] = df.apply(generate_resp_str, axis=1)

    loader = DataFrameLoader(df[[query_col + '_str', 'row_num']], page_content_column=query_col + '_str' )
    query_docs = loader.load()

    return df, query_docs

def create_batch_str(str_list_to_combine, llm):
    """
    Inputs: 
        - llm: llm 
        - list or pandas series that you want to combine the text by 

    Function will combine each item in the list or series by "\n".join and will return a list of combined str
    """
    batch_query = []
    left_index = 0
    max_token_count = 8000 - 600

    for index in range(1, len(str_list_to_combine) + 1):
        context_candidate = "\n".join(str_list_to_combine[left_index:index])
        token_count = llm.get_num_tokens(context_candidate)
        
        # if max tokens exceeded, parse string
        if token_count > max_token_count:
            print("Hit max token")
            batch_query.append("\n".join(str_list_to_combine[left_index:index-1]))
            left_index = index - 1 
            
            # start next string
            print("Start new string at index {left_index}")
            context_candidate = "\n".join(str_list_to_combine[left_index:index])
            token_count = llm.get_num_tokens(context_candidate)
        # if end of input reached, finish string append
        elif index == len(str_list_to_combine):
            print("Finished appending string")
            batch_query.append("\n".join(str_list_to_combine[left_index:index]))
            
    return batch_query


def summarize_logs_df(df):
    """
    Function to summarize logs of unanswered questions
    Return a txt file that summarizes the topics of unanswered questions
    """
    query_col = 'query'
    df = df.reset_index(drop=True)
    df, query_docs= process_df(df, query_col)
    print(f"Number of queries: {len(query_docs)}")

    def base_chat_llm():
        BASE_CONFIG = {
            "temperature": 0.5,
        }
        main_llm_config = {**get_default_chat_model_config(),
                "deployment_name":"telus-gpt-3_5-16k",
                "max_tokens": 8000
            }

        MAIN_LLM_CONFIG= { 
                    **BASE_CONFIG, 
                    **main_llm_config
                
                }
        
        chat_llm = get_default_chat_model()(**MAIN_LLM_CONFIG)
        return chat_llm

    chat_llm = base_chat_llm()

    # Combine resp_docs to fit within token limit 
    batch_query = create_batch_str(str_list_to_combine=[doc.page_content for doc in query_docs], 
                                llm=chat_llm)
    
    summary_str = ""
    
    # Get summary
    print('Feeding batch responses to LLM')
    for query_str in tqdm(batch_query):
        try:
            _input = prompt.format_prompt(context = query_str) # Using prompt defined above           
            output = chat_llm(_input.to_messages())
            # print(output)

            output_str = output_parser.parse(output.content)['unanswered_query_summary']
            summary_str = summary_str + output_str + "\n\n"
        except Exception as e:
            print('Error while parsing the output')
            print(e)
            summary_str = 'Error while parsing the output: ' + str(e)
            break

    return summary_str

def handle_summarize_unanswered_questions_request(
    queryObject: LogsSummaryBotQuery,
    project_id: Optional[str] = None
):

    bucket_nm = 'unanswered-questions-log-summaries'
    if project_id is None:
        project_id = 'cdo-gen-ai-island-np-204b23'
    
    bq_table = queryObject.bq_table
    date = queryObject.date
    
    folder_name = bq_table.split('.')[-1]
    
    # get df of queries
    client = bigquery.Client(project=project_id)

    sql = f"""
        SELECT JSON_VALUE(request.query) AS query
        FROM `{bq_table}` 
        WHERE partition_dt = '{date}'
        AND JSON_VALUE(response,'$.answer_unknown')="true"
    """

    df = client.query(sql).to_dataframe()
    
    summary_str = summarize_logs_df(df)
    
    # save as txt file and upload to gcs
    from google.cloud import storage

    client = storage.Client(project=project_id)
    bucket = client.get_bucket(bucket_nm)
    blob = bucket.blob(f"{folder_name}/{folder_name}_unanswered_questions_summary_{date}.txt")
    blob.upload_from_string(summary_str)
    
    return {"summary": summary_str}

