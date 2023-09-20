from api.v1_0.search.models import SearchQuery
from utils.milvus_helpers import AIAMilvus, MilvusHelper
from langchain.embeddings.base import Embeddings
from typing import Dict
from fastapi import HTTPException

def handle_search_request(
    queryObject: SearchQuery,
    milvus_helper: MilvusHelper,
    embedding_func: Embeddings,
    collections: list[str]
) -> Dict:
    search_kwargs={
        'query': queryObject.query,
        'k': queryObject.max_num_results,
        'expr': queryObject.content_filter,
    }
    if collections and queryObject.collection_name not in collections and 'all' not in collections:
        raise HTTPException(status_code=404, detail="Client does not have access to this collection")
    
    search_index = AIAMilvus(
        embedding_function=embedding_func,
        collection_name = queryObject.collection_name,
        milvus_helper=milvus_helper
    )
    try:
        docs_and_scores = search_index.similarity_search_with_score(**search_kwargs)
    except NameError as err:
        raise HTTPException(status_code=404, detail=err.__str__())
    except ValueError as err:
        raise HTTPException(status_code=400, detail=err.__str__())
    except Exception as err:
        raise HTTPException(status_code=500, detail=err.__str__())
    
    parsed_docs_and_scores = [{"document": doc, "distance": score} for doc, score in docs_and_scores if score < queryObject.max_distance]
    return parsed_docs_and_scores