import time
import json
import os, sys
absolute_path = os.path.dirname(__file__)
relative_path = "../../"
full_path = os.path.realpath(os.path.join(absolute_path, relative_path))
sys.path.append(full_path)
import uuid
from typing import Union, Optional
from typing_extensions import Annotated
from api.v2_0.usecases.onesource.models import OneSourceAPIRequest
from api.v2_0.usecases.onesource import helpers as onesource_api_helpers
from api.v2_0.usecases.summary import helpers as summary_api_helpers
from api.v2_0.llm.ask.models import QABotBaseQuery, QABotBaseQueryFlexible, RawChatQuery
from api.v2_0.usecases.spoc.models import SpocBotQuery
from api.v2_0.usecases.milo.models import MiloBotQuery
from api.v2_0.usecases.pnc.models import PnCBotQuery
from api.v2_0.usecases.public_mobile.models import PublicMobileBotQuery
from api.v2_0.usecases.pso.models import PsoBotQuery
from api.v2_0.usecases.standards.models import StandardsBotQuery
from api.v2_0.usecases.t_com.models import TcomBotQuery
from api.v2_0.usecases.spoc.helpers import handle_spoc_request
from api.v2_0.usecases.milo.helpers import handle_milo_request
from api.v2_0.usecases.pnc.helpers import handle_pnc_request
from api.v2_0.usecases.public_mobile.helpers import handle_public_mobile_request
from api.v2_0.usecases.pso.helpers import handle_pso_request
from api.v2_0.usecases.generic.helpers import handle_generic_request
from api.v2_0.usecases.standards.helpers import handle_standards_request
from api.v2_0.usecases.t_com.helpers import handle_tcom_support_request
from api.v2_0.llm.ask.helpers import handle_default_bot_request, handle_llm_ask_request, get_default_embedding_func
from api.v2_0.usecases.logs_summary.helpers import handle_summarize_unanswered_questions_request
from api.v2_0.usecases.logs_summary.models import LogsSummaryBotQuery
from api.v2_0.llm.ask.utility import custom_json_dumps, send_response_from_generate_func, num_tokens_from_messages, get_token_usage
from api.v2_0.vectordb.search.models import SearchQuery
from api.v2_0.vectordb.search.helpers import handle_search_request
from api.v2_0.usecases.summary.helpers import handle_Summary_Request, handle_knowledge_assist_request
from api.v2_0.helpers import generator_from_callback
from api.v2_0.usecases.summary.models import SummarizationRequest, KnowledgeAssistRequest
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from fastapi_utils.tasks import repeat_every
from utils.config import Config
from utils.bq_logging import bq_logging
import logging
from api.v2_0.vectordb.file import helpers as file_upload_helpers
from google.cloud import firestore
from utils.milvus_helpers import AIAMilvus, MilvusHelper
from utils.gcs_helpers import GCSBucket
import pandas as pd
import tiktoken
import jwt
from google.cloud.firestore_v1 import FieldFilter
logger = logging.getLogger("gunicorn.info")

router = APIRouter(
    prefix = Config.fetch('V2_0_URL')
)

logging_obj = {}
db = {}
gcs = {}
default_embedding_func = get_default_embedding_func()

@router.on_event("startup")
def startup_event():
    if Config.backend=="gsm": # Deployed on GCP
        logging_obj['PROJECT_ID'] = Config.gsm_project
        logging_obj["LOGGING_DATASET_ID"] = os.environ.get("LOGGING_DATASET")
        logging_obj["SPOC_TABLE_ID"] = os.environ.get("SPOC_TABLE")
        logging_obj["ONESOURCE_TABLE_ID"] =  os.environ.get("ONESOURCE_TABLE")
        logging_obj["UPLOAD_DB_TABLE_ID"] = os.environ.get("UPLOAD_DB_TABLE")
        logging_obj["MILO_TABLE_ID"] = os.environ.get("MILO_TABLE")
        logging_obj["PUBLIC_MOBILE_TABLE_ID"] = os.environ.get("PUBLIC_MOBILE_TABLE")
        logging_obj["PSO_TABLE_ID"] = os.environ.get("PSO_TABLE")
        logging_obj['TCOM_TABLE_ID'] = os.environ.get("TCOM_TABLE")
        logging_obj['PNC_TABLE_ID'] = os.environ.get("PNC_TABLE")

        bq = bq_logging(project_id=logging_obj['PROJECT_ID'] )

        # Get Dataset and Table IDs
        usecase_table_ids = [logging_obj["SPOC_TABLE_ID"], 
                             logging_obj["ONESOURCE_TABLE_ID"], 
                             logging_obj["UPLOAD_DB_TABLE_ID"],
                             logging_obj["MILO_TABLE_ID"],
                             logging_obj["PUBLIC_MOBILE_TABLE_ID"],
                             logging_obj["PSO_TABLE_ID"],
                             logging_obj['TCOM_TABLE_ID'],
                             logging_obj['PNC_TABLE_ID']]

        # Create dataset if it does not exist 
        if bq.dataset_exists(logging_obj["LOGGING_DATASET_ID"]) is False:
            bq.create_dataset(logging_obj["LOGGING_DATASET_ID"])
        else:
            logging.info(f"Dataset: {logging_obj['LOGGING_DATASET_ID']} already exists")

        # Create logging datasets + tables if they do not exist yet (only for GCP) for each use case
        for table_id in usecase_table_ids:
            if bq.table_exists(logging_obj["LOGGING_DATASET_ID"], table_id) is False:
                bq.create_log_table(logging_obj["LOGGING_DATASET_ID"], table_id)
            else:
                logging.info(f"Table: {table_id} already exists")

        logging_obj["bq"] = bq
        gcs['gcs'] = GCSBucket(PROJECT_ID=logging_obj['PROJECT_ID'],
                                BUCKET_NAME=os.environ.get("SELF_SERVE_BUCKET"))
                
    else: # Local dev
        logging_obj['PROJECT_ID'] = None
        logging_obj["LOGGING_DATASET_ID"] = None
        logging_obj["SPOC_TABLE_ID"] = None
        logging_obj["ONESOURCE_TABLE_ID"] = None
        logging_obj["UPLOAD_DB_TABLE_ID"] = None 
        logging_obj["MILO_TABLE_ID"] = None 
        logging_obj["PUBLIC_MOBILE_TABLE_ID"] = None
        logging_obj["PSO_TABLE_ID"] = None
        logging_obj['TCOM_TABLE_ID'] = None
        logging_obj['PNC_TABLE_ID'] = None

        logging_obj["bq"] = None
        gcs['gcs'] = None 
        #gcs['gcs'] = GCSBucket(PROJECT_ID='cdo-gen-ai-island-np-204b23', BUCKET_NAME='self-serve-data-cdo-gen-ai-island-np-204b23')
                
        logging.info("Local development: bq_logging object not created")

@router.on_event("startup")
def load_milvus_db():
    db["milvus_helper"] = MilvusHelper(
        connection_args=None, # use environment variables
        bypass_proxy=None, # use environment variables
    )

@router.on_event("startup")
@repeat_every(seconds=60*5) # Get the latest onesource collection name every 5 mins
async def load_onesource_firestore_db():
    doc_ref_backup = {
                "collection_name": "onesource_latest",
                "data_refresh_time" : ""
                }
    # If in prod, use 'milvus_info_pr', else in non-prod or local use 'milvus_info_np'
    MILVUS_COLLECTION_NAME = os.environ.get("MILVUS_COLLECTION_NAME", default="milvus_info_np")
    try:
        firestore_db = firestore.Client(project=logging_obj['PROJECT_ID'])
        doc_ref = firestore_db.collection(MILVUS_COLLECTION_NAME).document('onesource').get().to_dict() # Get the latest collection name
        
        # Add condition here if collection does not exist in milvus, to also use onesource
        if db["milvus_helper"].get_collection(collection_name = doc_ref['collection_name']) is None:
            logging.critical(f"Error loading onesource collection {MILVUS_COLLECTION_NAME} from firestore: {doc_ref['collection_name']}, using backup collection")
            db["firestore_onesource_doc_ref"] = doc_ref_backup
        else:
            logging.info(f"Loading onesource collection {MILVUS_COLLECTION_NAME} from firestore: {doc_ref['collection_name']}")
            db["firestore_onesource_doc_ref"] = doc_ref
    except Exception as e:
        logging.critical(f"Error loading onesource collection {MILVUS_COLLECTION_NAME} from firestore, using backup collection: {e}")
        db["firestore_onesource_doc_ref"] = doc_ref_backup

def get_db_dict():
    return db

def get_logging_dict():
    return logging_obj

def get_gcs():
    return gcs

def get_embedding_func():
    return default_embedding_func

def get_approved_collections(Authorization: Annotated[Union[str, None], Header()] = None):
    if Config.env != "":
        token_decoded = jwt.decode(Authorization.split(' ')[1], options={"verify_signature": False})
        client_id = token_decoded['client_id']
        try:
            ACCESS_CONTROL_COLLECTION_NAME = os.environ.get("ACCESS_CONTROL_COLLECTION_NAME", default="clients_info")
            firestore_db = firestore.Client(project=Config.gsm_project)
            rbac_info = firestore_db.collection(ACCESS_CONTROL_COLLECTION_NAME).where(filter=FieldFilter('client_id', '==', client_id)).get()
            if len(rbac_info) == 0:
                raise HTTPException(status_code=400, detail=f"Could not find access information for client_id {client_id}")
            else:
                milvus_collections = rbac_info[0].to_dict()['milvus_collections']
                return milvus_collections
        except HTTPException as httpe:
            raise httpe
        except Exception as e:
            logging.critical(f"Error loading {ACCESS_CONTROL_COLLECTION_NAME} collection from firestore.")

@router.post("/vectordb/search", tags=["vectordb"])
def process_similarity_search_request(queryObject: SearchQuery,
                                      db: dict = Depends(get_db_dict),
                                      default_embedding_func = Depends(get_embedding_func),
                                      milvus_collections: Union[list[str], None] = Depends(get_approved_collections)
                                      ):
    return handle_search_request(
        queryObject=queryObject,
        milvus_helper=db["milvus_helper"],
        embedding_func=default_embedding_func,
        collections=milvus_collections
    )

@router.post("/logs/unanswered_questions", tags=["usecases"])
def process_summarize_unanswered_questions_request(queryObject: LogsSummaryBotQuery,
                                                   logging_obj: dict = Depends(get_logging_dict)):
    return handle_summarize_unanswered_questions_request(queryObject, project_id=logging_obj['PROJECT_ID'] )

@router.post("/llm/ask", tags=["llm"])
def process_llm_ask_request(queryObject: RawChatQuery):
    generate_response_with_callback = handle_llm_ask_request(
        queryObject=queryObject,
    )
    return send_response_from_generate_func(queryObject, generate_response_with_callback)

@router.post("/bots/generic", tags=["common"])
def process_generic_bot_request(queryObject: QABotBaseQueryFlexible,
                                db: dict = Depends(get_db_dict),
                                default_embedding_func = Depends(get_embedding_func),
                                milvus_collections: Union[list[str], None] = Depends(get_approved_collections)):
    generate_response_with_callback = handle_generic_request(
        queryObject=queryObject,
        milvus_helper=db["milvus_helper"],
        embedding_func=default_embedding_func,
        collections=milvus_collections
    )
    return send_response_from_generate_func(queryObject, generate_response_with_callback)

@router.post("/bots/spoc", tags=["usecases"])
def process_spoc_bot_request(queryObject: SpocBotQuery,
                             db: dict = Depends(get_db_dict),
                             default_embedding_func = Depends(get_embedding_func),
                             logging_obj: dict = Depends(get_logging_dict),
                             milvus_collections: Union[list[str], None] = Depends(get_approved_collections)
                            ):
    generate_response_with_callback = handle_spoc_request(
        queryObject=queryObject,
        milvus_helper=db["milvus_helper"],
        embedding_func=default_embedding_func,
        collections=milvus_collections
    )
    return send_response_from_generate_func(
                                        queryObject=queryObject, 
                                        generate_response_with_callback=generate_response_with_callback, 
                                        table_id=f"{logging_obj['PROJECT_ID']}.{logging_obj['LOGGING_DATASET_ID']}.{logging_obj['SPOC_TABLE_ID']}" ,
                                        bq=logging_obj["bq"]
                                    )

@router.post("/bots/milo", tags=["usecases"])
def process_milo_bot_request(queryObject: MiloBotQuery,
                             db: dict = Depends(get_db_dict),
                             default_embedding_func = Depends(get_embedding_func),
                             logging_obj: dict = Depends(get_logging_dict),
                             milvus_collections: Union[list[str], None] = Depends(get_approved_collections)
                            ):
    generate_response_with_callback = handle_milo_request(
        queryObject=queryObject,
        milvus_helper=db["milvus_helper"],
        embedding_func=default_embedding_func,
        collections=milvus_collections
    )
    return send_response_from_generate_func(
                                        queryObject=queryObject, 
                                        generate_response_with_callback=generate_response_with_callback, 
                                        table_id=f"{logging_obj['PROJECT_ID']}.{logging_obj['LOGGING_DATASET_ID']}.{logging_obj['MILO_TABLE_ID']}" ,
                                        bq=logging_obj["bq"]
                                    )

@router.post("/bots/public_mobile", tags=["usecases"])
def process_public_mobile_bot_request(queryObject: PublicMobileBotQuery,
                                      db: dict = Depends(get_db_dict),
                                      default_embedding_func = Depends(get_embedding_func),
                                      logging_obj: dict = Depends(get_logging_dict),
                                      milvus_collections: Union[list[str], None] = Depends(get_approved_collections)
                                    ):
    generate_response_with_callback = handle_public_mobile_request(
        queryObject=queryObject,
        milvus_helper=db["milvus_helper"],
        embedding_func=default_embedding_func,
        collections=milvus_collections
    )
    return send_response_from_generate_func(
                                        queryObject=queryObject, 
                                        generate_response_with_callback=generate_response_with_callback, 
                                        table_id=f"{logging_obj['PROJECT_ID']}.{logging_obj['LOGGING_DATASET_ID']}.{logging_obj['PUBLIC_MOBILE_TABLE_ID']}" ,
                                        bq=logging_obj["bq"]
                                    )

@router.post("/bots/pnc", tags=["usecases"])
def process_pnc_bot_request(queryObject: PnCBotQuery):
    generate_response_with_callback = handle_pnc_request(
        queryObject=queryObject,
        milvus_helper=db["milvus_helper"],
        embedding_func=default_embedding_func,
    )
    return send_response_from_generate_func(
                                        queryObject=queryObject, 
                                        generate_response_with_callback=generate_response_with_callback, 
                                        table_id=f"{logging_obj['PROJECT_ID']}.{logging_obj['LOGGING_DATASET_ID']}.{logging_obj['PNC_TABLE_ID']}" ,
                                        bq=logging_obj["bq"]
                                    )

@router.post("/bots/pso", tags=["usecases"])
def process_pso_bot_request(queryObject: PsoBotQuery,
                            db: dict = Depends(get_db_dict),
                            default_embedding_func = Depends(get_embedding_func),
                            logging_obj: dict = Depends(get_logging_dict),
                            milvus_collections: Union[list[str], None] = Depends(get_approved_collections)
                         ):
    generate_response_with_callback = handle_pso_request(
        queryObject=queryObject,
        milvus_helper=db["milvus_helper"],
        embedding_func=default_embedding_func,
        collections=milvus_collections
    )
    return send_response_from_generate_func(
                                        queryObject=queryObject, 
                                        generate_response_with_callback=generate_response_with_callback, 
                                        table_id=f"{logging_obj['PROJECT_ID']}.{logging_obj['LOGGING_DATASET_ID']}.{logging_obj['PSO_TABLE_ID']}" ,
                                        bq=logging_obj["bq"]
                                    )

@router.post("/bots/onesource", tags=["usecases"])
def process_onesource_bot_request(queryObject: OneSourceAPIRequest,
                                  db: dict = Depends(get_db_dict),
                                  default_embedding_func = Depends(get_embedding_func),
                                  logging_obj: dict = Depends(get_logging_dict),
                                  milvus_collections: Union[list[str], None] = Depends(get_approved_collections)
                                ):
    generate_response_with_callback = onesource_api_helpers.handle_onesource_request(
                        queryObject=queryObject,
                        milvus_helper=db["milvus_helper"],
                        embedding_func=default_embedding_func,
                        firestore_doc_ref = db["firestore_onesource_doc_ref"],
                        collections=milvus_collections  
                    )
    return send_response_from_generate_func(
                                        queryObject=queryObject, 
                                        generate_response_with_callback=generate_response_with_callback, 
                                        table_id=f"{logging_obj['PROJECT_ID']}.{logging_obj['LOGGING_DATASET_ID']}.{logging_obj['ONESOURCE_TABLE_ID']}" ,
                                        bq=logging_obj["bq"]
                                    )

@router.post("/bots/onesourcetbs", tags=["usecases"])
def process_onesource_tbs_bot_request(queryObject: OneSourceAPIRequest,
                                      db: dict = Depends(get_db_dict),
                                      default_embedding_func = Depends(get_embedding_func),
                                      logging_obj: dict = Depends(get_logging_dict),
                                      milvus_collections: Union[list[str], None] = Depends(get_approved_collections)
                                    ):
    generate_response_with_callback = onesource_api_helpers.handle_onesource_tbs_request(
                        queryObject=queryObject,
                        milvus_helper=db["milvus_helper"],
                        embedding_func=default_embedding_func,
                        collections=milvus_collections
                    )
    return send_response_from_generate_func(
                                        queryObject=queryObject, 
                                        generate_response_with_callback=generate_response_with_callback, 
                                        table_id=f"{logging_obj['PROJECT_ID']}.{logging_obj['LOGGING_DATASET_ID']}.{logging_obj['ONESOURCE_TABLE_ID']}" ,
                                        bq=logging_obj["bq"]
                                    )

@router.post("/bots/tcom_support", tags=["usecases"])
def process_tcom_support_bot_request(queryObject: TcomBotQuery,
                                     db: dict = Depends(get_db_dict),
                                     default_embedding_func = Depends(get_embedding_func),
                                     logging_obj: dict = Depends(get_logging_dict),
                                     milvus_collections: Union[list[str], None] = Depends(get_approved_collections)
                                    ):
    generate_response_with_callback = handle_tcom_support_request(
                                                                queryObject=queryObject,
                                                                milvus_helper=db["milvus_helper"],
                                                                embedding_func=default_embedding_func,
                                                                collections=milvus_collections
                                                            )
    return send_response_from_generate_func(
                                        queryObject=queryObject, 
                                        generate_response_with_callback=generate_response_with_callback, 
                                        table_id=f"{logging_obj['PROJECT_ID']}.{logging_obj['LOGGING_DATASET_ID']}.{logging_obj['TCOM_TABLE_ID']}" ,
                                        bq=logging_obj["bq"]
                                    )


# https://stackoverflow.com/questions/65504438/how-to-add-both-file-and-json-body-in-a-fastapi-post-request/70640522#70640522

@router.post("/vectordb/file", tags=["vectordb"])
async def file_upload(
                        file: Optional[UploadFile] = File(default=None),
                        collection_name: Optional[str] = Form('self_serve') ,
                        user: str = Form(...),
                        email: str = Form(...),
                        file_id: Optional[str] = Form(None),
                        db: dict = Depends(get_db_dict),
                        default_embedding_func = Depends(get_embedding_func),
                        logging_obj: dict = Depends(get_logging_dict),
                        gcs: dict = Depends(get_gcs)
                    ):

    resp = await file_upload_helpers.handle_upload_request( 
                                                        milvus_helper=db["milvus_helper"], 
                                                        embedding_func=default_embedding_func,
                                                        gcs=gcs['gcs'],
                                                        file=file, 
                                                        file_id=file_id,
                                                        collection_name=collection_name,
                                                        user=user, 
                                                        email=email,
                                                        bq=logging_obj["bq"],
                                                        table_id=f"{logging_obj['PROJECT_ID']}.{logging_obj['LOGGING_DATASET_ID']}.{logging_obj['UPLOAD_DB_TABLE_ID']}"                                                                                                
                                                    )
    return resp 

@router.get("/vectordb/collection", tags=["vectordb"])
def get_collections(db: dict = Depends(get_db_dict)):
    return db["milvus_helper"].list_collections("")


@router.get("/milvus/selfserve_get_loaded_files", tags=["common"])
def get_loaded_files(db: dict = Depends(get_db_dict)):
    try:
        res = db["milvus_helper"].query_collection(collection_name="self_serve",
                            expr="pk >= 0",
                            output_fields=['pk', 'metadata_user', 'metadata_email', 'metadata_filename', 'metadata_file_id'])
        df = pd.DataFrame(res)
        df = df.groupby(['metadata_user', 'metadata_email', 'metadata_filename', 'metadata_file_id'])[['pk']].count().reset_index().rename(columns={'pk': 'num_chunks'})

        return {"loaded_files": df.to_dict(orient='records')}
    except NameError as err:
        raise HTTPException(status_code=404, detail=err.__str__())
    except ValueError as err:
        raise HTTPException(status_code=400, detail=err.__str__())
    except Exception as err:
        raise HTTPException(status_code=500, detail=err.__str__())

@router.post("/summarize", tags=["usecases"])
def summarize_cclog(queryObject: SummarizationRequest,
                    db: dict = Depends(get_db_dict),
                    milvus_collections: Union[list[str], None] = Depends(get_approved_collections)
                    ):
    return handle_Summary_Request(queryObject=queryObject,
                                       milvus_helper=db["milvus_helper"],
                                       collections=milvus_collections
                                       )

@router.post("/knowledge_assist", tags=["usecases"])
def process_knowledge_assist_request(queryObject: KnowledgeAssistRequest,
                                     db: dict = Depends(get_db_dict),
                                     default_embedding_func = Depends(get_embedding_func)
                                    ):
    return handle_knowledge_assist_request(
        queryObject=queryObject,
        milvus_helper=db["milvus_helper"],
        embedding_func=default_embedding_func,
        firestore_doc_ref = db["firestore_onesource_doc_ref"] 
    )
