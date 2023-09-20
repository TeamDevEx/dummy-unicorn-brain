from utils.milvus_helpers import AIAMilvus, MilvusHelper
from langchain.embeddings.base import Embeddings
import json 
from typing import Dict
from langchain.vectorstores import Milvus

from fastapi import HTTPException, UploadFile, File
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.docstore.document import Document
from langchain.document_loaders import GCSFileLoader
from langchain.text_splitter import TokenTextSplitter
from typing import List, Optional

import os 
import logging

from utils.gcs_helpers import GCSBucket
import time
import uuid
from datetime import datetime
from api.v1_0.usecases.base.utility import log_to_bq
from utils.bq_logging import bq_logging 
import hashlib 

accepted_file_types = ['application/pdf', 
                       'text/plain', 
                       'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                       ]

def check_if_file_exists_in_milvus(milvus_helper: MilvusHelper, 
                                   file_id : str,
                                   collection_name: str):
    
    search_expr = f"(metadata_file_id == \"{file_id}\")"
    res = milvus_helper.query_collection(collection_name=collection_name, expr=search_expr)
    if len(res) > 0:
        return True, len(res) # Return true if res list is not empty and the number of entitites in the file_id 
    else:
        return False, 0

# Define Query Object
async def handle_upload_request(
            milvus_helper: MilvusHelper,
            embedding_func: Embeddings,
            collection_name : str,
            user: str, 
            email: str,
            file: Optional[UploadFile] = None, 
            file_id: Optional[str] = None,
            gcs: Optional[GCSBucket] = None, 
            bq: Optional[bq_logging] = None,
            table_id: Optional[str] = None
        
    ) -> Dict:

    request_id = uuid.uuid4().fields[-1]
    request_time = datetime.utcnow()
    start_time = time.time()

    # Re-use request_id for metadata_file_id
    res_object_base = {
        'request_id': request_id,
        'request_time': request_time,
        'uploaded': False,
    }

    request_object = {
        "user" : user,
        "email" : email,
        "collection_name" : collection_name,
    }

    if file is None and file_id is None:
        raise HTTPException(status_code=400, detail="No file or file_id provided. Please provide either a file or file_id")
    
    if file is not None:      
        request_object = {
                **request_object,
                "filename": file.filename,
                "content_type": file.content_type,
                "filesize" : file.size,
        }

        if file.content_type not in accepted_file_types:
            res_object = {
                    **res_object_base,
                    'number_of_chunks_inserted': 0,
                    'elapsed': time.time() - start_time,
                    'uploaded' : False,
                    'status_code': 400,
                    'msg' : f"File type not supported: Only .pdf, .txt, .docx supported. "
            }

            log_to_bq(bq, table_id, request_object, res_object)
            raise HTTPException(status_code=400, detail=f"File type not supported: Only .pdf, .txt, .docx supported. ")

        elif file.size > 20 * 1000 * 1000:
            res_object = {
                    **res_object_base,
                    'number_of_chunks_inserted': 0,
                    'elapsed': time.time() - start_time,
                    'uploaded' : False,
                    'status_code': 400,
                    'msg' : f"File size too large: Max 20MB "
            }

            log_to_bq(bq, table_id, request_object, res_object)
            raise HTTPException(status_code=400, detail="File size too large: Max 20MB")
    
        
    if gcs is None:
        raise HTTPException(status_code=500, detail="GCS Bucket not initialized. Ensure user has access to GCS bucket - line 85 in api.v1_0.py")

    # Upload File Flow:
    # Receive either a 1. file via API or 2. file_id (sha256 hash of file)
    # Receive file:
    # 1. Get sha256 hash of file
    # 2. Check if file already exists in milvus based on sha256 (file_id)
    #        If exists: Return file exists in Milvus
    #        If doesnt exist in Milvus, upload to GCS 

    # Receive file_id:
    # 1. Check if file_id exists in Milvus -> If so, return file exists in Milvus
    # 2. If not, Check if file exists in GCS
    #        If file does not exist in GCS -> return an error

    # If file_exists_in_milvus == False -> Upload to Milvus (After above checks)
    #   File should be in GCS by now

    try:
        if file is not None: # If file_id and file is both provided, will overwrite file_id based on sha256 from file
            request_object = {
                            **request_object, 
                            "filename": file.filename
                }
            # Get sha256 of file
            file_bytes = await file.read()
            sha256_hash = hashlib.sha256(file_bytes)
            file_id = sha256_hash.hexdigest()
            #print(f"sha256 hash from memory: {file_id}")

            # Check if file exists in Milvus
            file_exists_in_milvus, num_entities = check_if_file_exists_in_milvus(milvus_helper=milvus_helper,
                                                                                file_id=file_id,
                                                                                collection_name=collection_name) 
            
            if file_exists_in_milvus == True:
                # File exists in milvus uploaded = False
                res_object = {
                    **res_object_base,
                    'collection_name': collection_name,
                    'number_of_chunks_inserted': 0,
                    'elapsed': time.time() - start_time,
                    'uploaded' : False,
                    'file_id': file_id,
                    'msg': 'file already exists in milvus'
                }
                
                log_to_bq(bq, table_id, request_object, res_object)
                return res_object                    

            else: # File does not exist in Milvus -> Upload to GCS  
                # Check if file exists in GCS
                # gcs_filename = sha256 of file + extension
                exists_in_gcs, gcs_filename = gcs.check_file_Exists_prefix(prefix=file_id) 
                if exists_in_gcs== False:
                    # File does not exist in GCS -> Upload to GCS
                
                    await file.seek(0) # Reset file position to beginning
                    file_metadata = {
                        "filename" : file.filename, 
                        "user" : user,
                        "email": email,
                        "file_extension": os.path.splitext(file.filename)[1]
                    } # Upload metadata to file -> Will be useful later when client provides a file_id and not a file
                    gcs.upload_file(file=file.file, 
                                    filename=f"{file_id}{file_metadata['file_extension']}", 
                                    content_type=file.content_type,
                                    metadata=file_metadata
                                ) # upload to GCS, name is file_id, if error is thrown during upload, exception will be raised
                    gcs_filename = f"{file_id}{file_metadata['file_extension']}"

        elif file_id is not None: # If file_id is provided, check if file exists in Milvus
            request_object = {
                            **request_object, 
                            "file_id": file_id
                }
            file_exists_in_milvus, num_entities = check_if_file_exists_in_milvus(milvus_helper=milvus_helper,
                                                                                file_id=file_id,
                                                                                collection_name=collection_name)
            if file_exists_in_milvus == True:
                # File exists in milvus uploaded = False
                res_object = {
                    **res_object_base,
                    'collection_name': collection_name,
                    'number_of_chunks_inserted': 0,
                    'elapsed': time.time() - start_time,
                    'uploaded' : False,
                    'file_id': file_id,
                    'msg': 'file already exists in milvus'
                }
                log_to_bq(bq, table_id, request_object, res_object)
                return res_object    
            
            else: # Check if file_id exists in GCS, if it does not, return an error
                exists_in_gcs, gcs_filename = gcs.check_file_Exists_prefix(prefix=file_id) 
                if exists_in_gcs == False:
                    res_object = {
                        **res_object_base,
                        'collection_name': collection_name,
                        'number_of_chunks_inserted': 0,
                        'elapsed': time.time() - start_time,
                        'uploaded' : False,
                        'file_id': file_id,
                        'msg': 'file_id does not exist in GCS or Milvus'
                    }
                    log_to_bq(bq, table_id, request_object, res_object)
                    return res_object

        if file_exists_in_milvus == False: 
            # File does not exist in Milvus -> Upload to Milvus from GCS
            # File currently exists in 
            metadata = gcs.get_gcs_file_metadata(filename=gcs_filename)

            # This loader uses UnstructuredFileLoader under the hood and takes care of the document types
            # Easier to just use this and then modify the metadata than to create own class
            loader = GCSFileLoader(bucket=gcs.bucket_name,
                                blob=gcs_filename,
                                project_name=gcs.project_id,
                                )

            docs = loader.load()         
            
            metadata = {
                        "filename" : metadata["filename"], 
                        "user" : user,
                        "email": email,
                        "file_extension": metadata["file_extension"],
                        "file_id": file_id
                        }

            # Specify metadata -> Needs to be the same format for all documents
            for doc in docs:
                doc.metadata = metadata

            # Default Chunking 
            text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
            source_chunks = text_splitter.split_documents(docs)

            # Embed documents
            embeddings_list = [embedding_func.embed_query(d.page_content) for d in source_chunks]

            index = AIAMilvus(embedding_function=embedding_func, 
                                milvus_helper=milvus_helper, 
                                collection_name=collection_name)
            
            # Create Collection if it doesn't exist
            if index.get_collection() is None:
                logging.info(f"Collection {collection_name} does not exist, creating collection!")
                # Create Collection + Load doc
                res = AIAMilvus.from_documents(
                                documents=source_chunks,
                                embedding=embedding_func,
                                embeddings_list=embeddings_list,
                                collection_name=collection_name,
                                connection_args=None, # use environment variables
                                milvus_bypass_proxy=None, # use environment variables
                                milvus_helper=milvus_helper
                            )
            
            else:
                # Append to collection
                res = AIAMilvus.add_documents(
                                        documents=source_chunks,
                                        embedding=embedding_func,
                                        embeddings_list=embeddings_list,
                                        collection_name=collection_name,
                                        connection_args=None,
                                        milvus_bypass_proxy=None,
                                        milvus_helper=milvus_helper
                                    )
                
            res_object = {
                **res_object_base,
                "filename": metadata["filename"],
                'collection_name': collection_name,
                'number_of_chunks_inserted': len(source_chunks),
                'elapsed': time.time() - start_time,
                'uploaded' : True,
                'file_id': file_id,
                'msg': 'file sucessfully uploaded to milvus!'
            }

            log_to_bq(bq, table_id, request_object, res_object)
            return res_object
        
        # If we get to this point in the code -> Error:
        res_object = {
                **res_object_base,
                'collection_name': collection_name,
                'number_of_chunks_inserted': 0,
                'elapsed': time.time() - start_time,
                'uploaded' : False,
                'msg': 'Unknown error occured during upload'
        
        }
        log_to_bq(bq, table_id, request_object, res_object)
        return res_object
    
    except Exception as e:

        res_object = {
            **res_object_base,
            'collection_name': collection_name,
            'number_of_chunks_inserted': 0,
            'elapsed': time.time() - start_time,
            'uploaded' : False,
            'file_id': file_id,
            'msg': f"error occured during upload: {str(e)}",
        }
        log_to_bq(bq, table_id, request_object, res_object)
        #return res_object
        raise HTTPException(status_code=500, detail=str(e))


        
            
        

