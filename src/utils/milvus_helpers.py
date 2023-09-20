import os
import json
from tqdm.auto import tqdm
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance
from langchain.embeddings.base import Embeddings
from datetime import datetime
from typing import Any, Iterable, List, Optional, Tuple, Union
import secrets
import logging
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    Role
)
import numpy as np
from utils.config import Config
from fastapi import HTTPException


# Langchain Milvus VectorStore: https://python.langchain.com/en/latest/_modules/langchain/vectorstores/milvus.html#Milvus

class MilvusHelper:
    def __init__(self, connection_args, bypass_proxy=None, timeout=10) -> None:
        self.timeout = timeout
        if bypass_proxy is None:
            bypass_proxy = os.getenv('MILVUS_BYPASS_PROXY', 'true')
            bypass_proxy = (bypass_proxy.lower() == 'true')
            print(f'Milvus Bypass Proxy: {bypass_proxy}')

        # sometimes proxy is undesirable if DB is local, so temporarily disabling it
        if bypass_proxy:
            http_proxy = os.getenv("HTTP_PROXY", "")
            https_proxy = os.getenv("HTTPS_PROXY", "")
            os.environ["HTTP_PROXY"] = ""
            os.environ["HTTPS_PROXY"] = ""

        if connection_args is None:
            milvus_host = Config.fetch('milvus-host', default='localhost')
            milvus_port = Config.fetch('milvus-port', default='19530')
            milvus_user = Config.fetch('milvus-user', default='root')
            milvus_pass = Config.fetch('milvus-password', default='Milvus')

            connection_args = {
                'alias': "default",
                'host': milvus_host,
                'port': milvus_port,
                'user': milvus_user,
                'password': milvus_pass,
                'timeout': self.timeout
            }
        self.connection_args = connection_args

        print(f"connecting to {connection_args['user']}@{connection_args['host']}:{connection_args['port']}")

        # check for connection alias
        #alias = connection_args.get('alias', 'default')
        #if not connections.has_connection(alias): # This doesnt actually test the connection to Milvus
        try:
            connections.connect(**connection_args)
            self.connections = connections 
            logging.info("Connected to Milvus")
        except Exception as e:
            logging.critical(f"Could not connect to Milvus: {e}")
        
        if bypass_proxy:
            try:
                print('setting the proxy back to', http_proxy)
                os.environ["HTTP_PROXY"] = http_proxy
                os.environ["HTTPS_PROXY"] = https_proxy
            except:
                print('error setting proxies back!')

    def close_connection(self):
        self.connections.disconnect(self.connection_args['alias'])
        
    @staticmethod
    def convert_metadata(m, k):
        # passing in metadata and key so that we can use hints in key name if needed
        if type(m[k]) is datetime:
            return m[k].timestamp()
        if type(m[k]) is list:
            return json.dumps(m[k])
        return m[k]
    
    @staticmethod
    def convert_entity_to_metadata(r, k):
        # passing in metadata and key so that we can use hints in key name if needed
        if k.endswith('_ts'):
            return datetime.fromtimestamp(r.entity.get(k))
        if k.endswith('_list'):
            return json.loads(r.entity.get(k))
        return r.entity.get(k)

    # TODO: add partition support here!
    def dump_docs(self, docs, embeddings, collection_name, chunk_size=1000):
        # find maximum page content length
        max_length = 0
        for d in docs:
            max_length = max(max_length, len(d.page_content))
        max_length = max(4096, max_length)

        # length of a single embedding
        embedding_length = len(embeddings[0])

        # create a list of fields to push
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=max_length + 1),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_length),
        ]
        meta_keys = list(docs[0].metadata.keys())
        for k in meta_keys:
            md = docs[0].metadata
            if type(md[k]) is int:
                dtype = DataType.INT64
                kwargs = {}
            elif type(md[k]) is float:
                dtype = DataType.DOUBLE
                kwargs = {}
            elif type(md[k]) is bool:
                dtype = DataType.BOOL
                kwargs = {}
            elif type(md[k]) is datetime:
                dtype = DataType.DOUBLE
                kwargs = {}
            else:
                dtype = DataType.VARCHAR
                kwargs = {"max_length": 2048}

            fields.append(
                FieldSchema(name=f"metadata_{k}", dtype=dtype, **kwargs)
            )

        # drop collection if exists
        if utility.has_collection(collection_name, timeout=self.timeout):
            utility.drop_collection(collection_name, timeout=self.timeout)

        schema = CollectionSchema(fields, f"{collection_name} created at {datetime.now()}")
        collection = Collection(collection_name, schema)

        # Index parameters for the collection
        index = {
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {"M": 8, "efConstruction": 64},
        }
        # Create the index
        collection.create_index("embedding", index)
        return self.insert_docs(docs, embeddings, collection, chunk_size)

    def insert_docs(self, docs, embeddings, collection, chunk_size=1000):
        meta_keys = list(docs[0].metadata.keys())
        # insert vectors
        entities = [
            #pk
            # text
            [d.page_content for d in docs],
            # vector_field
            embeddings,
            # f"metadata_{k}"
            *[
                [MilvusHelper.convert_metadata(d.metadata, k) for d in docs] for k in meta_keys
            ],
        ]

        all_res = []
        for idx in tqdm(range(0, len(entities[0]), chunk_size)):
            res = collection.insert([d[idx:idx+chunk_size] for d in entities])
            all_res.append(res)

        #collection.flush()
        logging.info(f'# Inserting {len(entities[0])} Entities into {collection.describe()["collection_name"]}!')
        logging.info(f'# Total Entities in Collection: {collection.num_entities}')

        # we'll load it on every push so that we can refresh it on the client side
        # to refresh, use `collection.load(_refresh = True)`
        collection.load()

        # flatten the all_res list (list of lists) in a list and return it
        return [d for res in all_res for d in res.primary_keys]

    def search_by_vector_with_scores(self, vectors_to_search, collection_name, expr="", limit=10, search_params=None):
        if not utility.has_collection(collection_name):
            raise NameError('Collection does not exist!')
        
        collection = Collection(collection_name)
        collection_metric_type = collection.indexes[0].params['metric_type']
        if search_params is None:
            if collection_metric_type == "L2":
                search_params = {
                    "metric_type": "L2",
                    "params": {"nprobe": 10, "ef": 10000},
                }
            elif collection_metric_type == "IP":
                search_params = {
                    "metric_type": "IP",
                    "params": {"nprobe": 10, "ef": 10000},
                }
            else:
                raise Exception(f"Collection Unsupported metric type: {collection_metric_type}")

        fields = [f.name for f in collection.schema.fields if f.name != "embedding"]
        meta_keys = [f.replace('metadata_', '') for f in fields if f.startswith('metadata_')]

        try:
            results = collection.search(
                vectors_to_search, "embedding", search_params, limit=limit,
                output_fields = fields,
                expr=expr,
                timeout=self.timeout
            )
        except Exception as e:
            if "cannot parse expression" in e.message:
                logging.critical(f"{collection_name} Invalid expression: {e.message}")
                raise ValueError("Invalid expression: ", e.message)
                
            else:
                logging.critical(f"{collection_name} Collection Error: {str(e)}")
                raise e

        ret_list = []
        # each search vectore
        for res in results:
            ret = []
            # each result for that search vector
            for r in res:
                meta = {k: MilvusHelper.convert_entity_to_metadata(r, f"metadata_{k}") for k in meta_keys}
                meta['pk'] = r.entity.get("pk")
                meta["text"] = r.entity.get("text")
                ret.append(
                    (
                        Document(
                            page_content=meta.pop("text"), metadata=meta
                        ),
                        r.distance,
                    )
                )
            
            ret_list.append(ret)

        return ret_list
    
    def search_by_vector_with_mmr_scores(self, vectors_to_search, collection_name, k, fetch_k, lambda_mult, expr="", search_params=None): 
        if not utility.has_collection(collection_name):
            raise NameError('Collection does not exist!')
        
        collection = Collection(collection_name)
        collection_metric_type = collection.indexes[0].params['metric_type']
        if search_params is None:
            if collection_metric_type == "L2":
                search_params = {
                    "metric_type": "L2",
                    "params": {"nprobe": 10, "ef": 10000},
                }
            elif collection_metric_type == "IP":
                search_params = {
                    "metric_type": "IP",
                    "params": {"nprobe": 10, "ef": 10000},
                }
            else:
                raise Exception(f"Collection Unsupported metric type: {collection_metric_type}")
        fields = [f.name for f in collection.schema.fields if f.name != "embedding"]
        meta_keys = [f.replace('metadata_', '') for f in fields if f.startswith('metadata_')]

        try:
            results = collection.search(
                vectors_to_search, "embedding", search_params, limit=fetch_k,
                output_fields = fields,
                expr=expr,
                timeout=self.timeout
            )
        except Exception as e:
            if "cannot parse expression" in e.message:
                logging.critical(f"{collection_name} Invalid expression: {e.message}")
                raise ValueError("Invalid expression: ", e.message)
                
            else:
                logging.critical(f"{collection_name} Collection Error: {str(e)}")
                raise e
            
        ids = []
        documents = []
        scores = []
        for result in results[0]:
            meta = {k: MilvusHelper.convert_entity_to_metadata(result, f"metadata_{k}") for k in meta_keys}
            meta['pk'] = result.entity.get("pk")
            meta["text"] = result.entity.get("text")
            doc = Document(page_content=meta.pop("text"), metadata=meta)
            documents.append(doc)
            scores.append(result.score)
            ids.append(result.id)
        vectors = collection.query(
            expr=f"pk in {ids}",
            output_fields=["pk", "embedding"],
            timeout=self.timeout,
        )
        # Reorganize the results from query to match search order.
        vectors = {x["pk"]: x["embedding"] for x in vectors}

        ordered_result_embeddings = [vectors[x] for x in ids]

        # Get the new order of results.
        new_ordering = maximal_marginal_relevance(
            np.array(vectors_to_search), ordered_result_embeddings, k=k, lambda_mult=lambda_mult
        )

        # Reorder the values and return.
        ret = []
        for x in new_ordering:
            # Function can return -1 index
            if x == -1:
                break
            else:
                ret.append((documents[x], scores[x]))
        return ret

    # user system function
    def create_users(self, config):
        """
            create any new users, generate and print password
            Parameters:
                config: json load of config file containing roles and users
        """
        users = config['users']
        for user_type in users.keys():
            for username in users[user_type]:
                if username in utility.list_usernames(using='default'):
                    print('User: ' + username + ' already exists')
                else:    
                    password = secrets.token_urlsafe(10)
                    print(password)
                    utility.create_user(user=username, password=password, using='default')

    def create_roles(self, config):
        """
            create a role and grant respective privileges
            bind role to respective users
            reads role names, types and privileges from roles config file
            Parameters: 
                config: json load of config file containing roles and users
        """
        roles = config['roles']
        users = config['users']

        for role_name in roles.keys():
            role = Role(role_name, using='default')
            if role.is_exist():
                print('Role ' + role_name + ' already exists')
        
            #create role and add grants for type dev or viewer
            else:
                role.create()
                for privilege, object in roles[role_name].items():
                    role.grant(object, '*', privilege)

            #add users to respective roles
            for user in users[role_name]:
                role.add_user(user)

    def get_role_grants(self, config):
        """ 
            lists all the privileges granted to a role
            Parameter(s):
                config: json load of config file containing roles and users
        """
        roles = config['roles']
        for role_name in roles.keys():
            role = Role(role_name, using='default')
            print('Role ' + role_name + '\n')
            print(role.list_grants())

    def list_collections(self, prefix):
        collection_list = utility.list_collections(timeout=self.timeout)
        filtered_collection_list = [name for name in collection_list if name.startswith(prefix)]
        return filtered_collection_list

    def get_collection(self, collection_name):
        if utility.has_collection(collection_name, timeout=self.timeout):
            collection = Collection(collection_name)
            return collection
        else:
            print(f"Collection {collection_name} does not exist!")
            return None

    def release_collection(self, collection_name):
        if utility.has_collection(collection_name, timeout=self.timeout):
            collection = Collection(collection_name)
            collection.release(timeout=self.timeout)
            print(f"Collection {collection_name} released!")
        else:
            print(f"Collection {collection_name} does not exist!")

    def drop_collection(self, collection_name):
        if utility.has_collection(collection_name, timeout=self.timeout):
            utility.drop_collection(collection_name, timeout=self.timeout)
            print(f"Collection {collection_name} dropped!")
        else:
            print(f"Collection {collection_name} does not exist!")

    def get_schema(self, collection_name):
        if utility.has_collection(collection_name, timeout=self.timeout):
            collection = Collection(collection_name)
            return collection.schema.to_dict()
        else:
            print(f"Collection {collection_name} does not exist!")
            return {}

    def get_field_names(self, collection_name):
        if utility.has_collection(collection_name, timeout=self.timeout):
            collection = Collection(collection_name)
            return [f.name for f in collection.schema.fields]
        else:
            print(f"Collection {collection_name} does not exist!")
            return []    
    
    def query_collection(self, collection_name, expr, offset= 0, k=None, output_fields=[], consistency_level='Strong'):
        # Need to specify the fields you want returned in output_fields, otherwise will only return pk
        # expr is the search expression, see: https://milvus.io/docs/boolean.md
        collection = self.get_collection(collection_name)
        if collection is None:
            return []

        else:
            collection.load()
            res = collection.query(
                expr=expr,
                offset=offset,
                limit=k,
                output_fields=output_fields,
                consistency_level=consistency_level,
                timeout=self.timeout
            )
            return res
        
    def get_delete_all_entities_expr(self, collection_name):
        res = self.query_collection(collection_name, expr='(pk>=0) || (pk<0)', output_fields=['pk'], timeout=self.timeout)
        pk_list = [str(val['pk']) for val in res]
        del_search_expr = f"pk in [{','.join(pk_list)}]"

        return del_search_expr


    # https://milvus.io/docs/v2.0.x/delete_data.md?shell#Delete-Entities
    # Expression to delete must be "FIELD in [value1, value2, ...]", cant have generic search expr
    def delete_entities(self, collection_name, expr=''): 
        collection = self.get_collection(collection_name)
        if collection is None:
            return False
        else:
            if expr== '': # Empty expr, assume delete all contents
                expr = self.get_delete_all_entities_expr(collection_name)
            collection.delete(expr)
            print(f"Entities in {collection_name} deleted!")
            return True


class AIAMilvus(VectorStore):
    def __init__(self, embedding_function, collection_name, connection_args=None, milvus_bypass_proxy=None, milvus_helper=None) -> None:
        super().__init__()
        self.collection_name = collection_name
        self.embedding_func = embedding_function
        if milvus_helper is not None:
            self.mh = milvus_helper
        else:
            self.mh = MilvusHelper(
                connection_args=connection_args,
                bypass_proxy=milvus_bypass_proxy,
            )
    
    def get_collection(self):
        return self.mh.get_collection(self.collection_name)
    
    def get_schema(self):
        return self.mh.get_schema(self.collection_name)
    
    def get_field_names(self):
        return self.mh.get_field_names(self.collection_name)
    
    def query_collection(self, expr, offset=0, k=100, output_fields=[], consistency_level='Strong'):
        # Need to specify the fields you want returned in output_fields, otherwise will only return pk
        # expr is the search expression, see: https://milvus.io/docs/boolean.md
        return self.mh.query_collection(self.collection_name, expr, offset, k, output_fields, consistency_level)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        docs = []
        for text, metadata in zip(texts, metadatas):
            docs.append(
                Document(
                    page_content=text,
                    metadata = metadata,
                )
            )
        return self.add_documents(
            documents=docs,
            **kwargs
        )
        

    def add_documents(
        documents: List[Document],
        embedding: Embeddings,
        collection_name: str,
        embeddings_list: Optional[List[float]] = None,
        # TODO: impleement partition name and timeout
        # partition_name: Optional[str] = None,
        # timeout: Optional[int] = None,
        connection_args: Optional[Any] = None,
        milvus_bypass_proxy: Optional[Union[bool, None]] = None,
        milvus_helper: Optional[Union[MilvusHelper, None]] = None,
        **kwargs: Any

    ) -> List[str]:
        texts = [d.page_content for d in documents]
        if embeddings_list is None:
            print('Embedding documents')
            try:
                embeddings_list = embedding.embed_documents(
                    list(texts)
                )
            except NotImplementedError:
                embeddings_list = [
                    embedding.embed_query(x) for x in texts
                ]
        else:
            print('Using provided embeddings')

        if milvus_helper is None:
            mh = MilvusHelper(
                connection_args=connection_args,
                bypass_proxy=milvus_bypass_proxy,
            )
        else:
            mh = milvus_helper

        collection = Collection(collection_name)        
        return mh.insert_docs(documents, embeddings_list, collection, chunk_size=1000)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        expr: str = "",
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding_func.embed_query(query)
        vectors_to_search = [ embedding ]
        res = self.mh.search_by_vector_with_scores(
            vectors_to_search,
            self.collection_name,
            expr,
            limit=k,
        )
        return res[0]

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score(
            query, k, **kwargs,
        )
        for doc in docs_and_scores:
            doc[0].metadata['similarity_score'] = doc[1]
        return [doc for doc, _ in docs_and_scores]
    
    def similarity_search_by_vector(self, embedding: List[float], k: int = 4, **kwargs: Any) -> List[Document]:
        vectors_to_search = [ embedding ]
        res = self.mh.search_by_vector_with_scores(
            vectors_to_search,
            self.collection_name,
            kwargs.get('expr', ''),
            limit=k,
        )
        return res[0]

    def max_marginal_relevance_search_with_score(
        self, 
        query: str,  
        k: int = 4, 
        fetch_k: int = 20, 
        lambda_mult: float = 0.5,
        expr:str = ""
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding_func.embed_query(query)
        vectors_to_search = [ embedding ]
        res = self.mh.search_by_vector_with_mmr_scores(
            vectors_to_search,
            self.collection_name,
            k,
            fetch_k,
            lambda_mult,
            expr
        )
        return res

    def max_marginal_relevance_search(self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, **kwargs: Any) -> List[Document]:
        docs_and_scores = self.max_marginal_relevance_search_with_score(
            query, 
            k, 
            fetch_k, 
            lambda_mult,
            **kwargs
        )
        for doc in docs_and_scores:
            doc[0].metadata['similarity_score'] = doc[1]
        return [doc for doc, _ in docs_and_scores]
    
    def max_marginal_relevance_search_by_vector(self, embedding: List[float], k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, **kwargs: Any) -> List[Document]:
        vectors_to_search = [ embedding ]
        res = self.mh.search_by_vector_with_mmr_scores(
            vectors_to_search,
            self.collection_name,
            k,
            fetch_k,
            lambda_mult,
            kwargs.get('expr', '')
        )
        return res

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        collection_name: str,
        embeddings_list: Optional[List[float]] = None,
        connection_args: Optional[Any] = None,
        milvus_bypass_proxy: Optional[Union[bool, None]] = None,
        # TODO: impleement partition name and timeout
        # partition_name: Optional[str] = None,
        # timeout: Optional[int] = None,
        chunk_size: Optional[int] = 1000,
        milvus_helper: Optional[Union[MilvusHelper, None]] = None,
        **kwargs: Any
    ) -> VectorStore:
        texts = [d.page_content for d in documents]

        if embeddings_list is None:
            print('Embedding documents')
            try:
                embeddings_list = embedding.embed_documents(
                    list(texts)
                )
            except NotImplementedError:
                embeddings_list = [
                    embedding.embed_query(x) for x in texts
                ]
        else:
            print('Using provided embeddings')
        
        if milvus_helper is None:
            mh = MilvusHelper(
                connection_args=connection_args,
                bypass_proxy=milvus_bypass_proxy,
            )
        else:
            mh = milvus_helper
            
        print('Connected to Milvus')
        return mh.dump_docs(documents, embeddings_list, collection_name, chunk_size=chunk_size)

        # milvus = cls(
        #     embedding_function=embedding,
        #     connection_args=connection_args,
        #     collection_name=collection_name,
        #     milvus_bypass_proxy=milvus_bypass_proxy,
        # )

        # return milvus


    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any
    ) -> VectorStore:
        docs = []
        for text, metadata in zip(texts, metadatas):
            docs.append(
                Document(
                    page_content=text,
                    metadata = metadata,
                )
            )
        return cls.from_documents(
            texts,
            embedding,
            metadatas,
            **kwargs,
        )
    