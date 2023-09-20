import pytest
import pytest_asyncio
import asyncio
from fastapi.testclient import TestClient
import os, sys 
absolute_path = os.path.dirname(__file__)
relative_path = "../../"
full_path = os.path.realpath(os.path.join(absolute_path, relative_path))
sys.path.append(full_path)
import requests
from api.v1_0.v1_0 import router, startup_event, load_milvus_db, load_onesource_firestore_db
from utils.milvus_helpers import AIAMilvus, MilvusHelper
from utils.config import Config
import json 


url = Config.fetch('V2_0_URL')
client = TestClient(router)

# Initializes startup event
@pytest.fixture(scope="module", autouse=True)
def initialize_for_test():

    startup_event()
    load_milvus_db()
    async def async_setup():
        await load_onesource_firestore_db()
    asyncio.run(async_setup())

# Can first get all collections and test all collections to make sure milvus collection still works
# Test search endpoint with all collections

mh = MilvusHelper(connection_args=None,
                  bypass_proxy=None)

db_collections = mh.list_collections(prefix='')
search_payloads = [{
                    "collection_name": collection,
                    "query": "test",
                    "max_num_results": 4,
                    "max_distance": 1,

                    } 
                    for collection in db_collections
                    ]

# Loops through all the collections in milvus and tests to see if a result can be returned using search endpoint
# Expand search endpoint for IP search 
@pytest.mark.parametrize("payload", search_payloads)
def test_collections_using_search(payload):
    response = client.post(f"{url}/search", json=payload)
    assert response.status_code==200

roles = ["Human", "Assistant", "System"]
raw_payloads = [{
                    "chat_history":[{"role": role, 
                                    "content" : "test"}],
                    "max_new_tokens": 100,
                    "temperature": 0.5,
                    "stream": False
                } for role in roles]

# Test raw endpoint
@pytest.mark.parametrize("payload", raw_payloads)
def test_raw(payload):
    response = client.post(f"{url}/bots/raw", json=payload)
    assert response.status_code==200
    assert response.json()["elapsed"] <= 5

# Testing Bots Stream = False
# generic, spoc, milo, public_mobile, pso, onesource, onesourcetbs, tcom_support
bot_test_cases = [
            (f"{url}/bots/generic", 
            {
                "chat_history": [],
                "query": "test",
                "language": "en",
                "stream": False,
                "return_docs": True,
                "collection_name" : "onesource_latest", # Could also initialize to a known collection or all collections similar to search test
                "temp": 0.4
            }),

            (f"{url}/bots/spoc",
            {
                "chat_history": [],
                "query": "how do i request software?",
                "stream": False,
                "return_docs": True,                  
            }),

            (f"{url}/bots/milo",
            {
                "chat_history": [],
                "query": "How do I return a tablet?",
                "stream": False,
                "return_docs": True,
                "brands": [
                    "TELUS",
                    "Koodo",
                    "Public"
                ],
                "provinces": [
                    "AB",
                    "BC"
                ],
                "status": "Current",
                "publish_date": "2023-08-15"
            }),

            (f"{url}/bots/public_mobile",
            {
                "chat_history": [],
                "query": "how do i join public points?",
                "language": "en",
                "stream": False,
                "return_docs": True

            }),

            (f"{url}/bots/pso",
            {
                "chat_history": [],
                "query": "hwhat is pso?",
                "language": "en",
                "stream": False,
                "return_docs": True
            }),

            (f"{url}/bots/onesource",
            {
                "site_ids": [
                    1,6,8,9,11,15,23
                ],
                "status_filter": "current",
                "qie_enabled": True,
                "chat_history": [],
                "query": "what is stream+?",
                "stream": False,
                "return_docs": True,
                "roles": [],
                "types": [],
                "x_id": ""
            }),

            (f"{url}/bots/onesourcetbs",
            {
                "site_ids": [
                    23
                ],
                "status_filter": "current",
                "qie_enabled": True,
                "chat_history": [],
                "query": "what is stream+?",
                "stream": False,
                "return_docs": True,
                "roles": [],
                "types": [],
                "x_id": ""
            }),

            (f"{url}/bots/tcom_support",
            {
                "chat_history": [],
                "query": "how do i enable roaming?",
                "language": "en",
                "stream": False,
                "return_docs": True

            }),
                
             
        ]

# Test with Streaming = False
@pytest.mark.parametrize("endpoint, payload", bot_test_cases)
def test_bot_endpoints_stream_false(endpoint, payload):
    response = client.post(endpoint, json=payload)
    response_json = response.json()

    assert response.status_code==200
    assert response_json['done'] == True 
    assert response_json['qie_status'] == True
    assert response_json['response'] != ""


# Use same test cases as above but with stream set to True
# Test output of stream is continuous and not broken up
bot_test_cases_stream_true =  [(x[0], {**x[1], "stream": True}) for x in bot_test_cases]
@pytest.mark.parametrize("endpoint, payload", bot_test_cases_stream_true)
def test_bot_endpoints_stream_true(endpoint, payload):
    
    response = client.post(endpoint, json=payload)
    prev_chunk = ""
    for chunk in response.iter_lines():
        if chunk:
            # Convert chunk into JSON to get response
            response_json = json.loads(chunk)
            response_field = response_json['response']
            if response_field != "" and prev_chunk != "":
                assert response_field.startswith(prev_chunk), f"Unexpected streamed content: {response_field}"
            prev_chunk = response_field

            
            if response_json['done'] == True:
                assert len(response_json['source_documents']) > 1 
                assert response_json['response'] != ""
                pass
           
    assert response.status_code==200

def test_log_unasnwered_questions():
    payload = {
                "bq_table":"cdo-gen-ai-island-np-204b23.logs.milo_np",
                "date":"2023-08-21"
            }
    response = client.post(f"{url}/logs/unanswered_questions", json=payload)
    assert response.status_code==200 