import requests
import json
import openai
from dotenv import load_dotenv
import os
# from util.config import Config
load_dotenv()
# replace with your access token
api_key = os.environ.get("api_key")

url = "https://ai-np.cloudapps.telus.com/openai/deployments/telus-gpt-3_5/chat/completions?api-version=2023-03-15-preview"

headers = {
    "api-key": api_key,  # assuming the token is a Bearer token
    "Content-Type": "application/json"
}
data = {
  "messages": [
      {"role":"system","content":"You are an AI assistant that helps people find information."},
      {"role":"user","content":"how far is sun"}
    ],
  "max_tokens": 800,
  "temperature": 0.7,
  "frequency_penalty": 0,
  "presence_penalty": 0,
  "top_p": 0.95,
#   "stop": null,
  "stream": False
}

response = requests.post(url, headers=headers, data=json.dumps(data))

# check the status of the request
if response.status_code == 200:
    print('Request was successful')
    print('Response:', response.json())
else:
    print('Request failed')
    print('Status code:', response.status_code)

openai.api_key = api_key
openai.api_version = '2023-03-15-preview'
openai.api_type = "azure"
openai.api_base = "https://telus-openai.openai.azure.com"

messages=  [
      {"role":"system","content":"You are an AI assistant that helps people find information."},
      {"role":"user","content":"how far is sun"}
]
openai.ChatCompletion.create(               engine="telus-gpt-3_5",
                                            messages=messages,
                                            temperature=0.3,
                                            stream=False)