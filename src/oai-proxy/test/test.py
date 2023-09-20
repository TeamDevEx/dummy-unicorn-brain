import requests
import json
from dotenv import load_dotenv
import os
# from util.config import Config
load_dotenv()
# replace with your access token
access_token = 'your-access-token'
client_secret = os.environ.get("client_secret")
client_id = os.environ.get("client_id")
scope = ''  # optional
print(client_secret)
print(client_id)
# these should match the values provided in your postman collection
token_url = "https://apigw-st.telus.com/st/token"
auth_url = "https://teamsso-its04.telus.com/as/authorization.oauth2"

# form the request body
data = {
    'grant_type': 'client_credentials',
    'client_id': client_id,
    'client_secret': client_secret,
    'scope': scope
}

# make the POST request
response = requests.post(token_url, data=data)

# handle the response
if response.status_code == 200:
    # if the request was successful, the access token will be in the response
    access_token = response.json()['access_token']
    print('Access token:', access_token)
else:
    print('Failed to retrieve access token')
    print('Status code:', response.status_code)
    print('Response:', response.text)

url = "https://ai-np.cloudapps.telus.com/openai/deployments/telus-gpt-3_5/chat/completions?api-version=2023-03-15-preview"

headers = {
    "Authorization": "Bearer " + access_token,  # assuming the token is a Bearer token
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