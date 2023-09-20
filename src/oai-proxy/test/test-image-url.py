import requests
import json
import openai
from dotenv import load_dotenv
import os
# from util.config import Config
load_dotenv()
# replace with your access token
api_key = os.environ.get("api_key")


url = "https://ai-np.cloudapps.telus.com/openai/images/generations:submit?api-version=2023-06-01-preview"

headers = {
    "api-key": api_key,  # assuming the token is a Bearer token
    "Content-Type": "application/json"
}
data = {
"prompt": "An avocado chair",
"size": "512x512",
"n": 3
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response)
