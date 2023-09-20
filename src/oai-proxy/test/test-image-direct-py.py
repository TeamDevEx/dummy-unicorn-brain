import requests
import json
import openai
from dotenv import load_dotenv
import os
# from util.config import Config
load_dotenv()
# replace with your access token
api_key = os.environ.get("direct_api_key")
url_base = os.environ.get("direct_url")

openai.api_key = api_key
openai.api_version = '2023-06-01-preview'
openai.api_type = "azure"
openai.api_base = url_base

response = openai.Image.create(
  prompt="a white siamese cat",
  n=1,
  size="1024x1024"
)

# print(response)

image_url = response['data'][0]['url']

print(image_url)