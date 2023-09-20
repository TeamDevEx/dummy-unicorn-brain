import requests
import json
import openai
from dotenv import load_dotenv
import os
# from util.config import Config
load_dotenv()
# replace with your access token
api_key = os.environ.get("api_key")

openai.api_key = api_key
openai.api_version = '2023-03-15-preview'
openai.api_type = "azure"
openai.api_base = "https://ai-np.cloudapps.telus.com"

messages=  [
      {"role":"system","content":"You are an AI assistant that helps people find information."},
      {"role":"user","content":"how far is sun"}
]

reply = openai.ChatCompletion.create(               engine="telus-gpt-3_5",
                                            messages=messages,
                                            temperature=0.3,
                                            stream=False)

answer = reply.choices[0].message.content

print(reply)