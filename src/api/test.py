import requests
import json
import sys
import argparse

# To run Script, specify bot:
# python test.py --bot spoc
# python test.py --bot onesource



def stream_response(response):
    print(" ")
    for line in response.iter_lines():
        if line:
            # Decode the line as JSON and print it
            json_data = line.decode('utf-8')
            json_object = json.loads(json_data)
            
    print(json_object)

# Define the JSON payload to send
payload_onesource = {
  "site_ids": [
    8,
    11
  ],
  "stream": True,
  "status_filter": "current",
  "content_filter": "",
  "creativity_level": "creative",
  #"temperature": 0.5, # Provide only either creativity_level or temperature,
 
  "question_improvement_engine_enabled": False,
  "query": "what is THTL?",
  "roles": [
    0
  ],
  "types": [
    0
  ],
   "x_id" : "txxxxx",
}

payload_spoc = {
    "chat_history": [
    {
      "role": "Human",
      "content": "How do I change my laptop?"
    },
    {
      "role": "Assistant",
      "content": "You will need to submit a TSR request."
    }
  ],
  "query": "how?",
  "language": "en",
  "stream": False,
  "return_docs": False
}


parser = argparse.ArgumentParser(description='Specify bot')
parser.add_argument('--bot', choices=['spoc', 'onesource'], help='Select a bot')
args = parser.parse_args()

if args.bot == 'spoc':
# Make the request to the FastAPI server
  response = requests.post('http://127.0.0.1:8080/1.0/bots/spoc', json=payload_spoc, stream=True)
elif args.bot == 'onesource':
  response = requests.post('http://127.0.0.1:8080/1.0/bots/onesource', json=payload_onesource, stream=True)
else:
  print('Please specify bot')
  sys.exit(1)

# Check if the request was successful
if response.status_code == 200:
    # Start streaming the response
    stream_response(response)
else:
    print('Error:', response.status_code)
    print(response.text)
