import os
import json
import requests
import threading
import time
from dotenv import load_dotenv
from flask import Flask, Response, request, jsonify
from util.config import Config

load_dotenv()

app = Flask(__name__)

BASE_DEPLOYMENT_UPSTREAM_URL = "https://_api_url/openai/deployments/_deployment_name/_subpath?api-version=_api_version"
BASE_IMAGE_UPSTREAM_URL = "https://_api_url/openai/images/_subpath:submit?api-version=_api_version"

failues = 0

# config_file_path = '/config/config.txt'

# properties = {}

# try:
#     with open(config_file_path, 'r') as config_file:
#         for line in config_file:
#             line = line.strip()
#             if line and not line.startswith('#'):
#                 key, value = line.split(':')
#                 properties[key.strip()] = value.strip()
#     print("Config file loaded.")
# except FileNotFoundError:
#     print("Config file not found.")

def build_upstream_url(api_url, deployment_name, subpath, api_version):
    return BASE_DEPLOYMENT_UPSTREAM_URL.replace('_api_url', api_url).\
            replace('_deployment_name', deployment_name).\
            replace('_subpath', subpath).\
            replace('_api_version', api_version)

def build_upstream_image_url(api_url, subpath, api_version):
    return BASE_IMAGE_UPSTREAM_URL.replace('_api_url', api_url).\
            replace('_subpath', subpath).\
            replace('_api_version', api_version)

def healthcheck_request():
    try:
        print("Background controller is running.")
        _api_url = f"{os.environ.get('OAI_INSTANCE')}-oai-instance-url"
        _api_key = f"{os.environ.get('OAI_INSTANCE')}-oai-instance-api-key"
        api_url = Config.fetch(_api_url.lower())
        api_key = Config.fetch(_api_key.lower())
        # api_url = os.environ.get('api_url')
        # api_key = os.environ.get('api_key')
        headers = {
            "Content-Type" : "application/json",
            "api-key": api_key
        }
        deployment_name = 'telus-gpt-3_5'
        subpath = 'chat/completions'
        q_api_version = '2023-03-15-preview'
        jdata = {
            "messages": [
                {"role":"user","content":"Hi"}
            ],
            "max_tokens": 20,
            "temperature": 0.1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "top_p": 0.95,
            "stream": False
        }
        upstream_response = requests.post(build_upstream_url(api_url, deployment_name, subpath, q_api_version),
                                        headers=headers, json=jdata, stream=True)

        if upstream_response.status_code == 200:
            # print('Response:', upstream_response.json())
            return "ok", upstream_response.status_code
        else:
            print('Health check request failed')
            print('Status code:', upstream_response.status_code)
            return "no-ok", upstream_response.status_code
    except Exception as e:
        print("Health check failing")
        print(e)
        return "no-ok", 500

# # Start the background controller thread when the app starts
# controller_thread = threading.Thread(target=background_healthcheck_controller())
# controller_thread.daemon = True  # Allow the thread to exit when the main program exits
# controller_thread.start()

def generate_stream(url, jdata, headers):
    upstream_response = requests.post(url, headers=headers, json=jdata)
    for chunk in upstream_response.iter_content(chunk_size=128):
        if chunk:
            yield chunk

def is_client_api_key_valid(client_api_key):
    client_app_ref, intermidiate_key = client_api_key.split(':')

    if client_app_ref is None or intermidiate_key is None:
        return False

    secret_name = f"{client_app_ref}-intermidiate-apikey"
    gsm_secret_value = Config.fetch(secret_name.lower())

    return intermidiate_key == gsm_secret_value

@app.route('/openai/deployments/<string:deployment_name>/<path:subpath>', methods=['POST'])
def openai(deployment_name, subpath):

    rHeaders = request.headers

    # Retrieve a specific header value
    client_api_key = rHeaders.get('api-key')

    if client_api_key is None or not is_client_api_key_valid(client_api_key):
        return "'api-key' is either invalid or not provided", 403

    _api_url = f"{os.environ.get('OAI_INSTANCE')}-oai-instance-url"
    _api_key = f"{os.environ.get('OAI_INSTANCE')}-oai-instance-api-key"
    api_url = Config.fetch(_api_url.lower())
    api_key = Config.fetch(_api_key.lower())
    headers = {
        "Content-Type" : "application/json",
        "api-key": api_key
    }

    q_api_version = request.args.get('api-version', default=None)
    if q_api_version is None:
        # If the 'api-version' query parameter is missing, return a 400 Bad Request error
        return "Missing 'api-version' parameter", 400

    jdata=json.loads(request.data.decode('utf-8'))

    if "stream" in jdata and jdata["stream"] == True:
        try:
            downstream_response = Response(generate_stream(build_upstream_url(api_url, deployment_name, subpath, q_api_version), jdata, headers), content_type='text/event-stream')
            return downstream_response
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        try:
            upstream_response = requests.post(build_upstream_url(api_url, deployment_name, subpath, q_api_version),
                                         headers=headers, json=jdata, stream=True)
            downstream_response = Response(
                response=upstream_response.iter_content(chunk_size=8192),
                status=upstream_response.status_code,
                content_type=upstream_response.headers.get('content-type')
            )

            for key, value in upstream_response.headers.items():
                downstream_response.headers[key] = value

            downstream_response.headers["x-api-url"] = api_url

            return downstream_response
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/openai/images/<path:subpath>', methods=['POST'])
def openaiImages(subpath):
    rHeaders = request.headers

    # Retrieve a specific header value
    client_api_key = rHeaders.get('api-key')

    if client_api_key is None or not is_client_api_key_valid(client_api_key):
        return "'api-key' is either invalid or not provided", 403

    _api_url = f"{os.environ.get('OAI_INSTANCE')}-oai-instance-url"
    _api_key = f"{os.environ.get('OAI_INSTANCE')}-oai-instance-api-key"
    api_url = Config.fetch(_api_url.lower())
    api_key = Config.fetch(_api_key.lower())
    headers = {
        "Content-Type" : "application/json",
        "api-key": api_key
    }

    q_api_version = request.args.get('api-version', default=None)
    if q_api_version is None:
        # If the 'api-version' query parameter is missing, return a 400 Bad Request error
        return "Missing 'api-version' parameter", 400

    jdata=json.loads(request.data.decode('utf-8'))
    print("build_upstream_image_url(api_url, subpath, q_api_version)", build_upstream_image_url(api_url, subpath, q_api_version))

    try:
        upstream_response = requests.post(build_upstream_image_url(api_url, subpath, q_api_version),
                                        headers=headers, json=jdata, stream=True)
        downstream_response = Response(
            response=upstream_response.iter_content(chunk_size=8192),
            status=upstream_response.status_code,
            content_type=upstream_response.headers.get('content-type')
        )

        for key, value in upstream_response.headers.items():
            downstream_response.headers[key] = value

        downstream_response.headers["x-api-url"] = api_url

        return downstream_response
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/lstatus', methods=['GET'])
def livliness():
    # return "ok", 200
    result1, resp_code = healthcheck_request()
    return result1, resp_code

@app.route('/hcstatus', methods=['GET'])
def hclivliness():
    return "ok", 200

if __name__ == '__main__':
    app.run(debug=True)
