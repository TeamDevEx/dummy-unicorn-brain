# add src for local imports
import os, sys
absolute_path = os.path.dirname(__file__)
relative_path = "../"
full_path = os.path.realpath(os.path.join(absolute_path, relative_path))
sys.path.append(full_path)

from api.v1_0 import api_v1_0
from api.v2_0 import api_v2_0
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from utils.config import Config

import logging
logger = logging.getLogger("gunicorn.info")

tags_metadata = [
    {
        "name": "llm",
        "description": "Common operations for interacting with platform's available llm models",
    },
    {
        "name": "vectordb",
        "description": "Common operations for interacting with the platform's vector database",
    },
    {
        "name": "common",
        "description": "Common operations",
    },
    {
        "name": "usecases",
        "description": "Operations for specific use cases.",
    },
    {
        "name": "V1.0",
        "description": "Operations available in v1.0 of the API",
    },
]

app = FastAPI(
    debug = False,
    docs_url = Config.fetch('V2_0_URL'),
    redoc_url = Config.fetch('V2_0_URL') + '/redoc',
    title = "Unicorn Brain - TELUS' Generative AI Platform",
    description = "unicorn.brain is TELUS' common Generative AI Platform that enables experimentation, rapid prototyping and building out gen AI solutions for TELUS in a shared, and safe way!",
    openapi_tags = tags_metadata,
)

app.include_router(api_v1_0)
app.include_router(api_v2_0)


@app.get('/', response_class=HTMLResponse, include_in_schema=False)
def index(request: Request):
    html_content = f'''<html>
    <body>
        Please use latest version <a href="{Config.fetch('V2_0_URL')}">{Config.fetch('V2_0_URL')}</a>
    </body>
</html>
'''
    return HTMLResponse(content=html_content, status_code=200)
