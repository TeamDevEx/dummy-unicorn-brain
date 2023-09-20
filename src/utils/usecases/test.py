import os, sys

# add src for local imports
absolute_path = os.path.dirname(__file__)
relative_path = "../../../../"
full_path = os.path.realpath(os.path.join(absolute_path, relative_path))
sys.path.append(full_path)

# import bot_interface modules
from utils.common_imports import *
# setup_http_proxy() # Note: try to use env variables if possible
# load .env
load_dot_env()

from utils import milvus_helpers
from utils.usecases import onesource_bot_helper

embedding_func = onesource_bot_helper.get_default_embedding_func()
search_index = milvus_helpers.AIAMilvus(
    embedding_function=embedding_func,
    connection_args=None, # use environment variables
    collection_name = "onesource",
    milvus_bypass_proxy=None, # use environment variables
)

site_ids = onesource_bot_helper.get_site_ids()

if __name__ == "__main__":
    q = 'who is ADT?'
    res = onesource_bot_helper.answer_question_default_config(
        query=q,
        site_ids=site_ids,
        search_index=search_index,
    )
    print(res['response'])
    print(res)