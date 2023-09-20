
# python imports
import os
import json
import re
import time
import logging

# package imports
from tqdm.auto import tqdm

# local imports
from utils.config import Config

# import openai and do its config
import openai
openai.proxy = {"http": os.getenv('HTTP_PROXY', ''), "https": os.getenv('HTTPS_PROXY', '')}


def setup_http_proxy():
    proxy = 'http://198.161.14.25:8080'

    # set internet access proxy
    os.environ['http_proxy'] = proxy
    os.environ['HTTP_PROXY'] = proxy
    os.environ['https_proxy'] = proxy
    os.environ['HTTPS_PROXY'] = proxy

def load_dot_env(env_rel_path = None):
    # get dotenv path
    import os
    import sys

    # loading environment variabels
    from dotenv import load_dotenv

    # if env path is not provided, use bot .env
    if env_rel_path is None:
        env_rel_path = '../../.env'
        absolute_path = os.path.dirname(__file__)
    else:
        absolute_path = os.getcwd()
    
    env_full_path = os.path.realpath(os.path.join(absolute_path, env_rel_path))
    print(f'loading .env from {env_full_path}')
    load_dotenv(dotenv_path=env_full_path)

def setup_logging(logger):
    """
    Sets up the logging structure of the project using the built-in logging library.
    Returns None
    """
    absolute_path = os.path.dirname(__file__)
    output_relative_path = '../../logs/os_bot.log'
    execution_log_file = os.path.abspath(os.path.join(absolute_path, output_relative_path))

    # set a format which is simpler for console use
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p'
    )

    print(f'Log path: {execution_log_file}')

    file_handler = logging.FileHandler(
        filename=execution_log_file,
        mode='a',
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    file_handler.formatter.converter = time.localtime
    logger.addHandler(file_handler)


    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    console.formatter.converter = time.localtime
    # add the handler to the root logger
    # logging.getLogger('').addHandler(console)
    logger.addHandler(console)

    logger.debug(f'Log path: {execution_log_file}')

# check the content filter doesn't have unmatched parenthesis (avoid injection attacks)
def is_valid_content_filter(input_str):
    stack = []
    opening_brackets = set(['(', '[', '{'])
    closing_brackets = set([')', ']', '}'])
    bracket_pairs = {
        ')': '(',
        ']': '[',
        '}': '{'
    }

    for char in input_str:
        if char in opening_brackets:
            stack.append(char)
        elif char in closing_brackets:
            if not stack or bracket_pairs[char] != stack[-1]:
                return False
            stack.pop()

    return len(stack) == 0