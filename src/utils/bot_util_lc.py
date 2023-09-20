import re
import os
import sys


from typing import Optional, Callable
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models.base import BaseChatModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.callbacks.base import BaseCallbackHandler


# add src for local imports
absolute_path = os.path.dirname(__file__)
relative_path = "../"
full_path = os.path.realpath(os.path.join(absolute_path, relative_path))
sys.path.append(full_path)

# local imports
from utils.common_imports import openai#, setup_logging
from utils.prompt_template import chat_prompt, llm_prompt
# setup_logging()
# print(openai.version.VERSION)


############################################
## LLM Helper Classes

from typing import Optional, Callable, Dict, Any, List, Union
from langchain.schema import AgentAction, AgentFinish, LLMResult

# more info: https://python.langchain.com/en/latest/modules/callbacks/getting_started.html
class FunctionCallbackHandler(BaseCallbackHandler):
    """
    A callback handler class that allows passing in optional callbacks for LLM 
    start, token, and progress events.
    """
    def __init__(
        self, 
        llm_start_cb: Optional[Callable[..., None]] = None, 
        llm_token_cb: Optional[Callable[..., None]] = None, 
        llm_progress_cb: Optional[Callable[..., None]] = None,
        llm_end_cb: Optional[Callable[..., None]] = None,
    ) -> None:
        """
        Initializes a new instance of FunctionCallbackHandler.

        :param llm_start_cb: An optional function that will be called when LLM starts.
        :param llm_token_cb: An optional function that will be called for each generated token.
        :param llm_progress_cb: An optional function that will be called to report LLM progress (total response so far).
        """
        super().__init__()
        self.llm_start_cb = llm_start_cb
        self.llm_token_cb = llm_token_cb
        self.llm_progress_cb = llm_progress_cb
        self.progress = ""
        self.llm_end_cb = llm_end_cb

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        super().on_llm_start(serialized, prompts, **kwargs)
        self.progress = ""
        if self.llm_start_cb is not None:
            self.llm_start_cb(serialized, prompts)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Do nothing."""
        if self.llm_end_cb is not None:
            self.llm_end_cb(response)


    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        super().on_llm_new_token(token, **kwargs)
        self.progress += token
        if self.llm_token_cb is not None:
            self.llm_token_cb(token)
        if self.llm_progress_cb is not None:
            self.llm_progress_cb(self.progress)

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized["name"]
        print(f"\n\n\033[1m> Entering new {class_name} chain...\033[0m")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        print("\n\033[1m> Finished chain.\033[0m")

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Do nothing."""
        pass

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action."""
        print(action)

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        print(output)

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Optional[str],
    ) -> None:
        """Run when agent ends."""
        print(text)

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        print(finish.log)


############################################
## LLM Helpers

def get_llm_config_from_lm(llm, is_stream, llm_prompt=llm_prompt, chat_prompt=chat_prompt):
    llm_config = {
        "llm": llm,
        "is_stream": is_stream,
    }
    llm_config['is_chat'] = isinstance(llm_config['llm'], BaseChatModel)
    if llm_config['is_chat']:
        llm_config['llm_chain'] = LLMChain(llm=llm_config['llm'], prompt=chat_prompt)
    else:
        llm_config['llm_chain'] = LLMChain(llm=llm_config['llm'], prompt=llm_prompt)
    return llm_config


def get_llm_config(
        llmclass,
        is_stream,
        args,
        stream_args,
        chat_prompt_override=None,
        llm_prompt_override=None,
    ):
    # print("Model config:", {**args,**(stream_args if is_stream else {})})
    llm_config = {
        "llm": llmclass(**args,**(stream_args if is_stream else {})),
        "is_stream": is_stream,
    }
    llm_config['is_chat'] = isinstance(llm_config['llm'], BaseChatModel)
    if llm_config['is_chat']:
        llm_config['llm_chain'] = LLMChain(llm=llm_config['llm'], prompt=chat_prompt_override or chat_prompt)
    else:
        llm_config['llm_chain'] = LLMChain(llm=llm_config['llm'], prompt=llm_prompt_override or llm_prompt)
    return llm_config

from utils.config import Config
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
def get_default_chat_model():
    if Config.fetch('azure-openai-api-type') == 'azure':
        return AzureChatOpenAI
    return ChatOpenAI

def get_default_chat_model_config():
    if Config.fetch('azure-openai-api-type') == 'azure':
        return {
            'openai_api_base': Config.fetch('azure-openai-api-base'),
            'openai_api_version': Config.fetch('azure-openai-api-version'),
            'deployment_name': Config.fetch('azure-deployment-name'),
            'openai_api_key': Config.fetch('azure-openai-api-key'),
            'openai_api_type': Config.fetch('azure-openai-api-type'),
        }

    return {
        "model_name": os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo"),
        "openai_api_key": Config.fetch('openai-api-key'),
    }

def get_default_llm(
    llm_progress_cb=None,
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    max_new_token=500,
    chat_prompt_override=None,
    llm_prompt_override=None,
):
    is_stream = (llm_progress_cb is not None)
    cb_handler = FunctionCallbackHandler(
        llm_progress_cb = llm_progress_cb
    )
    STREAM_CONFIG = {
        "streaming": True,
        "verbose": True,
        "callback_manager": BaseCallbackManager([cb_handler])
    }
    LLM_PARAMS = {
        **get_default_chat_model_config(),
        'temperature': temperature,
        'max_tokens': max_new_token,
        'model_name': model_name,
        'max_retries': 20,
    }
    llmclass = get_default_chat_model()
    llm_config = get_llm_config(
        llmclass=llmclass,
        is_stream=is_stream,
        args=LLM_PARAMS,
        stream_args=STREAM_CONFIG,
        chat_prompt_override=chat_prompt_override,
        llm_prompt_override=llm_prompt_override,
    )
    return llm_config

############################################
## Load Encoding and Test Searching

def generate_source_from_doc(doc):
    return f"[{doc.metadata['page_title']}]({doc.metadata['source']})"

def generate_source_object_from_doc(doc):
    return {
        'title': doc.metadata['page_title'],
        'url': doc.metadata['source'],
    }

def generate_context_from_doc(doc):
    metadata = doc.metadata
    content = f"""
    Information about "{metadata['page_title']}":
    {doc.page_content}.
    """.replace('\n', ' ')
    content = re.sub('\s+', ' ', content)
    content = content.strip()

    source = generate_source_from_doc(doc)
    return f"* Content: {content}\n  Source: {source}"

def generate_context_from_doc_md(doc):
    metadata = doc.metadata
    content = f"""
# {metadata['page_title']}
{doc.page_content}.
    """
    content = content.strip()
    content = '  ' + content.replace('\n', '\n  ')

    source = generate_source_from_doc(doc)
    status = metadata.get('status', 'N/A')
    return f"* Status:{status}\n  Source: {source}\n  Content:\n{content}\n  "

def generate_context_from_doc_md_exp(doc):
    metadata = doc.metadata
    content = f"""
# {metadata['page_title']}
{doc.page_content}.
    """
    content = content.strip()
    # content = '  ' + content.replace('\n', '\n  ')

    source = generate_source_from_doc(doc)
    return f"<source><content>{content}</content><link>{source}</link></source>"

def test_chunk_searching(q, search_index):
    docs = search_index.similarity_search(q, k=2)
    for doc in docs:
        print(generate_context_from_doc_md(doc))

############################################
## Answer Questions Based on Context

def get_context(
    q,
    search_index,
    llm,
    search_index_kwargs={},
    context_doc_count=5,
    max_token_count=2500,
    context_export_func=generate_context_from_doc_md,
    transform_docs_func=None,
):
    docs = search_index.similarity_search(q, k=context_doc_count, **search_index_kwargs)
    if transform_docs_func is not None:
        docs = transform_docs_func(docs)

    # find max idx we can fit
    for max_idx in range(1, len(docs) + 1):
        context_candidate = "\n".join([context_export_func(doc) for doc in docs[:max_idx]])
        token_count = llm.get_num_tokens(context_candidate)
        # maximum tokens allowed in the prompt
        if token_count > max_token_count:
            max_idx -= 1
            break

    docs = docs[:max_idx]
    context = "\n".join([context_export_func(doc) for doc in docs])
    return context, docs

def get_context_with_scores(
    q,
    search_index,
    llm,
    search_index_kwargs={},
    context_doc_count=5,
    max_token_count=2500,
    context_export_func=generate_context_from_doc_md,
    transform_docs_func=None,
):
    docs_and_scores = search_index.similarity_search_with_score(q, k=context_doc_count, **search_index_kwargs)

    docs = [doc for doc, _ in docs_and_scores]
    scores = [score for _, score in docs_and_scores]
    if transform_docs_func is not None:
        docs = transform_docs_func(docs)

    # find max idx we can fit
    max_idx = 1
    for max_idx in range(1, len(docs) + 1):
        context_candidate = "\n".join([context_export_func(doc) for doc in docs[:max_idx]])
        token_count = llm.get_num_tokens(context_candidate)
        # maximum tokens allowed in the prompt
        if token_count > max_token_count:
            max_idx -= 1
            break

    docs = docs[:max_idx]
    context = "\n".join([context_export_func(doc) for doc in docs])
    return context, docs, scores

from datetime import date
def get_today_str():
    today = date.today()
    return today.strftime("%b-%d-%Y")
