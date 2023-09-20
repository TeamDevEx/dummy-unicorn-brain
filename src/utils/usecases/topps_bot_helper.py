
from langchain.docstore.document import Document
from utils import bot_util_lc, milvus_helpers
from utils.config import Config

def get_default_llm(
    llm_progress_cb=None,
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    max_new_token=500,
    **kwargs,
):
    return bot_util_lc.get_default_llm(
        llm_progress_cb=llm_progress_cb,
        model_name=model_name,
        temperature=temperature,
        max_new_token=max_new_token,
        **kwargs
    )

def get_default_embedding_func():
    from langchain.embeddings.openai import OpenAIEmbeddings

    if Config.fetch('azure-openai-api-type') == 'azure':
        return OpenAIEmbeddings(
            openai_api_version=Config.fetch('azure-openai-api-version'),
            openai_api_type=Config.fetch('azure-openai-api-type'),
            openai_api_key=Config.fetch('azure-openai-api-key'),
            openai_api_base=Config.fetch('azure-openai-api-base'),
            deployment=Config.fetch('azure-embedding-deployment-name'),
        )
    
    embedding_func = OpenAIEmbeddings(openai_api_key=Config.fetch('openai-api-key'))
    return embedding_func

def get_context_default_config(query, search_index, llm, config_override={}):
    default_config = {
        'search_index_kwargs': {
            'expr': '', # Do we need to filter?
        },
        'context_doc_count': 8,
        'max_token_count': 2500,
        'context_export_func': bot_util_lc.generate_context_from_doc_md,
        # 'transform_docs_func': lambda docs: docs
    }
    config = {
        **default_config,
        **config_override,
    }

    try:
        context_str, docs, scores = bot_util_lc.get_context_with_scores(
            query,
            search_index,
            llm=llm,
            **config
        )
        return context_str, docs, scores
    # in case the index doesn't support scores, return 0 for everything
    except NotImplementedError:
        context_str, docs = bot_util_lc.get_context(
            query,
            search_index,
            llm=llm,
            **config
        )
        scores = [0 for _ in docs]
        return context_str, docs, scores

def generate_source_object_from_doc(doc):
    return {
        'title': doc.metadata['page_title'],
        'url': doc.metadata['source'],
    }

def check_llm_default_config():
    llm_config = get_default_llm()
    llm = llm_config['llm']
    is_chat = llm_config['is_chat']
    try:
        if is_chat:
            from langchain.schema import HumanMessage
            llm([HumanMessage(content="ping")])
        else:
            llm("ping")
        return True
    except:
        return False

def answer_question_default_config(
    query,
    search_index=None,
    llm_config_override={},
    context_config_override={},
):
    if search_index is None:
        embedding_func = get_default_embedding_func()
        search_index = milvus_helpers.AIAMilvus(
            embedding_function=embedding_func,
            connection_args=None, # use environment variables
            collection_name="topps",
            milvus_bypass_proxy=True,
        )

    llm_config_default = {}
    llm_args = {
        **llm_config_default,
        **llm_config_override,
    }

    llm_config = get_default_llm(
        **llm_args
    )

    llm = llm_config['llm']
    is_chat = llm_config['is_chat']
    llm_chain = llm_config['llm_chain']

    context_config_default = {}
    context_args = {
        **context_config_default,
        **context_config_override,
    }

    context_str, docs, scores = get_context_default_config(
        query,
        search_index,
        llm,
        config_override=context_args
    )

    links = []
    link_urls = set()
    for doc in docs:
        link = generate_source_object_from_doc(doc)
        if link['url'] not in link_urls:
            links.append(link)
            link_urls.add(link['url'])

    prompt = ""
    if is_chat:
        prompt = llm_chain.prompt.format_prompt(**{"context": context_str, "question": query, "date": bot_util_lc.get_today_str()}).to_string()
    else:
        prompt = llm_chain.prompt.format_prompt(**{"context": context_str, "question": query, "date": bot_util_lc.get_today_str()}).text

    if len(docs) > 0:
        # TODO: adapt response to non-chat llm models (e.g., davinci)
        response = llm_chain.run({"context": context_str, "question": query, "date": bot_util_lc.get_today_str()})
    else:
        response = "No results found that match the criteria selected."

    return {
        'response': response,
        'links': links,
        'docs': docs,
        'scores': scores,
        'prompt': prompt,
    }

def generate_other_useful_links(res, max_links=5):
    links_md = ""
    for link in res['links'][:max_links]:
        if link['url'] not in res['response']:
            links_md += f"- [{link['title']}]({link['url']})\n"

    # all found links are rendered
    if not links_md:
        return ""

    links_md = "Other potentially useful sources:\n" + links_md
    return links_md
