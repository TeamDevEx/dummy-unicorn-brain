from datetime import date
from langchain.docstore.document import Document
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from utils import bot_util_lc, milvus_helpers
from utils.config import Config

def get_site_options():
    return {
        "PureFibre": 1,
        "FFH": 6,
        "Consumer Mobility": 8,
        "Koodo": 9,
        #"PC Mobile": 10, # Removed Aug 15, 2023
        "Public Mobile": 11,
        "SmartHome": 15,
        "Business Mobility": 23
    }

def get_site_ids():
    return list(get_site_options().values())

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

def transform_doc_with_site_ids(doc, site_ids):
    site_id_idx = -1
    for idx, site_id in enumerate(doc.metadata['site_id_list']):
        if site_id in site_ids:
            site_id_idx = idx
    
    if site_id_idx < 0:
        raise Exception(f"Site ID {site_ids} not found in doc.metadata['site_id_list']!")
    
    new_metadata = { **doc.metadata }
    new_metadata['page_title'] = \
        f"{new_metadata['site_name_list'][site_id_idx]}: {new_metadata['page_title']}"
    new_metadata['source'] = new_metadata['site_url_list'][site_id_idx]

    return Document(
        page_content=doc.page_content,
        metadata=new_metadata,
    )

def transform_doc_list_with_site_ids(docs, site_ids):
    return [transform_doc_with_site_ids(doc, site_ids) for doc in docs]

def get_milvus_expr_for_site_ids(site_ids):
    expr_list = [f"(metadata_site_id_{site_id} == true)" for site_id in site_ids]
    return " || ".join(expr_list)

def get_context_default_config(query, site_ids, search_index, llm, config_override={}):
    default_config = {
        'search_index_kwargs': {
            'expr': get_milvus_expr_for_site_ids(site_ids),
        },
        'context_doc_count': 8,
        'max_token_count': 2500,
        'context_export_func': bot_util_lc.generate_context_from_doc_md,
        'transform_docs_func': lambda docs: transform_doc_list_with_site_ids(docs, site_ids)
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
        'id': str(doc.metadata['page_id']),
        'url': doc.metadata['source'],
    }

def check_llm_default_config():
    llm_config = get_default_llm()
    llm = llm_config['llm']
    is_chat = llm_config['is_chat']
    try:
        if is_chat:
            llm([HumanMessage(content="ping")])
        else:
            llm("ping")
        return True
    except:
        return False

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate

response_schemas = [
    ResponseSchema(name="user_message", description="message to be shown to the user in case further information is required"),
    ResponseSchema(name="clear_question", description="user's question written more clearly to be searched"),
    ResponseSchema(name="clarity_score", description="a score of 1-5 on how clear user's question is"),
    ResponseSchema(name="status_filter", description="status filter needed to apply. options are \"current\" (only content about current plans/promotion/pricing/etc) and \"all\" (all content, including non-current and expired). If asking about a specific plan code, use \"all\"."),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()
rewrite_cc_agent_question_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
"""
You are a tool to help clarify a question for TELUS customer support agents.
Below is a summary of the conversation so far, and a new question asked
by the call center agent. You are tasked to identify how
clear that question is, and rewrite it to make it more clear with minimal changes.
Rewrite the user's question in clearer terms in the clear_question field.
Don't expand abbreviations, but re-write them in capital letters.
If you have already responded to the question, repeat your response, don't ask the user to refer to the previous response.

Assumptions:
If the user asks about "TV" without specifying which TV product (Pik TV/Optik TV/Satellite TV/Guest TV/etc), replace it with "Optik TV"
If the user asks about "mobility" without specifying which product, replace it with "TELUS Mobility Consumer Postpaid"
If the user asks about their "phone" without specifying which product (wireless home phone/cellphone/etc), replace it with mobile phone

Here are some internal abbreviations and acronyms we use, if you see one use both in clear_question:
"DT1" => "Desktop One"
"STB" => "Set Top Box"
"THTL" => "Take Home Trade Later"
"EPP" => "Employee Purchase Program"
"DRA" => "Disaster Recovery Application"
"TFD" => "TELUS Family Discount"
"MBG" => "Money Back Guarantee"
"HS" => "High Speed Internet"
"SL" => "Single Line"
"CX" => "Customer"
"TOWN" => "Transfer of Ownership"
"TLC" => "Termination Liability Charges"
"aal" => "add a line"
"aces" => "advanced channel experience support"
"addrs" => "address"
"adj" => "adjustment"
"aha" => "at home agent"
"aie" => "activate in error"
"c2f" => "copper to fibre"
"ccl" => "critical care list"
"ccp" => "credit card payment"
"cef" => "customer equipment form"
"cfco" => "call forward fixed central office"
"cil" => "corporate individual liability"
"clos" => "channel live order support"
"cls" => "customer loyalty specialist"
"cms" => "central monitoring situation"
"COID" => "central office id"
"cpo" => "certified pre-owned"
"csa" => "customer service agreement"
"csr" => "customer service rep"
"css" => "consumer sales solutions"
"cx" => "customer"
"d2c" => "direct to consumer"
"dak" => "denies all knowledge"
"did" => "direct in dial"
"dmc" => "dispatch management centre"
"dnc" => "do not call"
"dnd" => "direct in dial"
"dsr" => "do sooner request"
"dt1" => "desktop one"
"eft" => "electronic funds transfer"
"eid" => "equifax eidverifier"
"emt" => "escalations management team"
"epp" => "employee purchase plan"
"ets" => "escalation tracking system"
"fpp" => "flexible payment plan"
"ftnp" => "first time no pay"
"gwp" => "gift with purchase"
"hp" => "home phone"
"hs" => "high speed"
"hsia" => "high speed internet access"
"ils" => "individual line service"
"iot" => "internet of things"
"l&r" => "loyalty & retention""
"ld" => "long distance"
"lwc" => "living well companion"
"m&h" => "mobile & home"
"mdu" => "multiple dwelling unit"
"mep" => "multi-element plan"
"mhd" => "mobility help desk"
"mog" => "mobility for good"
"mpia" => "months paid in advance"
"mss" => "mobility sales system"
"mtm" => "month to month"
"naas" => "network as a service"
"natl" => "national"
"nho" => "new home offer"
"npa" => "area code"
"obd" => "order based drop"
"pap" => "pre-authorized payments"
"parrs" => "payment arrangements"
"poa" => "power of attorney"
"ponp" => "pending on pending"
"ppu" => "pay per use"
"ppv" => "pay per view"
"sacg" => "service address control group"
"shs" => "smarthome security"
"SIP" => "sales incentive program"
"sl" => "single line"
"stb" => "set top box"
"t&m" => "time and materials"
"TLC" => "termination liability charge"
"tos" => "telus online security"
"town" => "transfer of ownership"
"ts" => "tech support"
"tsd" => "telus service discount"
"ubb" => "usage based billing"
"vm" => "voice mail"
"vod" => "video on demand"
"voip" => "voice over ip"
"wln" => "wireline"
"wls" => "wireless"
"wnp" => "wireless number portability"
"WHP" => "Wireless Home Phone"

Here are some well-known abbreviations that don't need to be expanded, leave them as is in the clear_question:
"FFH"
"MNH"
"SMS"

Always return your response in the following format. {format_instructions}
\nSummary:\n{summary}\n\n
"""
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: I have a very slow internet",
        ),
        AIMessagePromptTemplate.from_template(
"""
```json
{{
	"user_message": "",
	"clear_question": "The customer says they have a very slow internet, what are the troubleshooting steps?",
	"clarity_score": "3",
	"status_filter": "current"
}}
```"""
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: What is the Take Home Trade Later program?"
        ),
        AIMessagePromptTemplate.from_template(
"""
```json
{{
	"user_message": "",
	"clear_question": "What is the Take Home Trade Later program?",
	"clarity_score": "5",
	"status_filter": "current"
}}
```"""
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: portin in a line"
        ),
        AIMessagePromptTemplate.from_template(
"""
```json
{{
	"user_message": "",
	"clear_question": "How do I port in a line?",
	"clarity_score": "5",
	"status_filter": "current"
}}
```"""
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: mobility offers"
        ),
        AIMessagePromptTemplate.from_template(
"""
```json
{{
	"user_message": "",
	"clear_question": "What are the current TELUS Mobility Consumer Postpaid plans we offer?",
	"clarity_score": "3",
	"status_filter": "current"
}}
```"""
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: what are the mobility offers"
        ),
        AIMessagePromptTemplate.from_template(
"""
```json
{{
	"user_message": "",
	"clear_question": "What are the current TELUS Mobility Consumer Postpaid plans we offer?",
	"clarity_score": "4",
	"status_filter": "current"
}}
```"""
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: bill"
        ),
        AIMessagePromptTemplate.from_template(
"""
```json
{{
	"user_message": "I'm sorry, but I don't understand what you mean by \"bill\", could you clarify?",
	"clear_question": "What is a bill?",
	"clarity_score": "1",
	"status_filter": "current"
}}
```"""
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: how to verify"
        ),
        AIMessagePromptTemplate.from_template(
"""
```json
{{
	"user_message": "I'm sorry, could you clarify what you want to verify?",
	"clear_question": "How do I verify a customer?",
	"clarity_score": "2",
	"status_filter": "current"
}}
```"""
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: What is THTL?"
        ),
        AIMessagePromptTemplate.from_template(
"""
```json
{{
	"user_message": "",
	"clear_question": "What is Take Home Trade Later (THTL)?",
	"clarity_score": "5",
	"status_filter": "current"
}}
```"""
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: What are the current FFH Offers?"
        ),
        AIMessagePromptTemplate.from_template(
"""
```json
{{
	"user_message": "",
	"clear_question": "What are the current FFH Offers?",
	"clarity_score": "5",
	"status_filter": "current"
}}
```"""
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: MNH offers"
        ),
        AIMessagePromptTemplate.from_template(
"""
```json
{{
	"user_message": "",
	"clear_question": "What are the current MNH Offers?",
	"clarity_score": "4",
	"status_filter": "current"
}}
```"""
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: dt1 documentation guidelines"
        ),
        AIMessagePromptTemplate.from_template(
"""
```json
{{
	"user_message": "",
	"clear_question": "What are the documentation guidelines for Desktop One (DT1)?",
	"clarity_score": "4",
	"status_filter": "current"
}}
```"""
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: did we ever have manitoba only plans"
        ),
        AIMessagePromptTemplate.from_template(
"""
```json
{{
	"user_message": "",
	"clear_question": "Did TELUS ever offer Manitoba-only plans?",
	"clarity_score": "4",
	"status_filter": "all"
}}
```"""
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: YC PLATINUM ULNW 100 plan"
        ),
        AIMessagePromptTemplate.from_template(
"""
```json
{{
	"user_message": "",
	"clear_question": "What are the details of the YC PLATINUM ULNW 100 plan?",
	"clarity_score": "4",
	"status_filter": "all"
}}
```"""
        ),
        # showing repeat is ok
        HumanMessagePromptTemplate.from_template(
            "Question: YC PLATINUM ULNW 100 plan"
        ),
        AIMessagePromptTemplate.from_template(
"""
```json
{{
	"user_message": "",
	"clear_question": "What are the details of the YC PLATINUM ULNW 100 plan?",
	"clarity_score": "4",
	"status_filter": "all"
}}
```"""
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: is YOURCHOICE PREM+ UL NW 90 still available?"
        ),
        AIMessagePromptTemplate.from_template(
"""
```json
{{
	"user_message": "",
	"clear_question": "Is the YOURCHOICE PREM+ UL NW 90 plan still available?",
	"clarity_score": "5",
	"status_filter": "current"
}}
```"""
        ),
        HumanMessagePromptTemplate.from_template("Question: {question}")  
    ],
    input_variables=["question", "summary"],
    partial_variables={"format_instructions": format_instructions}
)

def rewrite_cc_agent_question_default_config(logger, question, summary=""):
    llm_config = get_default_llm(
        temperature=0
    )
    llm = llm_config['llm']
    is_chat = llm_config['is_chat']
    if is_chat:
        # return llm(rewrite_cc_agent_question_prompt.format_messages(
        #     summary=summary, question=question,
        # )).content
        _input = rewrite_cc_agent_question_prompt.format_prompt(question=question, summary='')
        try:
            output = llm(_input.to_messages())
            return output_parser.parse(output.content)
        except Exception:
            logger.exception('error while parsing the rewriting response, setting default')
            # return {
            #     'user_message': 'Sorry I don\'t understand the question, could you clarify a bit?',
            #     'clear_question': question,
            #     'clarity_score': '2',
            #     'status_filter': 'all',
            # }
            return {
                'user_message': '',
                'clear_question': question,
                'clarity_score': '4',
                'status_filter': 'current',
            }
    else:
        raise NotImplementedError
        # llm("ping")

def answer_question_default_config(
    query,
    site_ids=None,
    search_index=None,
    progress_object_callback=None,
    llm_config_override={},
    context_config_override={},
):
    if site_ids is None:
        site_ids = get_site_ids()

    if search_index is None:
        embedding_func = get_default_embedding_func()
        search_index = milvus_helpers.AIAMilvus(
            embedding_function=embedding_func,
            connection_args=None, # use environment variables
            collection_name="onesource",
            milvus_bypass_proxy=True,
        )
    
    context_config_default = {}
    context_args = {
        **context_config_default,
        **context_config_override,
    }

    # default llm chain for building context
    default_llm_config = get_default_llm(**llm_config_override)

    context_str, docs, scores = get_context_default_config(
        query,
        site_ids,
        search_index,
        llm=default_llm_config['llm'],
        config_override=context_args
    )

    # build prompt string
    prompt = ""
    if default_llm_config['is_chat']:
        prompt = default_llm_config['llm_chain'].prompt.format_prompt(**{"context": context_str, "question": query, "date": bot_util_lc.get_today_str()}).to_string()
    else:
        prompt = default_llm_config['llm_chain'].prompt.format_prompt(**{"context": context_str, "question": query, "date": bot_util_lc.get_today_str()}).text

    # default llm to use for context
    def get_default_response_object(response):
        links = []
        link_urls = set()
        for doc in docs:
            link = generate_source_object_from_doc(doc)
            if link['url'] not in link_urls:
                links.append(link)
                link_urls.add(link['url'])

        response_object = {
            'response': response,
            'links': links,
            'docs': docs,
            'scores': scores,
            'prompt': prompt,
        }
        
        if progress_object_callback is not None:
            progress_object_callback(response_object)

        return response_object

    llm_config_default = {
        'llm_progress_cb': get_default_response_object
    }
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

    if len(docs) > 0:
        # TODO: adapt response to non-chat llm models (e.g., davinci)
        response = llm_chain.run({"context": context_str, "question": query, "date": bot_util_lc.get_today_str()})
    else:
        response = "No results found that match the criteria selected."

    return get_default_response_object(response)

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

# extract pgid from the link
def extract_pgid_from_source(source):
    import re
    match = re.search(r"pgid=(\d+)", source)
    if match:
        return match.group(1)
    return None
