# flake8: noqa
from datetime import datetime

from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

def _get_datetime():
    now = datetime.now()
    return now.strftime("%m/%d/%Y")

condense_prompt_template = """
Given the following conversation and a question, rewrite the question in the same language.
If the follow up question is irrelevant to the conversation, do not include the conversation in the standalone question
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_prompt_template)

chat_template = """
You are a helpful chat assistant and a sales coach. You were created by AI Accelerator. 
Today is {date}.
Provide short responses to user's questions from given context below.
If the answer is not contained within the context, say "Sorry, the content required to answer your query 
does not seem to be included in the PSO documentation at this time. We will pass this on to the relevant team for further investigation. Thank you!"

=========
{context}
=========
"""   

MESSAGE_PROMPT = PromptTemplate(
    template=chat_template, 
    input_variables=["context"], 
    partial_variables={"date": _get_datetime}
)
messages = [
    SystemMessagePromptTemplate(prompt=MESSAGE_PROMPT),
    HumanMessagePromptTemplate.from_template("{question}"),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)