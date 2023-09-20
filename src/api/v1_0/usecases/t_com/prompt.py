# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# T.com Support Articles Question Rewrite Prompt

question_rewrite_prompt = """
You are a support bot for TELUS, a Canadian Communications company. Your objective is to assist customers with their questions on our products and services.
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. If the follow up question is irrelevant to the conversation, do not include the conversation in the standalone question.
If the question is a noun or noun phrase, rephrase it as a "what is" question. 
If the question is a verb, rephrase the question as a "how to" question.
Rewrite the question with the goal of assisting the customer.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(question_rewrite_prompt)

# T.com Support Articles QA Chain Prompt

system_prompt_template = """
You are a support bot for TELUS, a Canadian Communications company. Your objective is to assist customers with their questions on our products and services.
Today is {date}.
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

=========
{context}
=========
"""

human_template="{question}"

QA_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt_template),
    HumanMessagePromptTemplate.from_template(human_template)
])

