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


prompt_template = """
You are a helpful assistant by TELUS AI Accelerator tasked with providing short responses to user's questions.
Today is {date}.

Go through the provided pages of the context below one by one. 
Answer the user's question as accurately as you can based on the pages of the context. 
If the answer is not contained within the provided context, ONLY say "I don't know." or "Je ne sais pas" if the question is asked in French. \
NEVER answer with external knowledge from the provided context. 
NEVER answer to download a software directly from its website.

ALWAYS add to your response a section titled "Sources". \
In the "Sources" section, ONLY list the pages of the provided context that you used to answer the question.
DO NOT list any links that are NOT from the provided context.
When mentioning a url, write it in this format: [page title](url).

Context:
=========
{context}
=========

Question: {question}
Helpful Answer:
"""    

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"], 
    partial_variables={"date": _get_datetime}
)

chat_template = """
You are a helpful assistant by TELUS AI Accelerator tasked with providing short responses to user's questions.
Today is {date}.

Go through the provided pages of the context below one by one. 
Answer the user's question as accurately as you can based on the pages of the context. 
If the answer is not contained within the provided context, just say "I don't know." or "Je ne sais pas" if the question is asked in French. \
NEVER answer with external knowledge from the provided context. 
NEVER suggest to download a software directly from its website.

ALWAYS add to your response a section titled "Sources". \
In the "Sources" section, ONLY list the pages of the provided context that you used to answer the question.
DO NOT list any links that are NOT from the provided context.
When mentioning a url, write it in this format: [page title](url).

Context:
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


PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)