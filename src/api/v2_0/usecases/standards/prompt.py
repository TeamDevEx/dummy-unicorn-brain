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
Answer the question at the end as accurately as you can based on the pages of the context. 
If the answer is not contained within the provided context then ALWAYS start your answer stating the following: \
"I did not find related content within TELUS internal documents. This response is generated using external knowledge.\n". \
Then, try to answer the question. 

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
If the answer is not contained within the provided context then ALWAYS start your answer stating the following: \
"I did not find related content within TELUS internal documents. This response is generated using external knowledge.\n". \
Then, try to answer the question.

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