from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate

#DEFAULT SUMMARIZATION PROMPTS

default_stuff_prompt_template = """
Write a concise summary of the following,
Keep in mind a good summary is 10 to 15 percent the length of the provided text
```{text}```
SUMMARY:
"""
default_stuff_prompt = PromptTemplate(template=default_stuff_prompt_template, input_variables=["text"])

default_map_prompt_template = """
Write a summary of this chunk of text that includes the main points and any important details.
Return your response in bullet points which covers the key points of the text.
```{text}```
"""
default_map_prompt = PromptTemplate(template=default_map_prompt_template, input_variables=["text"])

default_combine_prompt_template = """
Write a concise summary of the following text delimited by triple backquotes.
Return your response in bullet points which covers the key points of the text.
```{text}```
BULLET POINT SUMMARY:
"""
default_combine_prompt = PromptTemplate(template=default_combine_prompt_template, input_variables=["text"])

#MILVUS SUMMARIZATION PROMPTS

milvus_stuff_prompt_template = """
Could you please provide a summary of the given text, including all key points and supporting details? 
The summary should be comprehensive and accurately reflect the main message and arguments presented in the original text, while also being concise and easy to understand. 
To ensure accuracy, please read the text carefully and pay attention to any nuances or complexities in the language. 
Additionally, the summary should avoid any personal biases or interpretations and remain objective and factual throughout.
Keep in mind a good summary is 10 to 15 percent the length of the provided text
```{text}```
SUMMARY:
"""
milvus_stuff_prompt = PromptTemplate(template=milvus_stuff_prompt_template, input_variables=["text"])

milvus_map_prompt_template = """
Write a summary of this chunk of text that includes the main points and any important details.
Return your response in bullet points which covers the key points of the text.
{text}
"""
milvus_map_prompt = PromptTemplate(template=milvus_map_prompt_template, input_variables=["text"])

milvus_combine_prompt_template = """
Write a concise summary of the following text delimited by triple backquotes.
Return your response in bullet points which covers the key points of the text.
```{text}```
BULLET POINT SUMMARY:
"""
milvus_combine_prompt = PromptTemplate(template=milvus_combine_prompt_template, input_variables=["text"])

#CHAT SUMMARIZATION PROMPTS

chat_stuff_prompt_template = """
Below a conversation log between a customer and a customer service agent
Could you please provide a summary of the given text, including all key points
This summary should serve as quick recap for a new service agent to understand what the customers problem is and what has already been done to fix it.
To ensure accuracy, please read the text carefully and pay attention to any nuances or complexities in the language. 
The summary should avoid any personal biases or interpretations and remain objective and factual throughout.
Try to keep the summary short and easy to digest
```{text}```
Please Return your summary in the following format:
ISSUE: Quick Sentence of what the customers issue
OUTCOME: Quick Sentence of the conversations outcome 
SUMMARY:Write a bullet point summary here
"""
chat_stuff_prompt = PromptTemplate(template=chat_stuff_prompt_template, input_variables=["text"])

chat_map_prompt_template = """
Below is a chunk of a conversation log between a customer and a customer service agent
Gather the main points and any important details of this conversation and return them in bullet points
To ensure accuracy, please read the text carefully and pay attention to any nuances or complexities in the language. 
Additionally, if you could note the sentiment of the customer during the call
{text}
"""
chat_map_prompt = PromptTemplate(template=chat_map_prompt_template, input_variables=["text"])

chat_combine_prompt_template = """
Write a concise summary of the following text delimited by triple backquotes.
The final summary should identify the customers issue, how the issue was resloved, and the customers sentiment
Return your final summary of the conversation in a paragraph
```{text}```
"""
chat_combine_prompt = PromptTemplate(template=chat_combine_prompt_template, input_variables=["text"])

#KNOWLEDGE ASSIST PROMPTS

knowledge_assist_stuff_prompt_template = """
Below is a chunk of a conversation log between a customer and a customer service agent
Gather the issue of the customer and how the customer service agent has tried to resolve it if the information is available
Keep your summary short, clear, and concise so it could be used a prompt for a search endpoint
Additionally, try to classify what the issue into a general technical issue such as billing, troubleshooting, etc.
{text}
Return the summary as follows:
Category: <Put the appropriate Category here as a bullet point>
ISSUE: <Describe the Customers Issue Here>
"""
ka_stuff_prompt = PromptTemplate(template=knowledge_assist_stuff_prompt_template, input_variables=["text"])

knowledge_assist_map_template = """
Below is a chunk of a conversation log between a customer and a customer service agent
Gather the issue of the customer and how the customer service agent has tried to resolve it if the information is available
Return the customer's issue with any important details
Additionally, try to classify what the issue into a general technical issue such as billing, troubleshooting, etc.
{text}
"""
ka_map_prompt = PromptTemplate(template=knowledge_assist_map_template, input_variables=["text"])

knowledge_assist_combine_template = """
Write a summary of the customers issue given the conversation between a customer and customer service agent
Keep your summary short, clear, and concise so it could be used a prompt for a search endpoint
'''{text}'''
"""
ka_combine_prompt = PromptTemplate(template=knowledge_assist_combine_template, input_variables=["text"])