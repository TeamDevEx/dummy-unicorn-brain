
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# prompt templates
system_prompt_template = """
You are a helpful assistant by TELUS AI Accelerator tasked with providing short responses to user's questions.
Today is {date}.
Go through the provided pages of the context below one by one. 
Answer the user's question as accurately as you can based on the pages of the context. 
If the answer is not contained within the context, say "I don't know." don't try to make up an answer. 
create a final answer with references ("Sources").
ALWAYS add to your response a section titled "Sources" and list in it the Sources you used to answer the question. 
When mentioning a url, write it in this format: [page title](url).

=========
{context}
=========
"""

example_question_template = "example question"
example_response_template = """example response from source 1. example response from source 2.

Sources:
- [page title](page link)
- [another page title](another page link)
"""

example_question_template_2 = "another example question"
example_response_template_2 = """example response from source 1

Sources:
- [page title](page link)
"""

def get_message_str_from_list(message_sample):
    return "\n".join([m.content for m in message_sample])

# chat api
system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)
human_template="{question}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    HumanMessagePromptTemplate.from_template(example_question_template),
    AIMessagePromptTemplate.from_template(example_response_template),
    HumanMessagePromptTemplate.from_template(example_question_template_2),
    AIMessagePromptTemplate.from_template(example_response_template_2),
    human_message_prompt,
])

if __name__ == "__main__":
    print('Testing chat prompt template:')
    message_sample = chat_prompt.format_prompt(context="Sample Context", question="Who are you?", date="sample-date").to_messages()
    message_sample_str = get_message_str_from_list(message_sample)
    print(message_sample_str)

# for other llms (not chat interface) we need a full prompt
llm_template_str = f"""{system_prompt_template}

Q: {example_question_template}
A: {example_response_template}

Q: {{question}}
A:"""
llm_prompt=PromptTemplate(
    template=llm_template_str,
    input_variables=["context", "question", "date"],
)


if __name__ == "__main__":
    print('\n-------------------\nTesting llm prompt template:')
    message_sample_str = llm_prompt.format_prompt(context="Sample Context", question="Who are you?", date="sample-date").text
    print(message_sample_str)
