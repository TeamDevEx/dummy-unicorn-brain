from langchain.docstore.document import Document
from langchain.document_loaders import DataFrameLoader
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

response_schema = [
    ResponseSchema(name="unanswered_query_summary", description="")
    ]

output_parser = StructuredOutputParser.from_response_schemas(response_schema)
format_instructions = output_parser.get_format_instructions()

prompt = ChatPromptTemplate(
    messages = [
        SystemMessagePromptTemplate.from_template(
            """
            You are a bot that summarizes a list of telecommunications query topics, by following the below steps. \
            
            1) Get a list of the number of queries.
            2) Assign one topic to each query (examples: technical support, billing questions, accounts, pricing and deals)
            3) Group all queries by topic, order the topics by query count, and select the topics with the largest query count (up to 20), all remaining topics will be renamed "Miscellanious"                                          

            You will be provided with a list of queries below in the following format:
            Query[1]: "Query 1 value..." 
            Query[2]: "Query 2 value..."
            The number within Query[] is the index.
            
            =========
            {context}
            =========
            
            Return how many queries there are and a numbered list of the topics. For each of the topics, please include the number of queries that fall into the topic \
                and also provide a sub-list with up to 3 examples of query values that are part of each topic. Topics renamed to "Miscellaneous" should not have any examples
            
            Always return your response in the following format. \n{format_instructions}
            
            """
            
        )
    ],
    input_variables=["context"],
    partial_variables={"format_instructions": format_instructions}
)