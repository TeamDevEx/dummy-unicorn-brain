# /summarize Endpoint Documentation
This is a general summarization endpoint that can be used to summarize a variety of different information, 
We have different prompts created to allow for users to get the most relevant summary for their data
See the prompt.py file in the summary folder to see which prompt best fits the style of summarization you are looking for

## Summarize Documents via Milvus
The following endpoint will allow users to get a summary of any document, the endpoint currently does not support streaming, meaning until the response has been fully generated nothing will be returned to the user. Additionally, document length will affect the endpoint run time, a rough estimate of around 1 minute for every 10K tokens.

The endpoint requires the user to specify the collection as well as the doc(s) within collection using an expression of the content's pk values to filter the content you would like to summarize. Users can also provide their own temperature value to tune the models creativity when creating a summary, from little to no creativity (0) to highly creative(1)

1. collection_name: str() 
2. content_filter: str() - expression of pk values
3. temperature: float - Value betweeen 0-1

## Uploading content to Milvus self serve

See the following page for how to upload content to Milvus self serve:https://github.com/telus/gen-ai-api/blob/main/docs/uploadfile_to_milvus_endpoint.md

This document will outline how you can load documents into the Milvus self_serve collection using the /milvus/upload_to_db

Then once you have uploaded the collecitons, you can use the /search endpoint to find the pk values of all the different chunks of the file you have uploaded,

Now you have the colleciton_name and content_filter information allowing you to summarize you document

Additionally, if you want to use a QA with this document you have uploaded you can use the /bots/generic endpoint to ask questions regarding the content

### Example body when using the endpoint:
```
{
"collection_name": "self_serve",
"content_filter": "(pk >= 443087289718017683) && (pk <= 443087289718017689)",
"temperature": 0.7,
"prompt_type": "milvus"
}
```