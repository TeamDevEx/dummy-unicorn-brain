# /milvus/upload_to_db Endpoint Documentation

This endpoint uploads files to Milvus into a self_serve collection. Currently works for .pdf, .txt, .docx filetypes. Once a file is sent, it's uploaded to a GCS Bucket before being loaded via Langchain's GCSLoader and embedded into Milvus. This is currently deployed in np

## Endpoint Request

1. file: UploadFile = File(...)
2. collection_name: Optional[str] = Form('self_serve')
3. user: str = Form(...)
4. email: str = Form(...)

### Example of sending a file in Python using Requests library:

```
import requests
import mimetypes
import os

form_data = {"user": "USERNAME", "email" : "EMAIL"}

file_path = "<PATH_TO_FILE/filename.docx>
file_data = (os.path.basename(file_path), open(file_path, "rb"), mimetypes.guess_type(file_path)[0])

url = "https://ai-np.cloudapps.telus.com/gen-ai-api/1.0/milvus/upload_to_db"

response = requests.post(url, files={"file": file_data}, data=form_data)

```

## Querying from Milvus

Once you have successfully uploaded the file to Milvus, you can search through the collection via the search endpoint or ask questions to it via the generic endpoint. The key is to search within the specific collection (self_serve) and apply the filters to search within the file that was just uploaded using the `content_filter` field.

### Example body using the search endpoint:

```
{
    "collection_name": "self_serve",
    "query": "question",
    "content_filter": "(metadata_user== \"USERNAME\") && (metadata_email==\"EMAIL\") && (metadata_filename == \"filename.docx\")",
    "max_num_results": 4,
    "max_distance":  1.0
}
```

### Example body using the generic endpoint:

```
{

  "chat_history": [],
  "query": "question about this document",
  "stream": false,
  "collection_name": "self_serve",
  "content_filter" : "(metadata_user== \"USERNAME\") && (metadata_email==\"EMAIL\") && (metadata_filename == \"filename.docx\")",
  "maximum_context_token_count" : 2500,
  "maximum_doc_count" : 5,
  "return_docs": false,
  "temperature": 0.4

}
```

### To get the files loaded into the `self_serve` collection in Milvus:

Make an empty post request to `https://ai-np.cloudapps.telus.com/gen-ai-api/1.0/milvus/selfserve_get_loaded_files`
