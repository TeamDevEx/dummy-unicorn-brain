from typing import Union, Dict, Optional
from pydantic import BaseModel, validator
from fastapi import UploadFile, File, Form, HTTPException

accepted_file_types = ['application/pdf', 
                       'text/plain', 
                       'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                       ]

# Doesnt work as File is based in as Form Data -> Not JSON
# https://stackoverflow.com/questions/65504438/how-to-add-both-file-and-json-body-in-a-fastapi-post-request/70640522#70640522
class UploadFileAPIRequest(BaseModel):
    file: UploadFile = File(...)
    collection_name: Optional[str] = Form(..., alias='self_serve') 
    user: str = Form(...)
    email: str = Form(...)

    @validator('file')
    def check_file(cls, file):
        if file.content_type not in accepted_file_types:
            raise HTTPException(status_code=400, detail=f"File type not supported: Only .pdf, .txt, .docx supported. ")

        elif file.size > 20 * 1000 * 1000:
            raise HTTPException(status_code=400, detail="File size too large: Max 20MB")
    
        return file