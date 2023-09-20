from google.cloud import storage
import logging 
from fastapi import HTTPException
import tempfile
import os 

class GCSBucket:
    def __init__(self, 
                 PROJECT_ID, 
                 BUCKET_NAME):
        self.project_id = PROJECT_ID
        self.bucket_name = BUCKET_NAME
        
        self.client = storage.Client(project=PROJECT_ID)
        self.bucket = self.client.bucket(BUCKET_NAME)

    def upload_file(self, file, filename, content_type, metadata={}):
        try:
            blob = self.bucket.blob(filename)
            blob.metadata = metadata
            blob.upload_from_file(file, content_type=content_type)
            logging.info(f"Uploaded {filename} to {self.bucket.name}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def get_filepath(self, filename):
        try:
            blob = self.bucket.blob(filename)
            print(blob)
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.abspath(os.path.expanduser(os.path.join(temp_dir, filename)))
                print(file_path)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                # Download the file to a destination
                blob.download_to_filename(file_path)
                return file_path
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    def download_file_bytes(self, filename):
        blob = self.bucket.blob(filename)
        content_bytes = blob.download_as_bytes()
        return content_bytes
    
    def check_file_Exists(self, filename):
        try:
            blob = self.bucket.blob(filename)
            return blob.exists()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    def check_file_Exists_prefix(self, prefix):
        try:
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            
            if len(blobs) > 0:
                return True, blobs[0].name # Return first filename
            else:
                return False, ''
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def get_gcs_file_metadata(self, filename):
        try:
            blob = self.bucket.blob(filename)
            blob.patch()
            return blob.metadata
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))