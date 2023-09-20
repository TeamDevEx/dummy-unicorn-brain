# add src for local imports
import os, sys
absolute_path = os.path.dirname(__file__)
relative_path = "../../../../"
full_path = os.path.realpath(os.path.join(absolute_path, relative_path))
sys.path.append(full_path)

from pydantic import BaseModel, validator, Field
import re

class LogsSummaryBotQuery(BaseModel):
    bq_table: str
    date: str

    @validator('bq_table')
    def validate_bq_str(cls, bq_table):
        if len(bq_table.split('.')) != 3:
            raise ValueError("BQ table path should be in the following format: project_id.dataset.table")
        elif not bq_table.startswith('cdo-gen-ai-island-np-204b23.logs.'):
            raise ValueError("This API is only setup to work with log files. Your BQ path should start with: cdo-gen-ai-island-np-204b23.logs.")
            
        return bq_table
    
    @validator('date')
    def validate_date_str(cls, date):
        pattern = re.compile(r"^\d{4}\-(0[1-9]|1[012])\-(0[1-9]|[12][0-9]|3[01])$")
        if not pattern.match(date):
            raise ValueError("The date is not in the correct format yyyy-mm-dd")
       
        return date


