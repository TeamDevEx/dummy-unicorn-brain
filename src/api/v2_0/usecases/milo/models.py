
from api.v2_0.llm.ask.models import QABotBaseQuery
from pydantic import Field, validator
from dateutil import parser
from typing import Union

class MiloBotQuery(QABotBaseQuery):
    language: str = Field("en", description="Language of the query (en/fr)") # do we need language for MILO??
    temperature: float = 0.4
    brands: list[str] #options: telus, koodo, public
    provinces: list[str] #options: BC, AB, ON, QC, NL, PE, YU, NT, NU, SK, MB, NS, NB
    status: str # this is freshness of the content, options: current, expired
    publish_date: Union[str, None] = None # date when content is published, only search content greater than this date

    # prospective filter
    # @validator('language')
    # def validate_language(cls, language):
    #     if language not in ['en', 'fr']:
    #         raise ValueError("The language can either be 'en' or 'fr'")
    #     return language
    
    @validator('brands')
    def validate_brands(cls, brands):
        for brand in brands:
            if brand not in ['TELUS', 'Koodo', 'Public']:
                raise ValueError("The brand can either be 'TELUS', 'Koodo', or 'Public'")
        return brands
    
    @validator('provinces')
    def validate_province(cls, provinces):
        for province in provinces:
            if province not in ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'PE', 'QC', 'SK', 'YU']:
                raise ValueError("The province can only be a valid Canadian province")
        return provinces
    
    @validator('status')
    def validate_status(cls, status): 
        if status not in ['Current', 'Expired']: # need to update based on list of content status
            raise ValueError("The status can either be 'Current' or 'Expired'")
        return status
    
    # prospective filter
    # @validator('business_type')
    # def validate_business_type(cls, business_type):
    #     if business_type not in ['consumer', 'business', 'smb']: # need to update based on complete list of business types
    #         raise ValueError("The business type can either be 'consumer', 'business', or 'smb'")
    #     return business_type
    
    @validator('publish_date')
    def validate_publish_date(cls, publish_date):
        if not bool(parser.parse(publish_date)):
            raise ValueError("The publish date must be a valid date of the format %Y-%m-%d")            
        return publish_date


