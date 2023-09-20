
from api.v1_0.usecases.base.models import QABotBaseQuery
from pydantic import Field, validator

class PsoBotQuery(QABotBaseQuery):
    language: str = Field("en", description="Language of the query (en/fr)")
    temperature: float = 0.4
    
    class Config:
        schema_extra = {
            "example": {
                "chat_history": [
                    {"role": "Human", "content": "What is PSO?"},
                    {"role": "Assistant", "content": "Premier Sales Organization."}
                ],
                "query": "how?",
                "language": "en",
                "stream": False,
                "return_docs": False
            }
        }

    @validator('language')
    def validate_language(cls, language):
        if language not in ['en', 'fr']:
            raise ValueError("The language can either be 'en' or 'fr'")
        return language


