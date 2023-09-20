
from api.v1_0.usecases.base.models import QABotBaseQuery
from pydantic import Field, validator

class StandardsBotQuery(QABotBaseQuery):
    class Config:
        schema_extra = {
            "example": {
                "chat_history": [
                    {"role": "Human", "content": "What is the allowance given under tariff 406?"},
                    {"role": "Assistant", "content": "Telus can give $2500 for each qualified principal premises."}
                ],
                "query": "how?",
                "stream": False,
                "return_docs": False
            }
        }


