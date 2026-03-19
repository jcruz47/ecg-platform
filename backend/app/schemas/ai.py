from pydantic import BaseModel


class AskAIRequest(BaseModel):
    question: str