from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser

class BaseResponse(BaseModel):
    "basic agent response"
    content: str
    sources: list[str]
    tools_used: list[str]

class RAGResearchResponse(BaseModel):
    "RAG Research agent response"
    topic: str
    content : str
    similarity_score : str
    sources: list[str]
    tools_used: list[str]

class KGResearchResponse(BaseModel):
    "KG Research agent response"
    topic: str
    content : str
    relationship_score :str
    sources: list[str]
    tools_used: list[str]


basic_parser = PydanticOutputParser(pydantic_object=BaseResponse)
RAG_parser = PydanticOutputParser(pydantic_object=RAGResearchResponse)
KG_parser = PydanticOutputParser(pydantic_object=KGResearchResponse)