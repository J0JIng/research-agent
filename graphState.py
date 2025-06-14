from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from typing_extensions import TypedDict


class GraphState(TypedDict):
    
    message: str
    next_agent : str 