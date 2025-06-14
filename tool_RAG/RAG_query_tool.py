from tool_RAG.src.vector_db import ChromaVectorDB

from langchain_openai import OpenAIEmbeddings
from langchain.tools import Tool

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
chromaDB = ChromaVectorDB(embedding_model, "research_docs", "./chromadb")

def query_rag_tool(query_text: str) -> str:
    information = ""
    result = chromaDB.query(query_text=query_text)
    for doc, score in result:
        information += '\n'
        information += f"Similarity score {score:3f}"
        information += '\n'
        information += str(doc.metadata)
        information += '\n'
        information += str(doc.page_content)
    return information

rag_query_tool = Tool(
    name="query_rag_tool",
    func=query_rag_tool,
    description="Query the local vector database with a natural language question."
)


