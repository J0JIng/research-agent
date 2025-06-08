from dotenv import load_dotenv
from src.vector_db import ChromaVectorDB

from langchain_openai import OpenAIEmbeddings
from langchain.tools import Tool

def query_rag_tool(chromaDB:ChromaVectorDB, query_text: str) :
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

query_tool = Tool(
    name="query_rag_tool",
    func=query_rag_tool,
    description="query local vector database for information.",
)


if __name__ == "__main__":
    load_dotenv()

    # store
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    chromaDB = ChromaVectorDB(embedding_model, "research_docs", "./chromadb")

    # query 
    result = chromaDB.query(query_text="collaborative filtering", n_results=10)
    for doc, score in result:
        print("======"*10)
        print('\n')
        print(f"Similarity score {score:3f}")
        print('\n')
        print(doc.metadata)
        print('\n')
        print(doc.page_content)
        print("======"*10) 


