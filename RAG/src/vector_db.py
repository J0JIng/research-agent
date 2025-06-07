from abc import ABC, abstractmethod
from dotenv import load_dotenv
from typing import Optional

from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

class BaseVectorDB(ABC):

    @abstractmethod
    def create(self) -> None:
        pass
    
    @abstractmethod
    def store(self) -> None:
        pass
    
    @abstractmethod
    def update(self) -> None:
        pass

    @abstractmethod
    def delete(self) -> None:
        pass
    
    @abstractmethod
    def query(self) -> list[str]:
        pass

class ChromaVectorDB(BaseVectorDB):
    def __init__(self, embedding_function: Embeddings, collection_name: str, filepath: str):
        self.embedding_function = embedding_function  
        self.collection_name = collection_name
        self.filepath = filepath
        
        self.client = PersistentClient(path=self.filepath)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.create()

    def create(self) -> None:
        try: 
            self.vector_store = Chroma(
                client = self.client,
                collection_name = self.collection_name,
                embedding_function = self.embedding_function,
            )
        except Exception as e:
            print(f"Exception in chroma vector store: {e}")

    def store(self, uuids: list[str], documents: list[Document]) -> None:
        try:
            self.vector_store.add_documents(
                ids = uuids,
                documents = documents,
            )
        except Exception as e:
            print(f"Exception in storing documents: {e}")
    

    def update(self, uuids: list[str], documents: list[Document]) -> None:
        try: 
            self.vector_store.update_documents(
                ids = uuids, 
                documents = documents,
            )
        except Exception as e:
            print(f"Exception in updating documents: {e}")

    def delete(self, uuids: list[str]) -> None:
        try: 
            self.collection.delete(
                ids = uuids
            )
        except Exception as e:
            print(f"Exception in deleting documents: {e}")

    def query(self, query_text: str, n_results: int = 4, filter: Optional[dict[str, str]] = None) -> list[tuple[Document, float]] | None:
        try: 
            result = self.vector_store.similarity_search_with_score(
                query = query_text,
                k = n_results,
                filter = filter,
            )
            return result
        except Exception as e:
            print(f"Exception in querying documents: {e}")

if __name__ == "__main__":
    embedding = OpenAIEmbeddings()
    db = ChromaVectorDB(embedding, "research_docs", "./chromadb")