from abc import ABC, abstractmethod
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def embed_documents(self, documents: list[str]): 
        pass    
    
    @abstractmethod
    def embed_query(self, query: str): 
        pass

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = OpenAIEmbeddings(model=model)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return self.model.embed_documents(documents)

    def embed_query(self, query: str) -> list[float]:
        return self.model.embed_query(query)

class HuggingfaceEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model: str = "minishlab/potion-multilingual-128M"):
        self.model = HuggingFaceEmbeddings(model=model)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return self.model.embed_documents(documents)

    def embed_query(self, query: str) -> list[float]:
        return self.model.embed_query(query)

if __name__ == "__main__":
    # Initialize the embedding model
    load_dotenv()
    openAImodel = OpenAIEmbeddingModel()
    embedding = openAImodel.embed_query("Apple")
    print(f"open ai embedding model dimension: {len(embedding)}")
    
    huggingfacemodel = HuggingfaceEmbeddingModel()
    embedding = huggingfacemodel.embed_query("Apple")
    print(f"huggingface embedding model dimension: {len(embedding)}")
    