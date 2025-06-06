from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# load_dotenv()

# # Initialize the embedding model
# embeddings_model = OpenAIEmbeddings()

# # Embed a list of strings
# texts = ["This is the first document.", "This is the second document."]
# embeddings = embeddings_model.embed_documents(texts)
# print(f"Number of embeddings: {len(embeddings)}")
# print(f"Length of the first embedding: {len(embeddings[0])}")

# # Embed a single string
# query = "What is the meaning of life?"
# query_embedding = embeddings_model.embed_query(query)
# print(f"Length of the query embedding: {len(query_embedding)}")
# print(f"embedding: {query_embedding}")

class Embedder:
    def __init__(self, model):
        self.model = model 

    def embed_documents(self):
        pass
        
