from pathlib import Path
from dotenv import load_dotenv
from uuid import uuid4

from src.vector_db import ChromaVectorDB
from src.embedder import OpenAIEmbeddingModel
from src.data_processing import PDFPreprocessor

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

if __name__ == "__main__":
    load_dotenv()
    # TODO
    # connect to db
    # retrieve all the documents stored and uuid 
    # start the document - chunking only if there is new documents 
    
    raw_dir = Path("./data/pdf/raw")
    cleaned_dir = Path("./data/pdf/cleaned")
    cleaned_dir.mkdir(exist_ok=True)
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # process docs 
    preprocessor = PDFPreprocessor(raw_dir, cleaned_dir)
    documents = preprocessor.run()

    # generate uuids
    uuids = [str(uuid4()) for _ in range(len(documents))]
    print(uuids)
    
    # store
    chromaDB = ChromaVectorDB(embedding_model, "research_docs", "./chromadb")
    chromaDB.store(uuids=uuids, documents=documents)

    # query 
    result = chromaDB.query(query_text="collaborative filtering")
    for doc, score in result:
        print("======"*10)
        print('\n')
        print(f"Similarity score {score:3f}")
        print('\n')
        print(doc.metadata)
        print('\n')
        print(doc.page_content)
        print("======"*10)
