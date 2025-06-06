import re
import fitz
from pathlib import Path

from langdetect import detect
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class PDFPreprocessor:
    def __init__(self, raw_dir: Path, cleaned_dir: Path):
        self.raw_dir = raw_dir
        self.cleaned_dir = cleaned_dir
        self.cleaned_dir.mkdir(exist_ok=True)

    def pre_process_pdf_documents(self) -> None:
        for pdf_file in self.raw_dir.glob("*.pdf"):
            doc = fitz.open(pdf_file)
            for i, page in enumerate(doc, start=1):
                text = page.get_text()
                if f"{i}" not in text.strip()[-10:]:
                    page.insert_text(
                        point=fitz.Point(50, page.rect.height - 30),
                        text=f"Page {i}",
                        fontsize=10,
                        color=(0, 0, 0),
                    )
            doc.save(self.cleaned_dir / pdf_file.name)

    def clean_text(self, text: str) -> str:
        text = text.replace('\n', ' ').replace('\t', ' ')
        return re.sub(r'\s+', ' ', text).strip()

    def generate_documents(self) -> list[Document]:
        documents = []
        for file in self.cleaned_dir.glob("*.pdf"):
            loader = PyPDFLoader(str(file))
            documents.extend(loader.load())
        return documents

    def post_process_documents(self, documents: list[Document]) -> list[Document]:
        for doc in documents:
            doc.page_content = self.clean_text(doc.page_content)
            doc.metadata["source"] = Path(doc.metadata.get("source", "")).stem
            doc.metadata["language"] = detect(doc.page_content)
        return documents

    def post_process_chunk(self, documents: list[Document]) -> list[Document]:
        return [doc for doc in documents if len(doc.page_content.strip()) > 50]

    def generate_chunks(self, documents: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
            strip_whitespace=True
        )
        chunks = splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def run(self) -> list[Document]:
        # Uncomment if preprocessing PDFs (adding page numbers)
        # self.pre_process_pdf_documents()
        docs = self.generate_documents()
        docs = self.post_process_documents(docs)
        chunks = self.generate_chunks(docs)
        docs = self.post_process_chunk(chunks)
        return docs

def main():
    raw_dir = Path("../data/raw")
    cleaned_dir = Path("../data/cleaned")
    cleaned_dir.mkdir(exist_ok=True)

    preprocessor = PDFPreprocessor(raw_dir, cleaned_dir)
    docs = preprocessor.run()
    print(docs[10].metadata)
    print(docs[10].page_content)

if __name__ == "__main__":
    main()