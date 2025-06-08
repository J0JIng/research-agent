# RAG Tools

This folder contains core tools for building a Retrieval-Augmented Generation (RAG) pipeline.

## Overview

The modules here support processing documents, embedding text, and performing vector-based retrieval — the essential components for RAG systems.

## Structure

- `data_processing.py`: Functions to load, clean, and chunk raw documents into manageable pieces for retrieval.
- `embedder.py`: Converts text chunks into vector embeddings using pretrained models.
- `vector_db.py`: Implements vector database operations such as indexing and searching embeddings.
