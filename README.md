# Research Agent

## Overview

The **Research Agent** project aims to create a modular, agent-based workflow for complex information retrieval and reasoning tasks. By composing multiple specialized agents into a graph-like structure (an agentic workflow), the system can handle multi-step research queries involving retrieval, knowledge graph reasoning, summarization, and verification.

This architecture leverages the strengths of different retrieval modalities (e.g., vector search, knowledge graphs) and natural language generation models (LLMs) to deliver accurate, explainable, and grounded responses.

## Key Features

- **Agentic Workflow / Graph**: Each task (retrieval, summarization, entity linking, fact-checking) is encapsulated in an independent agent node.
- **Multi-Modal Retrieval**: Supports both Retrieval-Augmented Generation (RAG) over unstructured documents and Knowledge Graph-based RAG (KG-RAG).
- **Composable & Extensible**: Easily add or swap agents to customize research pipelines.
- **Explainability**: Tracks sources and citations for transparency.
- **Use of LangGraph (or similar frameworks)**: Enables flexible orchestration of agents into directed workflows.

## Project Structure

`research-agent/
├── agent/ # Agent implementations (research, retrieval, summarization, etc.)
├── tool_RAG/ # Tools for RAG (vector retrieval, embedding, document processing)
├── tool_knowledge_graph_RAG/ # Tools for KG-based retrieval and reasoning
├── main.py # Entry point to run or test the agent workflows
├── requirements.txt # Python dependencies
├── README.md # This file`


## How It Works

- User input is received by a Research Agent which breaks down the query into subtasks.
- Subtasks are delegated to specialized agents:
- Document retrieval (vector search)
- Knowledge graph retrieval and reasoning
- Summarization and generation
- Citation and verification
- Agents communicate through a graph-based flow (e.g., LangGraph) to pass information and aggregate results.
- Final output is a well-formed, grounded research answer with citations.