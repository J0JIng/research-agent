# Agent

This folder contains the implementation of various agents used in the research pipeline.

## Overview

The core idea is to modularize different functionalities into individual agent classes. These agents can be composed together (e.g., using LangGraph) to form a larger research agent capable of complex tasks like retrieval, summarization, verification, and report generation.

## Structure

- `research_agent.py`: Orchestrates the research workflow by delegating tasks to sub-agents.
- `search_agent.py`: Handles information retrieval from external or internal sources.
- `summarizer_agent.py`: Summarizes long documents or retrieved content.
- `citation_agent.py`: Tracks and formats sources for attribution.
- `critique_agent.py`: Verifies and critiques the factual accuracy of responses.
- `report_generator.py`: Combines outputs into a final structured response.

## Usage

Each file defines an agent class with a clear purpose. These agents are designed to be used as nodes in a LangGraph flow or as standalone components.

You can import and use them like:

```python
from agent.research_agent import ResearchAgent

agent = ResearchAgent()
result = agent.run("What are the latest findings on LLM efficiency?")
