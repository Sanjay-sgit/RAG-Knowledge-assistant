RAG Mastery: 5-Day Journey into Retrieval Augmented Generation

A hands-on curriculum covering everything from RAG fundamentals to advanced, production-grade retrieval pipelines — no black boxes, built step by step with and without LangChain.

📚 Overview

This repository contains code, notebooks, and resources for a 5-day deep dive into Retrieval Augmented Generation (RAG). Each day builds on the last, moving from core concepts to a fully evaluated, advanced RAG system.

🗓️ Curriculum

1 — RAG Fundamentals


Introduction to RAG: Retrieval Augmented Generation Fundamentals
Building a Simple RAG Knowledge Assistant with GPT-4.1 Nano
Building a Simple RAG System: Dictionary Lookup and Context Retrieval
Vector Embeddings and Encoder LLMs: The Foundation of RAG
How Vector Embeddings Represent Meaning: From word2vec to Encoders
Understanding the Big Idea Behind RAG and Vector Data Stores


2 - Vector_Databases_and_Embeddings


Vectors for RAG: Introduction to LangChain and Vector Databases
Breaking Documents into Chunks with LangChain Text Splitters
Encoder Models vs Vector Databases: OpenAI, BERT, Chroma & FAISS
Creating Vector Stores with Chroma and Visualizing Embeddings with t-SNE
3D Vector Visualizations and Comparing Embedding Models


3 - End_to_End_RAG_System (RAG pipeline).ipynb


Building a Complete RAG Pipeline with LangChain and Chroma
Building a RAG Pipeline with LangChain: LLM & Retriever Setup
Building RAG with LangChain: Retriever and LLM Integration
Building Production RAG with Python Modules and Gradio UI
RAG with Conversation History: Building a Gradio UI and Debugging Chunking


4 — RAG Evaluation


RAG Evaluations: Measuring Performance and Iterating on Your Pipeline
Evaluating RAG Systems: Retrieval Metrics, LLM as Judge, and Golden Data
Evaluating RAG Systems: MRR, NDCG, and Test Data with Pydantic
LLM as a Judge: Evaluating RAG Answers with Structured Outputs
Running RAG Evaluations with Gradio: MRR, nDCG, and Test Results
Experimenting with Chunking Strategies and Embedding Models in RAG
Testing OpenAI Embeddings and Evaluating RAG Performance Gains


5 — Advanced RAG Techniques


Advanced RAG Techniques: Pre-processing, Re-ranking & Evals
Advanced RAG Techniques: Chunking, Encoders, and Query Rewriting
Advanced RAG Techniques: Query Expansion, Re-ranking & GraphRAG
Building Advanced RAG Without LangChain: Semantic Chunking with LLMs
Creating Embeddings with Chroma, Visualizing with t-SNE, and Re-ranking
Building RAG Without LangChain: Re-ranking and Query Rewriting
Building Production RAG with Query Expansion and Multiprocessing
Advanced RAG Evaluation: From 0.73 to 0.91 MRR with GPT-4o
Challenge: Beat My Results & Build Your Own Knowledge Worker


Key Concepts Covered:


Vector embeddings & encoder LLMs (word2vec, BERT, OpenAI embeddings)
Chunking strategies (fixed-size, semantic, LLM-based)
Vector databases: Chroma, FAISS
LangChain retrievers and pipeline orchestration
Conversational RAG with memory
RAG evaluation: MRR, nDCG, LLM-as-judge, golden datasets
Advanced techniques: query rewriting, query expansion, re-ranking, GraphRAG
Building RAG systems from scratch (without LangChain)
Production concerns: Gradio UIs, multiprocessing, structured outputs (Pydantic)


Tech Stack:


LLMs: GPT-4.1 Nano, GPT-4o, or openrouter free.
Frameworks: LangChain
Vector Stores: Chroma, FAISS
Embeddings: OpenAI embeddings, BERT-based encoders
Evaluation: Pydantic, custom MRR/nDCG scoring, LLM-as-judge
UI: Gradio
Visualization: t-SNE (2D/3D embedding plots)


Getting Started : 

1. Clone the repository
git clone YOUR_REPO_URL
cd YOUR_REPO_NAME
2. Install dependencies

Using uv:

uv sync

Using Poetry:

poetry install

Using pip:

pip install -e .
3. Configure environment variables

Set up your API keys (e.g., OPENAI_API_KEY) in a .env file
for free llm calling use openrouter api client.

4. Run the notebooks

Work through the notebooks day by day, from 01_RAG_Fundamentals.ipynb through 05_Production_Ready_Advanced_RAG.ipynb.

5. Run the application
python app.py
