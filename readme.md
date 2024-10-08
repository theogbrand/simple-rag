# TODO:

- add Exa search
- add NLI model step to filter out irrelevant context
    - if context is incomplete, tool to go back to full text to retrieve missing info recursively
- add 4-shot, 1K shot import for few shot examples (put human vetted FAQ in here)
- Advanced RAG:
  - Query Re-Writing
  - Multistep iterative anwering (decomopse complex question, retrieve until LLM thinks enough context)
  - Chunk Expansion (extracts chunk above and below selected top chunks) - if too many docs to human-semantic-split
- refactor for Structured Outputs mode (OpenAI + Claude style)
  - add metadata to document chunks (JSON chunks with document filename attached for querying across docs)
- add Guardrails based on https://cookbook.openai.com/examples/how_to_combine_gpt4o_with_rag_outfit_assistant
- add BM25 (perhaps not since Cohere Reranker is best)
- add hierarchical multi-agent system: https://cookbook.openai.com/examples/structured_outputs_multi_agent
- Self-rag https://github.com/AkariAsai/self-rag?tab=readme-ov-file

- PDFtoChat (https://github.com/Nutlope/pdftochat)
- RAGFlow (https://github.com/infiniflow/ragflow)

# Env Vars required:
OPENAI_API_KEY, COHERE_API_KEY

# Run Instructions:
1. create venv, pip install all requirements, activate env
2. run `python rag.py`