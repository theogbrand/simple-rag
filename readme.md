TODO:
* add Exa search
* add 4-shot, 1K shot import for few shot examples (put human vetted FAQ in here)
* Advanced RAG:
    * Query Re-Writing
    * Multistep iterative anwering (decomopse complex question, retrieve until LLM thinks enough context)
    * Chunk Expansion (extracts chunk above and below selected top chunks) - if too many docs to human-semantic-split
* refactor for Structured Outputs mode (OpenAI + Claude style)
    * add metadata to document chunks (JSON chunks with document filename attached for querying across docs)
* add Guardrails based on https://cookbook.openai.com/examples/how_to_combine_gpt4o_with_rag_outfit_assistant
* add BM25 (perhaps not since Cohere Reranker is best)
* add hierarchical multi-agent system: https://cookbook.openai.com/examples/structured_outputs_multi_agent

* PDFtoChat (https://github.com/Nutlope/pdftochat)
* RAGFlow (https://github.com/infiniflow/ragflow)