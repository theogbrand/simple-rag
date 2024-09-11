TODO:
* add 4-shot, 1K shot import for few shot examples
* Advanced RAG:
    * Query Re-Writing
    * Multistep iterative anwering (decomopse complex question, retrieve until LLM thinks enough context)
* add web search
* refactor for Structured Outputs mode (OpenAI + Claude style)
    * add metadata to document chunks
* add Guardrails based on https://cookbook.openai.com/examples/how_to_combine_gpt4o_with_rag_outfit_assistant
* add BM25 (perhaps not since Cohere Reranker is best)
* add hierarchical multi-agent system: https://cookbook.openai.com/examples/structured_outputs_multi_agent

* PDFtoChat (https://github.com/Nutlope/pdftochat)
* RAGFlow (https://github.com/infiniflow/ragflow)