import faiss
import time
import numpy as np
from tqdm import tqdm
import litellm
from litellm import completion
import dotenv
from directory_helper import DirectoryLoader
import cohere
from typing import List, Dict, Any
import json

dotenv.load_dotenv()
litellm.return_response_headers = True


class DIYIndex:
    def __init__(self, loader):
        self.loader = loader
        self.build_index()

    def build_index(self):
        self.content_chunks = []
        self.index = None
        for chunk_batch in tqdm(self.loader):
            embeddings = self.get_embeddings(chunk_batch)
            if self.index is None:
                self.index = faiss.IndexFlatL2(len(embeddings[0]))
            self.index.add(embeddings)
            self.content_chunks.extend(chunk_batch)

    def get_embeddings(self, examples):
        # ebd = Embedding()
        # embeddings = ebd.generate(examples)
        embedding_response = litellm.embedding(
            model="text-embedding-3-small",
            input=examples,
        )
        embedding_list = [
            embedding["embedding"] for embedding in embedding_response.data
        ]
        # all text are converted into their Vector form as numpy arrays, NO KNN here
        return np.array(embedding_list)

    def query(self, query, k=5):
        embedding = self.get_embeddings([query])[0]
        embedding_array = np.array([embedding])
        _, indices = self.index.search(embedding_array, k)
        return [self.content_chunks[i] for i in indices[0]]


class QueryEngine:
    def __init__(self, index, k=5, rerank_k=2, use_rerank=True):
        self.index = index
        self.k = k
        self.rerank_k = rerank_k
        self.use_rerank = use_rerank
        self.cohere_client = cohere.Client() if use_rerank else None

    def answer_question(self, question):
        if self.use_rerank:
            # Retrieve more documents initially
            most_similar = self.index.query(question, k=self.k)
            most_similar = most_similar[
                ::-1
            ]  # index.query uses L2 distance, so most similar is at the end
            print("Before Rerank")
            print(
                "------------------------------ Most Similar ------------------------------"
            )
            print("\n".join(most_similar))
            print(
                "----------------------------- End Most Similar -----------------------------"
            )
            # Rerank the retrieved documents
            reranked_results = self.rerank_documents(question, most_similar)
            print(
                "------------------------------ Reranked Results ------------------------------\n"
                + "\n".join(reranked_results)
                + "\n----------------------------- End Reranked Results -----------------------------"
            )
            # Select top k reranked documents
            top_k_documents = reranked_results[: self.k]
        else:
            # If not using rerank, just retrieve k documents
            top_k_documents = self.index.query(question, k=self.k)
            print(
                "------------------------------ Top K Documents ------------------------------\n"
                + "\n".join(top_k_documents)
                + "\n----------------------------- End Top K Documents -----------------------------"
            )
        prompt = "\n".join(top_k_documents) + "\n\n" + "Q: " + question + "\nA:"
        print(
            "------------------------------ Prompt ------------------------------\n"
            + prompt
            + "\n----------------------------- End Prompt -----------------------------"
        )

        response = completion(
            model="gpt-4o-mini-2024-07-18",
            messages=[{"content": prompt, "role": "user"}],
        )
        return response.choices[0].message.content

    # return self.model.generate("<s>[INST]" + prompt + "[/INST]") # for Instruct models that require templates

    def rerank_documents(self, query: str, documents: List[str]) -> List[str]:
        """
        Rerank documents using Cohere's Rerank API.

        Args:
            query (str): The query to rerank documents against.
            documents (List[str]): The list of documents to be reranked.

        Returns:
            List[str]: The reranked list of documents.
        """
        if not self.use_rerank:
            return documents

        reranked_response = self.cohere_client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=documents,
            top_n=self.rerank_k,
        )
        # print(reranked_response.results[0].index)

        # Extract the reranked indices from the response
        reranked_indices = [result.index for result in reranked_response.results]

        # Reorder the original documents based on the reranked indices
        reranked_documents = [documents[i] for i in reranked_indices]

        return reranked_documents


class RetrievalAugmentedRunner:
    def __init__(self, dir, k=5, use_rerank=True):
        self.k = k
        self.use_rerank = use_rerank
        self.loader = DirectoryLoader(dir)

    def train(self):
        self.index = DIYIndex(self.loader)

    def __call__(self, query):
        query_engine = QueryEngine(self.index, k=self.k, use_rerank=self.use_rerank)
        return query_engine.answer_question(query)


def main():
    use_rerank = True  # Set this to False to disable reranking
    model = RetrievalAugmentedRunner(dir="data", use_rerank=use_rerank)
    start = time.time()
    model.train()
    print("Time taken to build index: ", time.time() - start)
    while True:
        prompt = input(
            "\n\nEnter another investment question (e.g. Have we invested in any generative AI companies in 2023?): "
        )
        start = time.time()
        print(model(prompt))
        print("\nTime taken: ", time.time() - start)


main()
