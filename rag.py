import faiss
import time
import numpy as np
from tqdm import tqdm
import litellm
from litellm import completion
import dotenv
from directory_helper import DirectoryLoader

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
        embedding_list = [embedding["embedding"] for embedding in embedding_response.data]
        # all text are converted into their Vector form as numpy arrays, NO KNN here
        return np.array(embedding_list)

    def query(self, query, k=5):
        embedding = self.get_embeddings([query])[0]
        embedding_array = np.array([embedding])
        _, indices = self.index.search(embedding_array, k)
        return [self.content_chunks[i] for i in indices[0]]


class QueryEngine:
    def __init__(self, index, k=5):
        self.index = index
        self.k = k

    def answer_question(self, question):
        most_similar = self.index.query(question, k=self.k)
        prompt = "\n".join(reversed(most_similar)) + "\n\n" + question
        print("------------------------------ Prompt ------------------------------\n" + prompt + "\n----------------------------- End Prompt -----------------------------")
        # return self.model.generate("<s>[INST]" + prompt + "[/INST]") # for Instruct models that require templates
        response = completion(
            model="gpt-4o-mini-2024-07-18", messages=[{"content": prompt, "role": "user"}]
        )
        return response.choices[0].message.content


class RetrievalAugmentedRunner:
    def __init__(self, dir, k=5):
        self.k = k
        self.loader = DirectoryLoader(dir)

    def train(self):
        self.index = DIYIndex(self.loader)

    def __call__(self, query):
        query_engine = QueryEngine(self.index, k=self.k)
        return query_engine.answer_question(query)


def main():
    model = RetrievalAugmentedRunner(dir="data")
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
