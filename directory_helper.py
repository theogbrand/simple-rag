import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings


class SlidingWindowCharacterChunker:
    def __init__(
        self, chunk_size=512, step_size=256
    ):  # chunk_size is character count, step_size is sliding window step before next chunk is created again; sliding window useful for ensuring info not lost at boundaries of chunks
        self.chunk_size = chunk_size
        self.step_size = step_size

    def get_chunks(self, data):
        for text in data:
            for i in range(0, len(text), self.step_size):
                max_size = min(self.chunk_size, len(text) - i)
                yield text[i : i + max_size]


class RecursiveCharacterChunker:  # human to semantic split using \n\n, then use this
    def __init__(self, chunk_size=1024, chunk_overlap=24):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def get_chunks(self, data):
        for text in data:
            yield from self.text_splitter.split_text(text)


class LCSemanticChunker:
    def __init__(self, breakpoint_threshold_type="percentile"):
        self.text_splitter = SemanticChunker(
            OpenAIEmbeddings(), breakpoint_threshold_type=breakpoint_threshold_type
        )

    def get_chunks(self, data):
        for text in data:
            for doc in self.text_splitter.create_documents([text]):
                yield doc.page_content


class DirectoryLoader:
    def __init__(self, directory, batch_size=512, chunker=RecursiveCharacterChunker()):
        self.directory = directory
        self.chunker = chunker
        self.batch_size = batch_size

    def load(self):
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    print(f"Loading file: {file_path}")
                    content = f.read()
                    chunks = list(self.chunker.get_chunks([content]))
                    print(f"Number of chunks for {file_path}: {len(chunks)}")
                    yield content

    def get_chunks(self):
        return self.chunker.get_chunks(self.load())

    def get_chunk_batches(self):
        chunks = []
        for chunk in self.get_chunks():
            chunks.append(chunk)
            if len(chunks) == self.batch_size:
                yield chunks
                chunks = []

        if len(chunks) > 0:
            yield chunks

    def __iter__(self):
        return self.get_chunk_batches()
