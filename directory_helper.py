import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DefaultChunker:
    def __init__(self, chunk_size=512, step_size=256):
        self.chunk_size = chunk_size
        self.step_size = step_size

    def get_chunks(self, data):
        for text in data:
            for i in range(0, len(text), self.step_size):
                max_size = min(self.chunk_size, len(text) - i)
                yield text[i:i+max_size]

class LangChainChunker:
    def __init__(self, chunk_size=512, chunk_overlap=24):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def get_chunks(self, data):
        for text in data:
            yield from self.text_splitter.split_text(text)

class DirectoryLoader:
    def __init__(self, directory, batch_size=512, chunker=LangChainChunker()):
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