from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)

print("1. Starting loader...", flush=True)

loader = DirectoryLoader(
    path="./build",
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader,
    loader_kwargs={"strategy": "fast"},
    show_progress=True,
)

print("2. Loader initialized. Loading PDF...", flush=True)

docs = loader.load()

print("3. PDF loading completed!", flush=True)
print(f"Total documents: {len(docs)}", flush=True)