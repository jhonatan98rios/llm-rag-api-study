from langchain_community.vectorstores import Chroma
from .embeddings import embeddings

db = Chroma(
  persist_directory="./db",
  embedding_function=embeddings
)