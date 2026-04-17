from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Same embedding model
embeddings = HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load DB
db = Chroma(
  persist_directory="./db",
  embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 3})
query = "gpu for local AI"
docs = retriever.invoke(query)

print("\n=== RESULTS ===\n")

for d in docs:
  print(d.page_content)
  print("----")