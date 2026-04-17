import json
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load products
with open("./scripts/data/products.json") as f:
  products = json.load(f)

# Embedding model
embeddings = HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Convert to documents
docs = []
for p in products:
  content = f"""
  Name: {p['name']}
  Category: {p['category']}
  Price: {p['price']}
  Description: {p['description']}
  """
  docs.append(
    Document(
      page_content=content, 
      metadata={
        "id": p["id"],
        "price": p["price"],
        "category": p["category"]
      }
    )
  )

# Create DB
db = Chroma.from_documents(
  docs,
  embeddings,
  persist_directory="./db"
)

db.persist()

print("✅ Ingest complete")