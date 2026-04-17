import json
import time

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage

# LLM (LM Studio)
llm = ChatOpenAI(
  base_url="http://localhost:1234/v1",
  api_key="lm-studio",
  model="qwen3.5-0.8b",
  temperature=0.3,
  max_tokens=150
)

# Embeddings
embeddings = HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector DB
db = Chroma(
  persist_directory="./db",
  embedding_function=embeddings
)

def build_prompt(context, query):
  return f"""
    You are an e-commerce assistant.

    Rules:
    - Only use provided products
    - Do not invent products
    - If none match, say you couldn't find one

    Products:
    {context}

    User request:
    {query}

    Return ONLY JSON. No explanation. No text before or after.
    Example:
    {{
      "product": "<product_name>",
      "reason": "<why it's a good fit>",
      "price": "<price>"
    }}
  """

def extract_filters_prompt(query: str):
  prompt = f"""
    Extract structured filters from the user query.

    Return ONLY valid JSON.

    Fields:
    - category: one of ["mouse", "gpu", "notebook"]
    - price_lte: number (if user specifies max price)

    Examples:

    Query: "cheap mouse under 50"
    Output:
    {{"category": "mouse", "price_lte": 50}}

    Query: "gpu for AI"
    Output:
    {{"category": "gpu"}}

    Query: "{query}"
    Output:
  """
  return HumanMessage(content=prompt)

def extract_filters_llm(query: str):
  try:
    prompt = extract_filters_prompt(query)
    response = llm.invoke(prompt).content.strip()
    return json.loads(response)
  except:
    return {}


def build_where(filters: dict):
  conditions = []
  if "category" in filters:
    conditions.append({"category": filters["category"]})
  if "price_lte" in filters:
    conditions.append({"price": {"$lte": filters["price_lte"]}})
  if not conditions:
    return None
  if len(conditions) == 1:
    return conditions[0]
  return {"$and": conditions}


def answer(query: str):
  filters = extract_filters_llm(query)
  where = build_where(filters)

  retriever = db.as_retriever(
    search_kwargs={
      "k": 2,
      "filter": where
    }
  )

  docs = retriever.invoke(query)
  context = "\n\n".join([d.page_content for d in docs])
  prompt = build_prompt(context, query)
  response = llm.invoke(prompt).content.strip()

  if response.startswith("```"):
    response = response.split("```")[1]
  
  if response.startswith("json"):
    response = response.replace("json", "")

  return json.loads(response)

if __name__ == "__main__":
  start_time = time.time()
  query = "I need a mouse under $50 for gaming. What do you suggest?"
  result = answer(query)

  print("\n=== ANSWER ===\n")
  print(result)
  
  end_time = time.time()
  elapsed_time = end_time - start_time
  print(f"\n=== EXECUTION TIME ===\n{elapsed_time:.2f} seconds")


# Ministral 3.3B stats:
# - Latency: 16.83 seconds 
# - Latency: 19.67 seconds
# - Latency: 10.30 seconds

# qwen3.5-0.8b
# - Latency: 4.51 seconds
# - Latency: 3.82 seconds
# - Latency: 3.32 seconds