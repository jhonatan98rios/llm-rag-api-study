import json

from langchain_core.messages import HumanMessage
from .retriever import db
from ..llm.lmstudio import llm

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