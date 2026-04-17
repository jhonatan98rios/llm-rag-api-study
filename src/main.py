from time import time
from .rag.prompt import answer
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/search")
def search(q: str = Query(...)):
  start_time = time()
  
  res = answer(q)

  end_time = time()
  elapsed_time = end_time - start_time
  print(f"\n=== EXECUTION TIME ===\n{elapsed_time:.2f} seconds")
  return res

# 5.54 seconds Ministral 3 - 3B
# 2.90 seconds Qwen3.5 - 0.8B
