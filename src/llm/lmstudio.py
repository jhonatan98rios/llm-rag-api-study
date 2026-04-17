from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
  base_url="http://localhost:1234/v1",
  api_key="lm-studio",
  model="qwen3.5-0.8b",
  temperature=0.3,
  max_tokens=150
)