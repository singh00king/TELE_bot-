from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

groq_llm = init_chat_model(
    model="openai/gpt-oss-20b",
    model_provider="groq",
)

response = groq_llm.invoke("Say 'LLM is working!' and explain RAG in one sentence.")

print(response.content)
