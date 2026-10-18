from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from llm import groq_llm
from rag import search, build_vector_db
from tools import predict_kyphosis
from tools import search_private_documents


#memory = InMemorySaver()

agent = create_agent(
    model=groq_llm,
    tools=[search_private_documents, predict_kyphosis],
    #checkpointer=memory,      # here memory stores the temp data
    system_prompt="You are a helpful assistant. You are an intelligent Document Question-Answering Assistant with access to the user's private documents through the search_private_documents tool.")

agent.invoke({"message": [{"role": "user", "content": "What is RAG?"}]})