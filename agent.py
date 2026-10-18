from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from llm import groq_llm

#memory = InMemorySaver()

agent = create_agent(
    model=groq_llm,
    tools=[],
    #checkpointer=memory,      # here memory stores the temp data
    system_prompt="You are a helpful assistant.")

agent.invoke({"message": [{"role": "user", "content": "What is RAG?"}]})