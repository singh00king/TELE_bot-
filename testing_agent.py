from agent import agent

# Replace with whatever question you want to ask
user_input = "where arman studied in college?"

# Invoke the agent
response = agent.invoke({"messages": [("user", user_input)]})

# Print the agent's final response
print("\nAgent Response:")
print(response["messages"][-1].content)