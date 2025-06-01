import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage  # FIXED: Correct import

class State(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def run_command(cmd: str) -> str:
    """
    Takes a command line prompt and executes it on the user's machine and 
    returns the output of the command.
    Example: run_command(cmd="ls") where ls is the command to list the files.
    """
    try:
        result = os.popen(cmd).read()  # FIXED: Changed os.system to os.popen for output capture
        return result.strip()
    except Exception as e:
        return f"Error executing command: {e}"

from dotenv import load_dotenv
load_dotenv()

llm = init_chat_model(
    model_provider="openai",
    model="gpt-4.1",
    api_key=os.getenv("OPENAI_API_KEY")  # FIXED: Load key from environment
)

llm_with_tool = llm.bind_tools(tools=[run_command])

def chatbot(state: State):
    system_prompt = SystemMessage(content="""
        You are an AI Coding assistant who takes an input from the user and based on available
        tools you choose the correct tool and execute the commands.
                                  
        You can even execute commands and help the user with the output of the command.

        Always make sure to keep your generated codes and files in chat_gpt/ folder. You can create one if not already there.                           
    """)

    message = llm_with_tool.invoke([system_prompt] + state["messages"])
    return {"messages": [message]}

tool_node = ToolNode(tools=[run_command])

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)
