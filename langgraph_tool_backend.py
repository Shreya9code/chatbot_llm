from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import google.generativeai as genai
import json

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

# Mudra Database (simplified example)
MUDRA_DATABASE = {
    "pataka": {
        "sanskrit": "Pataka",
        "meaning": "Flag",
        "description": "All fingers extended and held together, thumb slightly bent",
        "usage": "Used for beginnings, clouds, forests, saying no",
        "dance_forms": ["Bharatanatyam", "Kathak", "Odissi"]
    },
    "tripataka": {
        "sanskrit": "Tripataka",
        "meaning": "Three-part flag",
        "description": "Like Pataka but ring finger bent",
        "usage": "Crown, tree, fire, arrow",
        "dance_forms": ["Bharatanatyam", "Kathakali"]
    }
}

@tool
def get_mudra_details(mudra_name: str) -> str:
    """Get detailed information about a specific dance mudra."""
    mudra_key = mudra_name.lower().strip()
    
    if mudra_key in MUDRA_DATABASE:
        mudra = MUDRA_DATABASE[mudra_key]
        return f"""
Mudra: {mudra['sanskrit']} ({mudra['meaning']})

Description: {mudra['description']}

Common Usages: {mudra['usage']}

Dance Forms: {', '.join(mudra['dance_forms'])}

This mudra is fundamental in Indian classical dance and has specific applications in storytelling.
"""
    else:
        return f"I don't have detailed information about '{mudra_name}' in my database. Please ask about common mudras like Pataka, Tripataka, etc."

@tool
def list_common_mudras() -> str:
    """List the most common dance mudras with their basic meanings."""
    mudra_list = []
    for name, info in MUDRA_DATABASE.items():
        mudra_list.append(f"{info['sanskrit']} - {info['meaning']}")
    
    return "Common Mudras:\n" + "\n".join(f"- {mudra}" for mudra in mudra_list)

tools = [get_mudra_details, list_common_mudras]
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# Checkpointer
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# Graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')
chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)