from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import google.generativeai as genai
import re

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

# Dance Mudra Knowledge Base
MUDRA_SYSTEM_PROMPT = """You are an expert AI assistant specialized in Indian classical dance mudras (both hand and full body gestures). 
Your role is to provide clear, structured information about mudras without using markdown formatting.

IMPORTANT FORMATTING RULES:
- NEVER use asterisks (*), bold, or markdown
- Use clean, natural language with proper line breaks
- Structure information clearly but simply
- Use bullet points with hyphens (-) instead of stars
- Keep explanations concise and educational

When discussing mudras, always include:
- Sanskrit name and meaning
- Hand position and finger arrangements
- Common usages in dance and mythology
- Related mudras or variations

Provide practical, actionable information for dance practitioners."""

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def format_response(text: str) -> str:
    """Clean up response formatting by removing markdown artifacts"""
    # Remove asterisks used for bold/emphasis
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # Replace markdown bullets with clean hyphens
    text = re.sub(r'^\s*[\*\-]\s+', '- ', text, flags=re.MULTILINE)
    
    # Clean up numbered lists
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()

def chat_node(state: ChatState):
    messages = state['messages']
    # Ensure system message is included
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SystemMessage(content=MUDRA_SYSTEM_PROMPT)] + messages
    
    response = llm.invoke(messages)
    cleaned_content = format_response(response.content)
    response.content = cleaned_content
    return {"messages": [response]}
    response = llm.invoke(messages)
    return {"messages": [response]}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)