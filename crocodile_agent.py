from typing_extensions import TypedDict
import os
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, BaseMessage,AIMessage, SystemMessage
from dotenv import load_dotenv
from typing import List, Any
from langgraph.graph import MessagesState

load_dotenv()

from crocodile_dataset import common_names_dict_notes, common_names_dict_conservation, common_names_dict_habitat, common_names_dict_country, common_names_dict_weight

api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def make_resumes(a:str):
    """
    Returns a summary that can be used for both summaries and jokes with notes, habitat, country, weight and conservation status for the crocodile with common name 'a'.    Args:
        a: Common name of the crocodile (e.g., 'American Crocodile')
    Returns:
        String with all available information about the crocodile.
    """
    info = (
    "Notes: " + common_names_dict_notes.get(a, "No info available.") +
    " | Habitat: " + common_names_dict_habitat.get(a, "No info available.") +
    " | Country: " + common_names_dict_country.get(a, "No info available.") +
    " | Weight: " + str(common_names_dict_weight.get(a, "No info available.")) +
    " | Conservation: " + common_names_dict_conservation.get(a, "No info available.")
)
    return info

def make_jokes(a:str):
    """
    Returns information for jokes that includes habitat, country, weight and conservation status for the crocodile with common name 'a'. Combine this with your knowledge
        Args:
        a: Common name of the crocodile (e.g., 'American Crocodile')
    Returns:
        String with all available information about the crocodile.
    """
    info = (
    "Notes: " + common_names_dict_notes.get(a, "No info available.") +
    " | Habitat: " + common_names_dict_habitat.get(a, "No info available.") +
    " | Country: " + common_names_dict_country.get(a, "No info available.") +
    " | Weight: " + str(common_names_dict_weight.get(a, "No info available.")) +
    " | Conservation: " + common_names_dict_conservation.get(a, "No info available.")
)
    return info 

def assistant(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}
        
tools = [make_resumes, make_jokes]
llm_with_tools = model.bind_tools(tools, parallel_tool_calls=False)

# --- Graph definition ---
builder = StateGraph(MessagesState)

builder.add_node("assistant", assistant)
builder.add_edge(START,"assistant")
builder.add_node("tools", ToolNode(tools))
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
graph = builder.compile()

# --- Example run ---

prompt = HumanMessage(content=input(""))
initial_state = {
    "messages": [prompt]
}

result = graph.invoke(initial_state)

for m in result['messages']:
    m.pretty_print()