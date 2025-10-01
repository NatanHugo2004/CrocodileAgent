import os
from typing_extensions import TypedDict
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, BaseMessage,AIMessage, SystemMessage
from dotenv import load_dotenv
from typing import List, Any
from langgraph.graph import MessagesState
from rag_query import recuperar
from build_index import carregar_documentos_txt, gerar_passagens, construir_indice
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def make_resumes(a:str):
    """
    Returns a summary that can be used for summaries with notes, habitat, country, 
    weight and conservation status for the crocodile with common name 'a', this is made by a
    RAG query, so the information will be in a rank of more relevants information about the crocodile.    Args:
        a: Common name of the crocodile (e.g., 'American Crocodile')
    Returns:
        String with all available information about the crocodile.
    """
    docs = carregar_documentos_txt(pasta="data")
    passagens = gerar_passagens(docs)
    construir_indice(passagens, out_dir="storage")
    info = recuperar(f"information about {a}", k=4)
    return info


def assistant(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    
    return {"messages": state["messages"] + [response]}
        
tools = [make_resumes]
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

prompt = HumanMessage(os.getenv("prompt_base"))
initial_state = {
    "messages": [prompt]
}

result = graph.invoke(initial_state)

for m in result['messages']:
    m.pretty_print()