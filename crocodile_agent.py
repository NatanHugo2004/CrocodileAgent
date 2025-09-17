from typing import Annotated
from typing_extensions import TypedDict
import os
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from crocodile_dataset import crocodile_test, common_names_dict

model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0)

class OverallState(TypedDict, total = False):
    messages: list
    action: str
    crocodile_specie: str

def make_resumes(state: OverallState):
    prompt = f"Talk about the notes off {state["crocodile_specie"]}, knowing that these are the notes about it:" + common_names_dict.get(state["crocodile_specie"], "")
    result = model.invoke(prompt).content
    return {"messages": state["messages"] +[result]}

def make_jokes(state: OverallState):
    prompt = f"Make a joke of  the crocodile {state["crocodile_specie"]}, knowing that these are the notes about it:" + common_names_dict.get(state["crocodile_specie"], "")
    result = model.invoke(prompt).content
    return {"messages": state["messages"] +[result]}

def condition(state: OverallState):
    action = state.get("action")
    if action == "resume":
        return "make_resumes"
    elif action == "joke":
        return "make_jokes"
    else:
        return END  

def assistant(state: OverallState):
    question = state["messages"][-1]
    if "make a resume" in question :
        crocodile_name = question.replace("make a resume about ", "").strip()
        return {"messages": [question], "action": "resume", "crocodile_specie":crocodile_name}
        
    elif "make a joke" in question:
        crocodile_name = question.replace("make a joke about ", "").strip()
        return {"messages": [question], "action": "joke", "crocodile_specie":crocodile_name}
    
    else:
        return {"messages": [question], "action": "none"}
builder = StateGraph(OverallState)
builder.add_node("assistant", assistant)
builder.add_edge(START,"assistant")
builder.add_node("make_jokes",make_jokes)
builder.add_conditional_edges("assistant",condition)
builder.add_node("make_resumes", make_resumes)
builder.add_edge("make_jokes", END)
builder.add_edge("make_resumes",END)
graph = builder.compile()
prompt = f"make a joke about {crocodile_test}"
initial_state = {
    "messages": [prompt]
}
resultado = graph.invoke(initial_state)
for i in resultado['messages']:
    print(i.replace("/n",""))