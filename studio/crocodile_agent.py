from typing_extensions import TypedDict
import os
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, BaseMessage,AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

from crocodile_dataset import crocodile_test, common_names_dict


model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0)

class OverallState(TypedDict, total = False):
    messages: list
    action: str
    crocodile_specie: str

def make_resumes(state: OverallState):
    prompt = (
        f"Talk about the notes of {state['crocodile_specie']}, knowing that these are the notes about it: "
        + common_names_dict[state["crocodile_specie"]]
        + """ You should use your notes to help you, and also use your internal and deductive knowledge
        to convey confidence in what you're saying. 
        Example answer: 
        Name: _crocodile_name
        Diet: _such_food
        Convey confidence no matter what happens.
        At the end, say that you will make a conclusion about this, but don't go deep, just one line talking that you will go do this in the next message"""
    )
    result = model.invoke([HumanMessage(content=prompt)])
    return {"messages": state["messages"] + [AIMessage(content=result.content)], "action": "conclusion"}

def make_jokes(state: OverallState):
    prompt = (
        f"Make a joke about the crocodile {state['crocodile_specie']}, knowing that these are the notes about it: "
        + common_names_dict[state["crocodile_specie"]]
    )
    result = model.invoke([HumanMessage(content=prompt)])
    return {"messages": state["messages"] + [AIMessage(content=result.content)], "action": "none"}

def condition(state: OverallState):
    action = state.get("action")
    if action == "resume":
        return "make_resumes"
    elif action == "joke":
        return "make_jokes"

    else:
        return END
      

def assistant(state: OverallState):
    last_message = state["messages"][-1]

    if isinstance(last_message, BaseMessage):
        question = last_message.content
    else:
        return {"messages": state["messages"], "action": "none"}

    if "make a resume" in question:
        crocodile_name = question.replace("make a resume about ", "").strip()
        return {
            "messages": state["messages"],
            "action": "resume",
            "crocodile_specie": crocodile_name,
        }
        
    elif "make a joke" in question:
        crocodile_name = question.replace("make a joke about ", "").strip()
        return {
            "messages": state["messages"],
            "action": "joke",
            "crocodile_specie": crocodile_name,
        }
    
    elif state["action"] == "conclusion":
        conclusion = model.invoke(
            [HumanMessage(content=f"make a conclusion about {question} and say without fail say goodbye")]
        )
        return {"messages": state["messages"] + [AIMessage(content=conclusion.content)], "action": "none"}


# --- Graph definition ---
builder = StateGraph(OverallState)
builder.add_node("assistant", assistant)
builder.add_node("make_jokes", make_jokes)
builder.add_node("make_resumes", make_resumes)
builder.add_edge(START,"assistant")
builder.add_conditional_edges("assistant", condition)
builder.add_edge("make_jokes", "assistant")
builder.add_edge("make_resumes","assistant")
graph = builder.compile()

