from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, AIMessage
from langgraph.graph import StateGraph


class State(TypedDict):
    messages: list[AnyMessage]
    extra_field: int


def node(state: State):
    messages = state["messages"]
    new_message = AIMessage("Hello!")
    return {"messages": messages + [new_message], "extra_field": 10}


builder = StateGraph(State)
builder.add_node(node)
builder.set_entry_point("node")
graph = builder.compile()


class StateGraphBuilder:
    
    def __init__(self):
        self._builder = StateGraph(State)
        self._builder.add_node(node)
        self._builder.set_entry_point("node")
        self._graph = self._builder.compile()

        
    @property
    def graph(self):
        return self._graph



    