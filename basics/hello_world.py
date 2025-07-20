from typing import TypedDict
from langgraph.graph import END, START, StateGraph
from util.langgraph_util import display


class HelloWorldState(TypedDict):
    message: str


def hello(state: HelloWorldState):
    print(f"Hello Node: {state['message']}")
    return {"message": "Hello "+state['message']}


def bye(state: HelloWorldState):
    print(f"Bye Node: {state['message']}")
    return {"message": "Bye "+state['message']}


graph = StateGraph(HelloWorldState)
graph.add_node("hello",hello)
graph.add_node("bye",bye)

graph.add_edge(START,"hello")
graph.add_edge("hello","bye")
graph.add_edge("bye",END)

runnable = graph.compile()
display(runnable)
output = runnable.invoke({"message": "Bharath"})
print(output)