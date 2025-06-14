from graphState import GraphState

# import agents 
from agents.base_agent import BaseAgent

# import tools
from tool_general.save_tool import save_tool
from tool_general.wiki_tool import wiki_tool
from tool_RAG.RAG_query_tool import rag_query_tool

from dotenv import load_dotenv
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, AnyMessage

class AgentFlowController:

    def __init__(self):
        self.builder = StateGraph(GraphState)
        self._createNodes()
        self._createEdges()
        self.graph = self.builder.compile()

    def _createNodes(self) -> None:
        self.builder.add_node("basic_agent", self._basicAgent)
    
    def _createEdges(self) -> None:
        self.builder.set_entry_point("basic_agent")
        self.builder.add_edge("basic_agent", END)

    def _basicAgent(self, state: GraphState) -> GraphState:
        """Define Basic Agent"""
        agent = BaseAgent(tools=[wiki_tool, rag_query_tool, save_tool])
        last_msg = state["message"]
        print(last_msg)
        print(type(state["message"]))

        result = agent.execute(last_msg)
        return {"message": result, "next_agent": "end"}
    
    def _print_graph(self):
        pass

if __name__ == "__main__":
    load_dotenv()
    # message = input("What is your question? : ")
    message = "What is your question? : what is collaborative filtering, pleas use the vectordb as a tool. save me into a textfile"
    input_state = {"message": message, "next": ""}
    controller = AgentFlowController()
    response = controller.graph.invoke(input_state)
    print("Final state:", response)