import pandas as pd
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_experimental.tools import PythonAstREPLTool
from langchain_groq import ChatGroq

from src.ds_tools import analyze_clusters, predict_trend, set_dataframe


# Save history chat messages
class AgentState(TypedDict):
    messages : list


def build_data_analyst_graph(df: pd.DataFrame, api_key: str):

    set_dataframe(df)

    # SETUP TOOLS
    python_tool = PythonAstREPLTool(locals={"df": df}, name="python_interpreter")
    
    custom_tools = [analyze_clusters, predict_trend]
    all_tools = [python_tool] + custom_tools

    tool_nodes = ToolNode(all_tools)

    # SETUP MODEL
    llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=api_key, temperature=0)

    model_wtih_tools = llm.bind_tools(custom_tools)

    # DEFINE NODES

    def call_model(state: AgentState):
        """Thinking Node: Given message -> Call LLM -> Return new message"""
        messages =  state["messages"]

         # Add system prompt if first message
        if len(messages) == 1 or not isinstance(messages[0], SystemMessage):
            system_prompt = SystemMessage(content="""You are a data analyst assistant. 
You have access to tools for clustering and trend prediction.
When answering questions:
1. Use tools when appropriate
2. Provide clear, concise answers
3. Stop after giving your final answer - don't call tools unnecessarily
4. If a tool was already called and gave results, summarize them instead of calling again""")

            messages = [system_prompt] + messages

        response = model_wtih_tools.invoke(messages)
        
        return {"messages": [response]}
    
    def should_continue(state: AgentState):
        """Condition Edge: Decide LLM wants to call more tools or finish"""
        messages = state["messages"]
        last_message = messages[-1]

        # If LLM generated tool calls -> go to tool nodes
        if last_message.tool_calls:
            return "tools"
        
        return END
    
    # DRAW GRAPH

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_nodes)

    # Starting point -> Agent
    workflow.set_entry_point("agent")

    # Agent -> Checking condition
    workflow.add_conditional_edges("agent", should_continue)

    # Tools -> Back to Agent (ReAct Loop)
    workflow.add_edge("tools", "agent")

    # Complete graph
    app = workflow.compile()

    return app
