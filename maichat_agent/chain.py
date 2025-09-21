from typing import List, Optional, TypedDict
from typing_extensions import NotRequired

from langchain_core.output_parsers import JsonOutputToolsParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from maichat_agent.utils.calories import calories_calculator
from maichat_agent.utils.menu import menu_recommendation


class GenerativeUIState(TypedDict, total=False):
    input: NotRequired[HumanMessage]
    result: Optional[str]
    """Plain text response if no tool was used."""
    tool_calls: Optional[List[dict]]
    """A list of parsed tool calls."""
    tool_result: Optional[dict]
    """The result of a tool call."""
    """The result of a tool call."""


def invoke_model(state: GenerativeUIState, config: RunnableConfig) -> GenerativeUIState:
    tools_parser = JsonOutputToolsParser()
    initial_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant with expertise in nutrition. You're provided a list of tools, and an input from the user. Answer only in Indonesian language\n"
                + "Your job is to provide personalized food recommendations and detailed nutritional information based on the user's needs and preferences. When given an input from the user, determine whether you can offer advice directly or if you should utilize a specific tool to assist further. Always aim to provide actionable and evidence-based guidance in a clear and supportive manner.",
            ),
            MessagesPlaceholder("input"),
        ]
    )

    model = ChatOpenAI(model="gpt-5-mini", temperature=0, streaming=True)
    tools = [
        calories_calculator,
        menu_recommendation,
    ]
    model_with_tools = model.bind_tools(tools)
    chain = initial_prompt | model_with_tools
    input_msg = state.get("input")
    if input_msg is None:
        raise ValueError("Missing 'input' in state.")
    result = chain.invoke({"input": input_msg}, config)

    if not isinstance(result, AIMessage):
        raise ValueError("Invalid result from model. Expected AIMessage.")

    if isinstance(result.tool_calls, list) and len(result.tool_calls) > 0:
        parsed_tools = tools_parser.invoke(result, config)
        # Do not include HumanMessage in returned state
        return {"tool_calls": parsed_tools}
    else:
        # Do not include HumanMessage in returned state
        return {"result": str(result.content)}


def invoke_tools_or_return(state: GenerativeUIState) -> str:
    if "result" in state and isinstance(state["result"], str):
        return END
    elif "tool_calls" in state and isinstance(state["tool_calls"], list):
        return "invoke_tools"
    else:
        raise ValueError("Invalid state. No result or tool calls found.")


def invoke_tools(state: GenerativeUIState) -> GenerativeUIState:
    tools_map = {
        "calories-calculator": calories_calculator,
        "menu-recommendation": menu_recommendation,
    }

    if (
        "tool_calls" in state
        and isinstance(state["tool_calls"], list)
        and len(state["tool_calls"]) > 0
    ):
        tool_calls = state["tool_calls"]
        tool = tool_calls[0]
        selected_tool = tools_map[tool["type"]]
        # Do not include HumanMessage in returned state
        return {
            "tool_result": selected_tool.invoke(tool["args"]),
        }
    else:
        raise ValueError("No tool calls found in state.")


def create_graph():
    workflow = StateGraph(GenerativeUIState)
    workflow.add_node("invoke_model", invoke_model)  # type: ignore
    workflow.add_node("invoke_tools", invoke_tools)
    workflow.add_conditional_edges("invoke_model", invoke_tools_or_return)
    workflow.set_entry_point("invoke_model")
    workflow.set_finish_point("invoke_tools")

    graph = workflow.compile()
    return graph
