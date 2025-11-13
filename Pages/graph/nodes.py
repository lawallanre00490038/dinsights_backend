from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from .state import AgentState
from .model import llm
from typing import Literal
from .tools import complete_python_task
from dotenv import load_dotenv
import os

load_dotenv()


tools = [complete_python_task]

model = llm.bind_tools(tools)
# tool_executor = ToolExecutor(tools)

with open(os.path.join(os.path.dirname(__file__), "../prompts/main_prompt.md"), "r",  encoding="utf-8") as file:
    prompt = file.read()

chat_template = ChatPromptTemplate.from_messages([
    ("system", prompt),
    ("placeholder", "{messages}"),
])
model = chat_template | model

def create_data_summary(state: AgentState) -> str:
    summary = ""
    variables = []
    for d in state["input_data"]:
        variables.append(d.variable_name)
        summary += f"\n\nVariable: {d.variable_name}\n"
        summary += f"Description: {d.data_description}"
    
    if "current_variables" in state:
        remaining_variables = [v for v in state["current_variables"] if v not in variables]
        for v in remaining_variables:
            summary += f"\n\nVariable: {v}"
    return summary



def route_to_tools(
    state: AgentState,
) -> Literal["tools", "__end__"]:
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, end the graph.
    """

    if messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"


MAX_HISTORY_MESSAGES = 10 

def call_model(state: AgentState):
    current_data_template = """The following data is available:\n{data_summary}"""
    current_data_message = HumanMessage(content=current_data_template.format(data_summary=create_data_summary(state)))

    # Get the existing chat history (which includes messages from previous turns)
    chat_history = state["messages"]

    # 1. 🛑 Truncate the history to keep only the most recent messages.
    truncated_history = chat_history[-MAX_HISTORY_MESSAGES:] 

    messages_for_llm = [current_data_message] + truncated_history

    # 3. Construct the input dictionary the prompt template expects
    llm_input = {"messages": messages_for_llm}
    

    # 4. Invoke the model with the dictionary input
    try:
        llm_outputs = model.invoke(llm_input) # Pass the dictionary directly!
    except Exception as exc:
        # Handle cases where the model attempted to call a tool that wasn't registered
        err_text = str(exc)
        if "attempted to call tool" in err_text or "tool call validation failed" in err_text or "tool_use_failed" in err_text:
            # Try to auto-extract the failed function call from the exception (if present) and run it inside complete_python_task
            import re
            fn_match = re.search(r"<function=(.*?)></function>", err_text)
            if fn_match:
                fn_text = fn_match.group(1)
                # Build a tiny python snippet that prints the result of the helper call
                auto_code = f"print({fn_text})"
                try:
                    # Execute the helper call inside the safe tool environment
                    args = {"thought": "Auto-executing helper from failed tool call", "python_code": auto_code, "graph_state": {k: v for k, v in state.items() if k != 'messages'}}
                    auto_result, auto_updates = complete_python_task.invoke(args)
                    # Return the auto-execution result as a human message so the model can see it and continue
                    return {
                        "messages": [HumanMessage(content=f"Auto-executed helper result:\n{auto_result}")],
                        "intermediate_outputs": [current_data_message.content],
                    }
                except Exception:
                    # fall through to guidance if auto-execution fails
                    pass

            # Create a helpful AI message prompting the model to call only the allowed tool
            guidance = (
                "I attempted to call a helper as a separate tool but that is not allowed. "
                "If you need to inspect data or create plots, call the single tool `complete_python_task` and put any calls to helpers (e.g., available_columns, dataset_info, plot_mean_pie, plot_scatter) inside the `python_code` string.\n\n"
                "Example (preferred):\n"
                "complete_python_task({{\n  'thought': 'I will inspect the dataset columns and then plot',\n  'python_code': \"print(available_columns('sales_sample')); plot_mean_pie('sales_sample')\"\n}})\n\n"
                "Please regenerate a response that calls only `complete_python_task` when performing analysis."
            )
            # Return an AIMessage so the graph can continue without crashing
            return {
                "messages": [HumanMessage(content=guidance)],
                "intermediate_outputs": [current_data_message.content],
            }
        # Re-raise if it's some other unexpected error
        raise
    # If the model didn't produce any tool calls but the user's request clearly
    # asks for analysis/visualization, ask the model once to produce a tool call.
    intent_keywords = ["plot", "visualize", "chart", "histogram", "scatter", "boxplot", "mean", "median", "trend", "describe", "summary", "show me"]
    def _detect_intent(messages):
        # look at the latest user message(s)
        for m in reversed(messages):
            try:
                text = getattr(m, "content", str(m)).lower()
            except Exception:
                continue
            for kw in intent_keywords:
                if kw in text:
                    return True
        return False

    no_tool_calls = not (hasattr(llm_outputs, "tool_calls") and getattr(llm_outputs, "tool_calls"))
    tool_suggestion_done = bool(state.get("tool_suggestion_done", False))

    if no_tool_calls and _detect_intent(messages_for_llm) and not tool_suggestion_done:
        # Ask the model to produce a single complete_python_task tool call.
        followup = HumanMessage(content=(
            "The user's request appears to require data analysis or visualization. "
            "Please return exactly one tool call to `complete_python_task` with the `python_code` that performs the analysis and appends any figures to `plotly_figures`. "
            "Use helpers like `available_columns('<var>')`, `dataset_info('<var>')`, and the plotting helpers (plot_mean_pie, plot_scatter, plot_histogram, plot_boxplot).")
        )
        retry_messages = [current_data_message, followup] + truncated_history
        llm_input_retry = {"messages": retry_messages}
        llm_outputs = model.invoke(llm_input_retry)
        # mark in returned state that we've suggested a tool call so we don't loop
        return {
            "messages": [llm_outputs],
            "intermediate_outputs": [current_data_message.content],
            "tool_suggestion_done": True,
        }

    return {
        "messages": [llm_outputs],
        "intermediate_outputs": [current_data_message.content]
    }



def call_tools(state: AgentState):
    last_message = state["messages"][-1]
    tool_messages = []
    state_updates = {}

    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls"):
        for tc in last_message.tool_calls:
            tool_name = tc["name"]
            args = tc["args"].copy() 
            
            # Exclude the "messages" list, as it contains non-serializable objects.
            safe_state = {k: v for k, v in state.items() if k != "messages"} 
            # read whether we've already attempted an automated retry in this graph execution
            auto_retry_done = bool(safe_state.get("auto_retry_done", False))
            
            # Inject the safe state copy into tool args
            if "graph_state" not in args:
                args["graph_state"] = safe_state 

            # Call the tool
            if tool_name == complete_python_task.name:
                result, updates = complete_python_task.invoke(args) 
                
                tool_messages.append(
                    ToolMessage(content=str(result), name=tool_name, tool_call_id=tc["id"])
                )
                # Apply updates (like 'current_variables' or 'output_image_paths')
                state_updates.update(updates)
                # If the tool returned an execution error or hints about missing columns, attempt one auto-retry
                try:
                    result_text = str(result)
                except Exception:
                    result_text = ""

                failure_indicators = ["Column/DF hints", "not found", "not a valid file", "Traceback", "Error code:"]
                if any(ind in result_text for ind in failure_indicators) and not auto_retry_done:
                    # Prepare a failure message including the intermediate outputs (if present)
                    failure_hint = result_text
                    # Create a compact data summary to help the model correct column names
                    data_summary_msg = HumanMessage(content="The following data is available:\n" + create_data_summary(state))
                    retry_message = HumanMessage(content=(
                        "The previous python execution failed with the following error:\n" +
                        failure_hint +
                        "\n\nPlease regenerate a corrected `complete_python_task` tool call.\n"
                        "Use `available_columns(<variable>)` or `dataset_info(<variable>)` to discover schema and prefer the provided plotting helpers (plot_mean_pie, plot_bar_with_mean, etc.).\n"
                        "Return exactly one tool call for `complete_python_task` with corrected `python_code`."
                    ))

                    # Construct messages for the retry: data summary + retry hint + recent history
                    retry_messages = [data_summary_msg, retry_message] + state["messages"][-MAX_HISTORY_MESSAGES:]
                    llm_input = {"messages": retry_messages}
                    llm_outputs = model.invoke(llm_input)

                    # If the new LLM output contains tool_calls, attempt to execute them (single pass)
                    if hasattr(llm_outputs, "tool_calls") and llm_outputs.tool_calls:
                        for retry_tc in llm_outputs.tool_calls:
                            if retry_tc["name"] == complete_python_task.name:
                                retry_args = retry_tc["args"].copy()
                                if "graph_state" not in retry_args:
                                    retry_args["graph_state"] = safe_state
                                retry_result, retry_updates = complete_python_task.invoke(retry_args)
                                tool_messages.append(
                                    ToolMessage(content=str(retry_result), name=complete_python_task.name, tool_call_id=retry_tc.get("id"))
                                )
                                # Merge any updates from the retry
                                state_updates.update(retry_updates)
                    # mark that we've attempted an automated retry so we don't loop
                    state_updates["auto_retry_done"] = True

    state_updates["messages"] = state["messages"] + tool_messages
    
    return state_updates
