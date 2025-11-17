

from langchain_core.messages import HumanMessage
from typing import List, Optional, Dict, Any
from langgraph.graph import StateGraph
from Pages.graph.state import AgentState
from Pages.graph.nodes import call_model, call_tools, route_to_tools
from Pages.data_models import InputData

class PythonChatbot:
    def __init__(self, *, persistent_state: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.reset_chat()
        self.graph = self.create_graph()
        if persistent_state:
            self.chat_history = persistent_state.get("chat_history", self.chat_history)
            self.intermediate_outputs = persistent_state.get("intermediate_outputs", self.intermediate_outputs)
            self.output_image_paths = persistent_state.get("output_image_paths", self.output_image_paths)
            self.current_variables = persistent_state.get("current_variables", self.current_variables)
        
    def create_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node('agent', call_model)
        workflow.add_node('tools', call_tools)

        workflow.add_conditional_edges('agent', route_to_tools)

        workflow.add_edge('tools', 'agent')
        workflow.set_entry_point('agent')
        # Compile without config parameter - recursion_limit is set during invoke
        return workflow.compile()

    def user_sent_message(self, user_query, input_data: List[InputData]):

        if not input_data:
            input_data = []

        all_starting_images = sum(self.output_image_paths.values(), [])
        starting_image_paths_set = set(all_starting_images)

        input_state = {
            "messages": self.chat_history + [HumanMessage(content=user_query)],
            "output_image_paths": all_starting_images,
            "input_data": input_data,
            "intermediate_outputs": [],
            "current_variables": self.current_variables,
            "tool_suggestion_done": False,
            "auto_retry_done": False,
            "tool_call_count": 0,
        }

        try:
            result = self.graph.invoke(input_state, config={"recursion_limit": 50})
        except Exception as exc:
            # Capture diagnostic information so callers can return it to clients or logs
            err_msg = f"Graph invocation failed: {exc.__class__.__name__}: {exc}"
            # Keep the intermediate outputs for debugging traces
            self.intermediate_outputs.append({"error": err_msg})
            # Re-raise a RuntimeError to be handled by the API layer
            raise RuntimeError(err_msg) from exc

        # Update chat history and image paths
        self.chat_history = result["messages"]
        new_image_paths = set(result.get("output_image_paths", [])) - starting_image_paths_set
        if new_image_paths:
            self.output_image_paths[len(self.chat_history) - 1] = list(new_image_paths)
        if "intermediate_outputs" in result:
            self.intermediate_outputs.extend(result["intermediate_outputs"])
        if "current_variables" in result and isinstance(result["current_variables"], dict):
            self.current_variables = result["current_variables"]


    def reset_chat(self):
        self.chat_history = []
        self.intermediate_outputs = []
        self.output_image_paths = {}
        self.current_variables = {}

    def snapshot(self) -> Dict[str, Any]:
        return {
            "chat_history": self.chat_history,
            "intermediate_outputs": self.intermediate_outputs,
            "output_image_paths": self.output_image_paths,
            "current_variables": self.current_variables,
        }
