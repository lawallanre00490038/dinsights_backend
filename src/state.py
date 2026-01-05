from typing_extensions import TypedDict
from typing import Annotated, List, Optional, Any
import pandas as pd
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class AnalysisState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    saved_messages: Annotated[List[AnyMessage], add_messages]
    dataset_id: Optional[str]
    csv_path: Optional[str]
    df: Optional[pd.DataFrame]
    plots: Optional[List[dict]]   # Plotly JSON specs
    described: Optional[bool]
