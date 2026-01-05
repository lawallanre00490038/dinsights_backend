from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from pandas.core.frame import DataFrame
from langchain_groq import ChatGroq
from langgraph.prebuilt import InjectedState
from typing import Annotated, Dict, Any
import os
from dotenv import load_dotenv
import plotly.express as px
import plotly.io as pio 
import json
import pandas as pd
from typing import Dict, Any, Annotated, Optional


load_dotenv()


@tool
def load_csv(path: str) -> str:
    """Load a CSV file and return column info."""
    df = pd.read_csv(path)
    return f"Loaded CSV with columns: {list(df.columns)} and shape {df.shape}"


@tool
def describe_dataframe(
    graph_state: Annotated[Dict[str, Any], InjectedState] = None
) -> str:
    """
    Generate basic descriptive statistics for the dataset in the state.
    Returns information about columns, data types, shape, and basic statistics.
    """

    df = graph_state.get("df") if graph_state else None
    
    if df is None:
        return "No dataset available. Please upload a CSV file first."
    
    if not isinstance(df, pd.DataFrame):
        return f"Expected DataFrame, got {type(df)}"
    
    if df.empty:
        return "The dataset is empty."

    
    print("I, the describe df tool have been called\n\n")
    
    # Generate descriptive information
    info_parts = [
        f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns",
        f"\nColumns: {list(df.columns)}",
        f"\nData Types:\n{df.dtypes.to_string()}",
        f"\n\nDescriptive Statistics:\n{df.describe().to_string()}",
    ]
    
    # Add info about missing values
    missing = df.isnull().sum()
    if missing.any():
        info_parts.append(f"\n\nMissing Values:\n{missing[missing > 0].to_string()}")
    
    return "".join(info_parts)


@tool
def generate_chart(
    chart_type: str,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    color_column: Optional[str] = None,
    title: Optional[str] = None,
    graph_state: Annotated[Dict[str, Any], InjectedState] = None
) -> dict:
    """
    Generate a Plotly chart from the dataset in the state.
    
    Args:
        chart_type: Type of chart to create. Options: 'bar', 'line', 'scatter', 'histogram', 'boxplot', 'pie'
        x_column: Name of the column to use for x-axis (required for bar, line, scatter, boxplot)
        y_column: Name of the column to use for y-axis (required for bar, line, scatter)
        color_column: Optional column name to use for color grouping
        title: Optional title for the chart
        graph_state: Injected state containing the DataFrame (automatically provided)
    
    Returns:
        Dictionary with 'plot' (Plotly JSON spec) and 'message' (success message)
    """
    df = graph_state.get("df")
    if df is None:
        raise ValueError("No dataset available.")

    valid_columns = set(df.columns)

    # Validate columns
    for col in [x_column, y_column, color_column]:
        if col and col not in valid_columns:
            return {
                "error": "INVALID_COLUMN",
                "message": f"Column '{col}' does not exist.",
                "valid_columns": list(valid_columns)
            }

    
    import plotly.express as px
    import json
    import plotly.io as pio
    
    fig = None
    chart_type = chart_type.lower()

    if chart_type == "histogram" and not pd.api.types.is_numeric_dtype(df[x_column]):
        chart_type = "bar"
    
    if chart_type == "bar":
        # For bar charts, aggregate data if x_column is categorical to reduce verbosity
        # This prevents sending thousands of individual data points
        if x_column and y_column and not pd.api.types.is_numeric_dtype(df[x_column]) and pd.api.types.is_numeric_dtype(df[y_column]):
            # Aggregate by x_column (sum by default for bar charts)
            if color_column:
                df_agg = df.groupby([x_column, color_column], as_index=False)[y_column].sum()
                fig = px.bar(df_agg, x=x_column, y=y_column, color=color_column, title=title)
            else:
                df_agg = df.groupby(x_column, as_index=False)[y_column].sum()
                fig = px.bar(df_agg, x=x_column, y=y_column, title=title)
        else:
            fig = px.bar(df, x=x_column, y=y_column, color=color_column, title=title)
    elif chart_type == "line":
        fig = px.line(df, x=x_column, y=y_column, color=color_column, title=title)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x_column, y=y_column, color=color_column, title=title)
    elif chart_type == "histogram":
        fig = px.histogram(df, x=x_column, color=color_column, title=title)
    elif chart_type == "boxplot":
        fig = px.box(df, x=x_column, y=y_column, color=color_column, title=title)
    elif chart_type == "pie":
        fig = px.pie(df, names=x_column, values=y_column, title=title)
    else:
        raise ValueError(f"Unsupported chart type {chart_type}")
    
    fig.update_layout(template=None)
    
    # Use to_dict() instead of to_json() to get cleaner Python-native data structures
    # This avoids base64 encoding issues
    plot_dict = fig.to_dict()
    
    # Clean up the plot JSON for Plotly React
    # Remove unnecessary fields and simplify structure
    cleaned_plot = _clean_plotly_json(plot_dict)
    

    print(f"Generated {chart_type} chart")
    return {
        "plot": cleaned_plot, 
        "message": f"{chart_type} chart generated successfully"
    }

def _clean_plotly_json(plot_dict: dict) -> dict:
    """
    Clean Plotly dict to remove unnecessary data and optimize for frontend.
    Converts numpy arrays to lists for JSON serialization.
    """
    import copy
    import numpy as np
    
    cleaned = copy.deepcopy(plot_dict)
    
    def convert_to_list(data):
        """Convert data to list format for JSON serialization."""
        if isinstance(data, list):
            # Recursively convert any numpy arrays in the list
            return [convert_to_list(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.floating)):
            return data.item()
        elif isinstance(data, dict):
            # Recursively clean dict
            return {k: convert_to_list(v) for k, v in data.items()}
        elif isinstance(data, tuple):
            return [convert_to_list(item) for item in data]
        else:
            return data
    
    # Clean data traces
    if "data" in cleaned:
        for trace in cleaned["data"]:
            # Convert x and y to lists
            if "x" in trace:
                trace["x"] = convert_to_list(trace["x"])
            
            if "y" in trace:
                trace["y"] = convert_to_list(trace["y"])
            
            # Convert other array fields
            for field in ["z", "text", "customdata", "ids"]:
                if field in trace:
                    trace[field] = convert_to_list(trace[field])
            
            # Remove unnecessary fields that Plotly React doesn't need
            unnecessary_fields = ["uid", "ids", "customdata", "meta", "selectedpoints", "error_x", "error_y"]
            for field in unnecessary_fields:
                if field in trace:
                    del trace[field]
            
            # Simplify marker if it exists
            if "marker" in trace and isinstance(trace["marker"], dict):
                if "pattern" in trace["marker"] and isinstance(trace["marker"]["pattern"], dict):
                    if not trace["marker"]["pattern"] or trace["marker"]["pattern"].get("shape") == "":
                        del trace["marker"]["pattern"]
    
    # Clean layout
    if "layout" in cleaned:
        layout = cleaned["layout"]
        # Remove template reference if empty
        if "template" in layout and (not layout["template"] or layout["template"] == {}):
            del layout["template"]
        
        # Remove unnecessary layout fields
        unnecessary_layout_fields = ["autosize", "dragmode", "hovermode", "selectdirection", "uirevision"]
        for field in unnecessary_layout_fields:
            if field in layout:
                del layout[field]
    
    return cleaned




def convert_plotly_pickles_to_json(image_paths: list) -> list:
    """
    Reads Plotly figure pickle files and converts them into a list of
    JSON-serializable dictionaries (Plotly JSON specification).
    
    Note: This function is kept for backward compatibility but may not be needed
    if using the new generate_chart tool directly.
    """
    import pickle
    import os
    
    chart_data_list = []
    base_dir = "images/plotly_figures/pickle"

    for filename in image_paths:
        file_path = os.path.join(base_dir, filename)
        
        try:
            # 1. Load the Plotly figure object from the pickle file
            with open(file_path, 'rb') as f:
                fig = pickle.load(f)

            # 2. Remove the template from the figure object to make it more compact
            fig.update_layout(template=None)

            fig_dict = json.loads(pio.to_json(fig))
            
            chart_data_list.append(fig_dict)
            
        except Exception as e:
            print(f"Error processing Plotly file {filename}: {e}")
            continue
    print(chart_data_list)
    return chart_data_list







# Initialize the language model
llm = ChatGroq(
  model="llama-3.1-8b-instant",
  api_key=os.getenv("GROQ_API_KEY")
)

# Bind tools to the LLM - this is required for the LLM to know about and call tools
tools_list = [describe_dataframe, generate_chart]
llm_with_tool = llm.bind_tools(tools_list)

