from langchain_core.tools import tool
from typing import Tuple, Dict, Any, Annotated

import sys
import os
import json
import pickle
import uuid
import re
from io import StringIO
from types import ModuleType
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

import pandas as pd
import numpy as np
import plotly
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import sklearn

from Pages.graph.read_files import load_csv_with_fallback


BASE_EXEC_CONTEXT = {
    "__builtins__": __builtins__,
    "pd": pd,
    "plotly": plotly,
    "px": px,
    "go": go,
    "np": np,
    "sklearn": sklearn,
}

PROTECTED_NAMES = set(BASE_EXEC_CONTEXT.keys()) | {"plotly_figures"}
IMPORT_PATTERN = re.compile(r"(^|[\n\r;])\s*(from\s+\w|import\s)", re.IGNORECASE)

@tool
def complete_python_task(
    thought: str,
    python_code: str,
    graph_state: Annotated[Dict[str, Any], InjectedState]  # mark this as injected
) -> Tuple[str, Dict[str, Any]]:
    """Completes a python task

    Args:
        thought: Internal thought about the next action to be taken, and the reasoning behind it. This should be formatted in MARKDOWN and be high quality.
        python_code: Python code to be executed to perform analyses, create a new dataset or create a visualization.
    """
    current_variables = graph_state.get("current_variables") or {}

    if not isinstance(current_variables, dict):
        print(f"WARNING: Resetting 'current_variables' from {type(current_variables)} to dict.")
        current_variables = {}

    # Load input datasets if any
    for input_dataset in graph_state.get("input_data", []):

        if isinstance(input_dataset, dict):
            variable_name = input_dataset.get("variable_name")
            data_path = input_dataset.get("data_path")
            data_content = input_dataset.get("data_content")
        else:
            # Fallback for when it's still the InputData class instance
            variable_name = input_dataset.variable_name
            data_path = getattr(input_dataset, "data_path", None)
            data_content = getattr(input_dataset, "data_content", None)

        if variable_name and variable_name not in current_variables:
            # Prefer inline content if provided (bytes or str); otherwise fall back to path
            if data_content is not None:
                current_variables[variable_name] = load_csv_with_fallback(data_content)
            elif data_path:
                current_variables[variable_name] = load_csv_with_fallback(data_path)
            else:
                # Neither path nor content provided. Raise a clear error so the API can return a helpful message
                raise ValueError(
                    f"No data provided for variable '{variable_name}'.\n"
                    "Provide either `data_content` (inline CSV string or bytes) in the JSON request, "
                    "or a valid `data_path` pointing to an uploaded file on the server.\n"
                    "If your frontend sends multipart/form-data, adjust the backend to accept file uploads or send the file content as a string in `data_content`."
                )

    if IMPORT_PATTERN.search(python_code):
        warning_message = (
            "Imports are not allowed in python_code. Please reuse the already available libraries "
            "(pandas as pd, numpy as np, plotly, px, go, sklearn)."
        )
        updated_state = {
            "intermediate_outputs": [
                {"thought": thought, "code": python_code, "output": warning_message}
            ],
            "current_variables": current_variables,
        }
        return warning_message, updated_state

    os.makedirs("images/plotly_figures/pickle", exist_ok=True)

    old_stdout = sys.stdout
    buffer = StringIO()
    sys.stdout = buffer
    exec_globals: Dict[str, Any] = dict(BASE_EXEC_CONTEXT)
    exec_globals.update(current_variables)
    exec_globals["plotly_figures"] = []
    # Provide helpers that make it easier for model-generated code to inspect datasets
    def _dataset_info(name: str) -> Dict[str, Any]:
        df = exec_globals.get(name)
        if df is None:
            raise ValueError(f"Dataset '{name}' is not available in the execution context.")
        try:
            cols = list(df.columns)
            dtypes = {c: str(t) for c, t in df.dtypes.items()}
            head = df.head(3).to_dict(orient="list")
            return {"columns": cols, "dtypes": dtypes, "head": head, "rows": len(df)}
        except Exception:
            raise ValueError(f"Unable to inspect dataset '{name}'. Is it a pandas DataFrame?")

    def _available_columns(name: str) -> list:
        return _dataset_info(name)["columns"]

    def _safe_get(df, col):
        # Safe getter that raises a clear KeyError listing available columns
        if col in df.columns:
            return df[col]
        raise KeyError(f"Column '{col}' not found. Available columns: {list(df.columns)}")

    # Inject helpers into execution globals for the generated code to use
    exec_globals["dataset_info"] = _dataset_info
    exec_globals["available_columns"] = _available_columns
    exec_globals["safe_get"] = _safe_get

    # High-level plotting helpers to make it easier for the model to create correct charts
    def plot_mean_pie(df, value_cols=None, labels=None, title=None):
        """Create a pie chart of means for the provided numeric columns.
        - df: pandas DataFrame or name of variable already in exec_globals
        - value_cols: list of column names or None to auto-select numeric columns
        - labels: labels for slices (optional)
        """
        import plotly.express as px
        # If df is a string, resolve from exec_globals
        if isinstance(df, str):
            df_obj = exec_globals.get(df)
        else:
            df_obj = df
        if df_obj is None:
            raise ValueError("DataFrame not found for plot_mean_pie")
        if value_cols is None:
            value_cols = [c for c in df_obj.columns if pd.api.types.is_numeric_dtype(df_obj[c])]
        means = df_obj[value_cols].mean()
        fig = px.pie(values=means.values, names=value_cols if labels is None else labels, title=title or "Mean by Column")
        exec_globals["plotly_figures"].append(fig)
        return fig

    def plot_bar_with_mean(df, value_col, title=None):
        """Create a bar chart for a column with a mean line."""
        import plotly.graph_objects as go
        if isinstance(df, str):
            df_obj = exec_globals.get(df)
        else:
            df_obj = df
        if df_obj is None:
            raise ValueError("DataFrame not found for plot_bar_with_mean")
        if value_col not in df_obj.columns:
            raise KeyError(f"Column '{value_col}' not found. Available columns: {list(df_obj.columns)}")
        mean_val = df_obj[value_col].mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_obj.index.astype(str), y=df_obj[value_col], name=value_col))
        fig.add_hline(y=mean_val, line_dash="dash", line_color="red", annotation_text=f"mean={mean_val:.2f}")
        fig.update_layout(title=title or f"{value_col} with mean")
        exec_globals["plotly_figures"].append(fig)
        return fig

    exec_globals["plot_mean_pie"] = plot_mean_pie
    exec_globals["plot_bar_with_mean"] = plot_bar_with_mean
    
    def plot_scatter(df, x, y, color=None, title=None):
        """Create a scatter plot using px.scatter.
        df: DataFrame or variable name
        x,y: column names
        color: optional column name for color grouping
        """
        if isinstance(df, str):
            df_obj = exec_globals.get(df)
        else:
            df_obj = df
        if df_obj is None:
            raise ValueError("DataFrame not found for plot_scatter")
        if x not in df_obj.columns or y not in df_obj.columns:
            raise KeyError(f"Columns '{x}' or '{y}' not found. Available: {list(df_obj.columns)}")
        fig = px.scatter(df_obj, x=x, y=y, color=color, title=title or f"{y} vs {x}")
        exec_globals["plotly_figures"].append(fig)
        return fig

    def plot_histogram(df, col, nbins=None, title=None):
        """Create a histogram for a single column."""
        if isinstance(df, str):
            df_obj = exec_globals.get(df)
        else:
            df_obj = df
        if df_obj is None:
            raise ValueError("DataFrame not found for plot_histogram")
        if col not in df_obj.columns:
            raise KeyError(f"Column '{col}' not found. Available columns: {list(df_obj.columns)}")
        fig = px.histogram(df_obj, x=col, nbins=nbins, title=title or f"Histogram of {col}")
        exec_globals["plotly_figures"].append(fig)
        return fig

    def plot_boxplot(df, cols, title=None):
        """Create boxplots for one or multiple columns."""
        if isinstance(df, str):
            df_obj = exec_globals.get(df)
        else:
            df_obj = df
        if df_obj is None:
            raise ValueError("DataFrame not found for plot_boxplot")
        # Accept single column name or list
        if isinstance(cols, str):
            cols = [cols]
        missing = [c for c in cols if c not in df_obj.columns]
        if missing:
            raise KeyError(f"Columns {missing} not found. Available columns: {list(df_obj.columns)}")
        df_melt = df_obj[cols]
        fig = px.box(df_melt, y=cols, title=title or "Boxplot")
        exec_globals["plotly_figures"].append(fig)
        return fig

    exec_globals["plot_scatter"] = plot_scatter
    exec_globals["plot_histogram"] = plot_histogram
    exec_globals["plot_boxplot"] = plot_boxplot

    try:
        exec(python_code, exec_globals)
        output_text = buffer.getvalue()
    except Exception as exc:
        output_text = buffer.getvalue()
        # Build a helpful error message and include dataset/column hints when possible
        base_error = f"{exc.__class__.__name__}: {exc}"
        suggestion = ""

        # If it's a KeyError, provide available columns from any DataFrame in the context
        if isinstance(exc, KeyError):
            try:
                hints = {}
                for name, val in exec_globals.items():
                    try:
                        # check for pandas DataFrame
                        if hasattr(val, "columns"):
                            hints[name] = list(val.columns)
                    except Exception:
                        continue
                if hints:
                    suggestion = "\nColumn/DF hints:\n"
                    for nm, cols in hints.items():
                        suggestion += f"- {nm}: {cols}\n"
            except Exception:
                suggestion = ""

        # If the exception suggests an invalid file/data input, offer a clearer message
        elif "not a valid file" in str(exc).lower() or "is not a file" in str(exc).lower():
            try:
                available = {n: (len(v) if hasattr(v, '__len__') else 'unknown') for n, v in exec_globals.items() if hasattr(v, 'columns')}
                suggestion = f"\nAvailable DataFrames: {list(available.keys())}\n"
            except Exception:
                suggestion = ""

        error_message = base_error + suggestion
        updated_state = {
            "intermediate_outputs": [
                {"thought": thought, "code": python_code, "output": (output_text + ("\n" if output_text else "") + error_message).strip()}
            ],
            "current_variables": current_variables,
        }
        return error_message, updated_state
    finally:
        sys.stdout = old_stdout

    next_current_variables: Dict[str, Any] = {}
    for name, value in exec_globals.items():
        if name in PROTECTED_NAMES or name.startswith("__"):
            continue
        if isinstance(value, ModuleType):
            continue
        next_current_variables[name] = value

    plots = exec_globals.get("plotly_figures", [])
    new_files = []
    if isinstance(plots, list) and plots:
        for figure in plots:
            file_name = f"{uuid.uuid4()}.pickle"
            file_path = os.path.join("images", "plotly_figures", "pickle", file_name)
            try:
                with open(file_path, "wb") as handle:
                    pickle.dump(figure, handle)
                new_files.append(file_name)
            except Exception as exc:
                print(f"Error saving Plotly figure {file_name}: {exc}")

    updated_state: Dict[str, Any] = {
        "intermediate_outputs": [{"thought": thought, "code": python_code, "output": output_text}],
        "current_variables": next_current_variables,
    }

    if new_files:
        updated_state["output_image_paths"] = new_files

    return output_text, updated_state




def convert_plotly_pickles_to_json(image_paths: list) -> list:
    """
    Reads Plotly figure pickle files and converts them into a list of
    JSON-serializable dictionaries (Plotly JSON specification).
    """
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

    return chart_data_list
