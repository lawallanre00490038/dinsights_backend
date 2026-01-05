from threading import Lock
from typing import  Optional, List, Any
from uuid import uuid4
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import StateGraph, START

import pandas as pd
import json
import re
from langchain_core.messages import ToolMessage

from src.injestion import store_file_and_register_dataset
from src.tools import describe_dataframe, llm_with_tool, load_csv, generate_chart
from src.state import AnalysisState
from src.registry import DATASET_REGISTRY, update_dataset_access, cleanup_expired_datasets, get_registry_stats, delete_dataset
from src.db import conn

# ----------------------------
# ChatRequest definition
# ----------------------------
class InputData(BaseModel):
    key: str
    value: Any


class ChatRequest(BaseModel):
    user_query: str
    input_data: Optional[List[InputData]] = None
    session_id: Optional[str] = Field(
        None, description="Identifier used to persist conversation state across requests."
    )
    reset_session: bool = Field(
        False, description="If true, resets the state for the provided session_id before handling the request."
    )


# ----------------------------
# FastAPI app and session lock
# ----------------------------
app = FastAPI(
    title="Data Analysis API",
    description="AI-powered data analysis and visualization API",
    version="1.0.0"
)
_session_lock = Lock()

# Add rate limiting middleware
from src.rate_limit import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware, requests_per_minute=100)



def load_data_node(state: AnalysisState):
    dataset_id = state.get("dataset_id")
    df = state.get("df")

    if dataset_id and df is None and dataset_id in DATASET_REGISTRY:
        # Update last accessed time
        update_dataset_access(dataset_id)
        
        table_name = DATASET_REGISTRY[dataset_id]["table"]
        df = conn.execute(f"SELECT * FROM {table_name}").df()
        meta = DATASET_REGISTRY[dataset_id]

        numeric_cols = list(df.select_dtypes(include="number").columns)
        categorical_cols = list(df.select_dtypes(include="object").columns)

        df_sample = df.head(2).to_dict(orient="records")

        return {
            "df": df,  # kept in state, NOT sent to LLM
            "messages": [
                SystemMessage(
                    content=(
                        "Dataset loaded successfully.\n"
                        f"Rows: {meta['rows']}\n"
                        f"Columns: {meta['columns']}\n"
                        f"Numeric columns: {numeric_cols}\n"
                        f"Categorical columns: {categorical_cols}\n"
                        f"Sample rows: {df_sample}\n"
                        "You may now analyze the dataset or call generate_chart."
                    )
                )
            ],
        }
    elif dataset_id is None:
        # No dataset uploaded - inform the LLM
        return {
            "messages": [
                SystemMessage(
                    content=(
                        "No dataset has been uploaded. "
                        "You are a helpful data analyst assistant. "
                        "When users ask what you do, explain that you help analyze data and create visualizations, "
                        "but they need to upload a dataset first. "
                        "Do NOT attempt to use any tools (describe_dataframe, generate_chart) until a dataset is uploaded. "
                        "Provide friendly, natural language responses to general questions."
                    )
                )
            ],
        }

    return {}



# ----------------------------
# Plot extraction node
# ----------------------------
def extract_plots_node(state: AnalysisState):
    """Extract plots from tool results and add them to state."""

    plots = state.get("plots", []) or []
    messages = state.get("messages", [])
    
    # Track which plots we've already extracted to avoid duplicates
    existing_plot_count = len(plots)
    
    # Look for ToolMessage results from generate_chart
    # ToolNode wraps tool results in ToolMessage, and the content is the return value
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        if msg.name != "generate_chart":
            continue
        
        print(f"Found ToolMessage: name={getattr(msg, 'name', 'N/A')}, content_type={type(msg.content)}")
        # Check if this is from generate_chart tool by checking the tool name or content structure
        try:
            # Parse the tool result - it could be a dict or JSON string
            if isinstance(msg.content, str):
                result = json.loads(msg.content)
            elif isinstance(msg.content, dict):
                result = msg.content
            else:
                print(f"Skipping ToolMessage with unexpected content type: {type(msg.content)}")
                continue
            
            print(f"Parsed tool result: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
            
            # Check if this looks like a generate_chart result (has "plot" key)
            if isinstance(result, dict) and "plot" in result:
                plot_data = result["plot"]
                # Only add if it's not already in plots
                # Use a simple check - compare JSON strings to avoid deep comparison issues
                plot_json = json.dumps(plot_data, sort_keys=True)
                existing_plots_json = [json.dumps(p, sort_keys=True) for p in plots]
                if plot_json not in existing_plots_json:
                    plots.append(plot_data)
                    print(f"✓ Extracted plot from tool result: {len(plots)} plots total")
                else:
                    print(f"Plot already exists, skipping duplicate")
            else:
                print(f"Tool result does not contain 'plot' key: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
            # Log error but continue
            print(f"Error extracting plot from tool result: {e}, content: {str(msg.content)[:100]}")
            continue
    
    # Only return plots if we added new ones
    if len(plots) > existing_plot_count:
        return {"plots": plots}
    
    return {}


SYSTEM_PROMPT = """
You are a highly intelligent data analyst agent.
You MUST NOT write Python, pandas, or matplotlib code under any circumstance.
All analysis and chart generation must be performed using tools.

IMPORTANT: Only use tools when a dataset has been loaded. Check system messages to know if a dataset is available.

When a dataset IS available:
- A pandas DataFrame is loaded and available in memory as the dataset.
- You can use describe_dataframe and generate_chart tools to analyze the data.

Rules for tool usage:
1. When creating charts, you MUST use ONLY the `generate_chart` tool.
2. You can generate up to 5 charts maximum to provide comprehensive analysis.
3. Generate multiple charts when the user's question requires different perspectives, comparisons, or when analyzing multiple aspects of the data.
4. Ensure chart arguments are valid for the dataset columns and chart type.
5. AFTER successfully generating charts with tools, you MUST provide a natural language explanation for EACH chart in a conversational, engaging way.
6. When explaining multiple charts, number them (1, 2, 3...) and explain what insights each chart reveals.

CRITICAL: When you see tool results (ToolMessage), do NOT repeat or include the raw JSON data, plot specifications, or technical details from the tool response. 

For EACH chart generated, provide a conversational explanation that includes:
- What the chart shows (e.g., "I've created a bar chart showing the top-selling product lines")
- Key insights from the chart (e.g., "The chart reveals that Classic Cars generate the highest sales, followed by Motorcycles")
- What it means for the user (e.g., "This suggests focusing marketing efforts on these top-performing categories")

When multiple charts are generated, explain each one separately in a natural, conversational flow. For example:
"To give you a comprehensive view, I've created several visualizations:

1. First, a bar chart showing sales by product line - this reveals that Classic Cars are our top performer with $X in sales.

2. Second, a line chart tracking sales trends over time - we can see a steady growth pattern in Q4.

3. Finally, a pie chart showing the distribution across regions - North America accounts for 45% of total sales."

When the user asks for charts or visualizations, use the `generate_chart` tool with the following parameters:
- chart_type: One of 'bar', 'line', 'scatter', 'histogram', 'boxplot', 'pie'
- x_column: Name of the column for x-axis (required for most chart types)
- y_column: Name of the column for y-axis (required for bar, line, scatter, pie)
- color_column: Optional column name for color grouping
- title: Optional title for the chart

Chart type guidelines:
- Use 'bar' for categorical comparisons and top-N rankings
- Use 'line' for trends over time or sequential data
- Use 'scatter' to compare two continuous variables
- Use 'histogram' to show distributions of a single variable
- Use 'boxplot' to show spread, outliers, and distributions by category
- Use 'pie' for proportional representation of categories

IMPORTANT: 
- When responding after tool execution, write ONLY natural language explanations. Never include JSON, plot data, or technical specifications in your response.
- NEVER include function call syntax like <function=...> in your responses. Only make actual tool calls when appropriate.
- If no dataset is available, provide friendly explanations without attempting to use tools.
"""



# ----------------------------
# LLM analyst node
# ----------------------------
# Configuration: Adjust based on your needs and token limits
MAX_MESSAGES = 10  # Increased from 2 to allow better context retention
MAX_CONTEXT_TOKENS = 8000  # Adjust based on your LLM's context window

def llm_analyst(state: AnalysisState):

    system_message = SystemMessage(content=SYSTEM_PROMPT)

    recent_messages = state.get("messages", [])[-MAX_MESSAGES:]
    msgs = add_messages([system_message], recent_messages)

    for msg in msgs:
        if isinstance(msg, ToolMessage):
            print(msg)

    result = llm_with_tool.invoke(msgs)
    
    # Debug: Check if the LLM made tool calls
    if hasattr(result, "tool_calls") and result.tool_calls:
        print(f"LLM made {len(result.tool_calls)} tool call(s): {[tc.get('name') for tc in result.tool_calls]}")
    else:
        print("LLM did not make any tool calls")

    return {
        "messages": [result],
        "saved_messages": [result]
    }



# ----------------------------
# Tool setup
# ----------------------------
tools = [load_csv, describe_dataframe, generate_chart]
tool_node = ToolNode(tools)

# ----------------------------
# State graph setup
# ----------------------------
graph = StateGraph(AnalysisState)
graph.add_node("LLM", llm_analyst)
graph.add_node("load_data", load_data_node)
graph.add_node("tools", tool_node)
graph.add_node("extract_plots", extract_plots_node)

graph.add_edge(START, "load_data")
graph.add_edge("load_data", "LLM")

# tools_condition returns "tools" if tool_calls exist, "__end__" otherwise
graph.add_conditional_edges("LLM", tools_condition)  # checks last AIMessage.tool_calls
graph.add_edge("tools", "extract_plots")
graph.add_edge("extract_plots", "LLM")  # loop back to LLM after extracting plots

graph = graph.compile()


# ----------------------------
# API endpoints
# ----------------------------
@app.post("/api/chat")
async def analyze(
    prompt: str = Form(...),
    file: UploadFile | None = File(None)
):
    try:
        dataset_id = None

        if file:
            try:
                dataset_id = store_file_and_register_dataset(file)
            except HTTPException:
                raise  # Re-raise HTTP exceptions from ingestion
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing uploaded file: {str(e)}"
                )

        if not prompt or not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        query = HumanMessage(content=prompt.strip())
        
        try:
            result = graph.invoke({
                "messages": [query],
                "saved_messages": [query], 
                "dataset_id": dataset_id,
                "described" : False,
                "plots": []
            })
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing request: {str(e)}"
            )

        print("\n\n\nsaved_messages", result.get("saved_messages", []))
        
        # Find the last AIMessage that is a final response (no tool calls)
        # This will be the natural language explanation after tools have executed
        messages = result.get("messages", [])
        message_content = "Response generated."
        
        # Look backwards through messages to find the last AIMessage without tool calls
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                # Check if this AIMessage has tool calls
                has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls and len(msg.tool_calls) > 0
                if not has_tool_calls:
                    # This is a final response without tool calls - use it
                    content = msg.content if msg.content else "Response generated."
                    # Clean up any function call syntax that might have leaked through
                    # Remove patterns like <function=name>{...}</function>
                    content = re.sub(r'<function=[^>]+>.*?</function>', '', content, flags=re.DOTALL)
                    content = re.sub(r'<function_calls>.*?</function_calls>', '', content, flags=re.DOTALL)
                    message_content = content.strip() if content.strip() else "Response generated."
                    break
        
        return {
            "messages": message_content,
            "plots": result.get("plots", [])
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )



# ----------------------------
# Registry Management Endpoints
# ----------------------------
@app.get("/api/registry/stats")
async def get_stats():
    """Get registry statistics."""
    stats = get_registry_stats()
    return stats

@app.post("/api/registry/cleanup")
async def cleanup_datasets(background_tasks: BackgroundTasks):
    """Manually trigger cleanup of expired datasets."""
    def run_cleanup():
        deleted = cleanup_expired_datasets()
        print(f"Cleanup completed: {deleted} datasets deleted")
    
    background_tasks.add_task(run_cleanup)
    return {"message": "Cleanup started in background", "status": "processing"}

@app.delete("/api/registry/dataset/{dataset_id}")
async def delete_dataset_endpoint(dataset_id: str):
    """Delete a specific dataset."""
    if dataset_id not in DATASET_REGISTRY:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    success = delete_dataset(dataset_id)
    if success:
        return {"message": f"Dataset {dataset_id} deleted successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete dataset")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "registry_loaded": len(DATASET_REGISTRY) > 0}

@app.on_event("startup")
async def startup_event():
    """Run cleanup on server startup."""
    print("Starting up... Running initial dataset cleanup...")
    deleted = cleanup_expired_datasets()
    print(f"Startup cleanup completed: {deleted} expired datasets deleted")
    print(f"Registry loaded: {len(DATASET_REGISTRY)} datasets")


# ----------------------------
# Custom validation error handler
# ----------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    try:
        safe = jsonable_encoder(
            exc.errors(),
            custom_encoder={bytes: lambda o: o.decode('latin1', errors='replace')}
        )
    except Exception:
        safe = [dict(err) for err in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": safe})




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=10000, reload=True)