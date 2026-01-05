from datetime import datetime
from threading import Lock
from typing import  Optional, List, Any, Dict
from uuid import uuid4
import uuid
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from urllib.parse import urlparse
import time, random, traceback
from starlette.responses import JSONResponse

from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import StateGraph, START

import pandas as pd
import json
import re, io
from langchain_core.messages import ToolMessage

from src.injestion import store_file_and_register_dataset
from src.tools import describe_dataframe, llm_with_tool, load_csv, generate_chart
from src.state import AnalysisState
from src.registry import DATASET_REGISTRY, update_dataset_access, cleanup_expired_datasets, get_registry_stats, delete_dataset
from src.db import conn
from src.utils.s3_config import download_file_from_s3

# ----------------------------
# ChatRequest definition
# ----------------------------
class InputData(BaseModel):
    variable_name: str
    data_path: Optional[str] = None
    data_description: Optional[str] = ""
    fileId: Optional[str] = None
    sourceType: Optional[str] = None

class ChatRequest(BaseModel):
    user_query: str = "Analyze the file"
    session_id: Optional[str] = Field(None, description="Identifier used to persist conversation state across requests.")
    reset_session: bool = False
    input_data: Optional[List[InputData]] = None




class ChatMessageResponse(BaseModel):
    id: str
    role: str  # "USER" | "ASSISTANT" | "SYSTEM"
    content: str
    charts: Optional[List[dict]] = Field(default_factory=list)
    references: Optional[List[dict]] = Field(default_factory=list)
    inputData: Optional[List[dict]] = Field(default_factory=list)
    createdAt: datetime


class ChatResponseSchema(BaseModel):
    sessionId: str
    messages: List[ChatMessageResponse]

# New, separate message models so `USER` message does not include charts/inputData
class UserMessage(BaseModel):
    id: str
    role: str  # "USER"
    content: str
    createdAt: datetime

class AssistantMessage(BaseModel):
    id: str
    role: str  # "ASSISTANT"
    content: str
    charts: Optional[List[dict]] = Field(default_factory=list)
    references: Optional[List[dict]] = Field(default_factory=list)
    inputData: Optional[List[dict]] = Field(default_factory=list)
    createdAt: datetime

class ChatResponseV2(BaseModel):
    sessionId: str
    user: UserMessage
    assistant: AssistantMessage


# ----------------------------
# FastAPI app and session lock
# ----------------------------
app = FastAPI(
    title="Data Analysis API",
    description="AI-powered data analysis and visualization API",
    version="1.0.0"
)
_session_lock = Lock()

# Simple in-memory session store mapping session_id -> metadata (e.g., dataset_id)
# This keeps the association between a frontend session and an uploaded dataset
# so subsequent requests that omit `input_data` still operate on the same file.
SESSION_STORE: Dict[str, dict] = {}


def normalize_s3_path(s3_url: str) -> str:
    """Convert various S3 URL formats into the expected "bucket/key" form.

    Supported inputs:
    - s3://bucket/key
    - https://bucket.s3.<region>.amazonaws.com/key (virtual-hosted)
    - https://s3.<region>.amazonaws.com/bucket/key (path-style)
    - https://s3.amazonaws.com/bucket/key
    """
    if not s3_url:
        return s3_url

    if s3_url.startswith("s3://"):
        return s3_url[len("s3://"):]

    parsed = urlparse(s3_url)
    host = parsed.netloc
    path = parsed.path.lstrip("/")

    # virtual-hosted style: bucket.s3.region.amazonaws.com
    if ".s3." in host or host.endswith(".s3.amazonaws.com"):
        bucket = host.split(".s3.")[0]
        key = path
        return f"{bucket}/{key}"

    # path-style: s3.amazonaws.com/bucket/key or s3.region.amazonaws.com/bucket/key
    if host.startswith("s3") or host.endswith("amazonaws.com"):
        parts = path.split("/")
        if len(parts) >= 2:
            bucket = parts[0]
            key = "/".join(parts[1:])
            return f"{bucket}/{key}"

    # Fallback: treat the full path after scheme as bucket/key
    return path


def retry_call(fn, attempts: int = 3, base_delay: float = 0.5, *args, **kwargs):
    """Retry wrapper for transient errors.

    fn can be a callable that accepts args/kwargs. Returns the function's result or raises the last exception.
    """
    last_exc = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            # small jittered exponential backoff
            if i == attempts - 1:
                break
            delay = base_delay * (2 ** i) + random.random() * 0.1
            time.sleep(delay)
    # raise the last exception
    raise last_exc



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





from fastapi import Body

@app.post("/api/chat", response_model=ChatResponseV2)
async def analyze_chat(request: ChatRequest = Body(...)):

    try:
        print(request)
        # Determine session id early so we can persist/restore dataset association
        session_id = request.session_id or str(uuid.uuid4())

        # If requested, reset any session state
        if request.reset_session and request.session_id:
            with _session_lock:
                SESSION_STORE.pop(request.session_id, None)

        dataset_id = None

        # Handle S3 file ingestion (new upload overrides stored dataset for session)
        if request.input_data:
            for item in request.input_data:
                # support both dicts and Pydantic InputData objects
                if isinstance(item, dict):
                    s3_url = item.get("data_path")
                else:
                    s3_url = getattr(item, "data_path", None)

                if s3_url:
                    try:
                        # normalize common S3 URL formats to "bucket/key"
                        normalized = normalize_s3_path(s3_url)
                        # download with retries
                        file_content = retry_call(download_file_from_s3, 3, 0.5, normalized)
                        file_like = UploadFile(filename=s3_url.split("/")[-1], file=io.BytesIO(file_content))
                        dataset_id = store_file_and_register_dataset(file_like)
                        # persist dataset_id for this session
                        with _session_lock:
                            SESSION_STORE[session_id] = {"dataset_id": dataset_id, "updated_at": datetime.utcnow().isoformat()}
                    except HTTPException:
                        raise
                    except Exception as e:
                        raise HTTPException(status_code=500, detail=f"S3 file ingestion error: {e}")
        else:
            # No input_data provided: try to restore dataset_id from session store
            if request.session_id:
                with _session_lock:
                    sess = SESSION_STORE.get(request.session_id)
                    if sess:
                        dataset_id = sess.get("dataset_id")

        if not request.user_query or not request.user_query.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        query = HumanMessage(content=request.user_query.strip())

        # Invoke state graph with the (possibly restored) dataset_id
        invoke_state = {
            "messages": [query],
            "saved_messages": [query],
            "dataset_id": dataset_id,
            "described": False,
            "plots": []
        }

        try:
            # attempt graph.invoke with retries to handle transient LLM/tool failures
            result = retry_call(lambda state: graph.invoke(state), 3, 1.0, invoke_state)
        except Exception as tool_err:
            # Log the error and continue with a graceful assistant response
            error_id = str(uuid.uuid4())
            print(f"Graph invocation error (ref {error_id}): {tool_err}")
            traceback.print_exc()
            result = {"messages": [], "plots": []}
            result = {"messages": [], "plots": []}

            # Try to generate fallback charts directly on the server if a dataset is available
            fallback_plots = []
            try:
                if dataset_id and dataset_id in DATASET_REGISTRY:
                    table_name = DATASET_REGISTRY[dataset_id]["table"]
                    df = conn.execute(f"SELECT * FROM {table_name}").df()

                    numeric_cols = list(df.select_dtypes(include="number").columns)
                    categorical_cols = list(df.select_dtypes(include="object").columns)

                    # Attempt up to 3 fallback charts: bar, histogram, boxplot (when applicable)
                    # 1) Bar: top categories by first numeric column
                    if categorical_cols and numeric_cols:
                        try:
                            res = generate_chart(
                                chart_type="bar",
                                x_column=categorical_cols[0],
                                y_column=numeric_cols[0],
                                title=f"Top {categorical_cols[0]} by {numeric_cols[0]}",
                                graph_state={"df": df},
                            )
                            # handle structured error result from generate_chart
                            if isinstance(res, dict) and "plot" in res:
                                fallback_plots.append(res["plot"])
                            elif isinstance(res, dict) and res.get("error"):
                                print(f"generate_chart returned error: {res}")
                        except Exception as e:
                            print(f"Fallback bar chart failed: {e}")

                    # 2) Histogram: distribution of first numeric column
                    if numeric_cols:
                        try:
                            res = generate_chart(
                                chart_type="histogram",
                                x_column=numeric_cols[0],
                                title=f"Distribution of {numeric_cols[0]}",
                                graph_state={"df": df},
                            )
                            if isinstance(res, dict) and "plot" in res:
                                fallback_plots.append(res["plot"])
                            elif isinstance(res, dict) and res.get("error"):
                                print(f"generate_chart returned error: {res}")
                        except Exception as e:
                            print(f"Fallback histogram failed: {e}")

                    # 3) Boxplot: numeric by category
                    if categorical_cols and numeric_cols:
                        try:
                            res = generate_chart(
                                chart_type="boxplot",
                                x_column=categorical_cols[0],
                                y_column=numeric_cols[0],
                                title=f"Spread of {numeric_cols[0]} by {categorical_cols[0]}",
                                graph_state={"df": df},
                            )
                            if isinstance(res, dict) and "plot" in res:
                                fallback_plots.append(res["plot"])
                            elif isinstance(res, dict) and res.get("error"):
                                print(f"generate_chart returned error: {res}")
                        except Exception as e:
                            print(f"Fallback boxplot failed: {e}")

            except Exception as e:
                print(f"Fallback generation error: {e}")

            # If we generated any fallback plots, attach them and set explanatory message
            if fallback_plots:
                result["plots"] = fallback_plots
                message_content = (
                    f"I couldn't complete the original tool-driven analysis (ref: {error_id}), but I generated some fallback visualizations from the uploaded dataset. "
                    "See the charts attached."
                )
            else:
                # Provide a concise, user-friendly explanation without leaking internals
                message_content = (
                    f"Sorry — I couldn't complete the analysis because a tool failed (ref: {error_id}). "
                    "Please try again, or upload the dataset and retry."
                )

        # If message_content wasn't already set by a tool error, extract AI message content
        if 'message_content' not in locals():
            message_content = "Response generated."
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage):
                    has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
                    if not has_tool_calls:
                        content = msg.content or "Response generated."
                        content = re.sub(r'<function=[^>]+>.*?</function>', '', content, flags=re.DOTALL)
                        content = re.sub(r'<function_calls>.*?</function_calls>', '', content, flags=re.DOTALL)
                        message_content = content.strip() or "Response generated."
                        break


        # Generate IDs and timestamps
        now_iso = datetime.utcnow().isoformat() + "Z"

        # Minimal user_message (only id, role, content, createdAt)
        user_message = {
            "id": str(uuid.uuid4()),
            "role": "USER",
            "content": request.user_query.strip(),
            "createdAt": now_iso
        }

        # Normalize input_data for response: convert Pydantic models to dicts
        input_data_out = []
        if request.input_data:
            for i in request.input_data:
                if hasattr(i, "dict"):
                    input_data_out.append(i.dict())
                else:
                    input_data_out.append(i)

        # Assistant message contains the chat info, charts, references and any inputData
        assistant_message = {
            "id": str(uuid.uuid4()),
            "role": "ASSISTANT",
            "content": message_content,
            "charts": result.get("plots", []),
            "references": [],
            "inputData": input_data_out,
            "createdAt": now_iso
        }

        # Return new typed response with separate user/assistant objects
        return ChatResponseV2(
            sessionId=session_id,
            user=UserMessage(**user_message),
            assistant=AssistantMessage(**assistant_message)
        )

    except HTTPException:
        raise
    except Exception as e:
        # Catch-all: return a graceful assistant response instead of 500
        error_id = str(uuid.uuid4())
        print(f"Unhandled exception in /api/chat (ref {error_id}): {e}")
        traceback.print_exc()
        now_iso = datetime.utcnow().isoformat() + "Z"

        user_message = {
            "id": str(uuid.uuid4()),
            "role": "USER",
            "content": request.user_query.strip() if hasattr(request, 'user_query') else "",
            "createdAt": now_iso
        }

        assistant_message = {
            "id": str(uuid.uuid4()),
            "role": "ASSISTANT",
            "content": (
                f"Sorry — an unexpected error occurred while processing your request (ref: {error_id}). "
                "Please try again, or contact support if the problem persists."
            ),
            "charts": [],
            "references": [],
            "inputData": [],
            "createdAt": now_iso
        }

        return ChatResponseV2(
            sessionId=(request.session_id or str(uuid.uuid4())),
            user=UserMessage(**user_message),
            assistant=AssistantMessage(**assistant_message)
        )





# ----------------------------
# API endpoints
# ----------------------------
# @app.post("/api/chat")
# async def analyze(
#     prompt: str = Form(...),
#     file: UploadFile | None = File(None)
# ):
#     try:
#         dataset_id = None

#         if file:
#             try:
#                 dataset_id = store_file_and_register_dataset(file)
#             except HTTPException:
#                 raise  # Re-raise HTTP exceptions from ingestion
#             except Exception as e:
#                 raise HTTPException(
#                     status_code=500,
#                     detail=f"Error processing uploaded file: {str(e)}"
#                 )

#         if not prompt or not prompt.strip():
#             raise HTTPException(status_code=400, detail="Prompt cannot be empty")

#         query = HumanMessage(content=prompt.strip())
        
#         try:
#             result = graph.invoke({
#                 "messages": [query],
#                 "saved_messages": [query], 
#                 "dataset_id": dataset_id,
#                 "described" : False,
#                 "plots": []
#             })
#         except Exception as e:
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Error processing request: {str(e)}"
#             )

#         print("\n\n\nsaved_messages", result.get("saved_messages", []))
        
#         # Find the last AIMessage that is a final response (no tool calls)
#         # This will be the natural language explanation after tools have executed
#         messages = result.get("messages", [])
#         message_content = "Response generated."
        
#         # Look backwards through messages to find the last AIMessage without tool calls
#         for msg in reversed(messages):
#             if isinstance(msg, AIMessage):
#                 # Check if this AIMessage has tool calls
#                 has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls and len(msg.tool_calls) > 0
#                 if not has_tool_calls:
#                     # This is a final response without tool calls - use it
#                     content = msg.content if msg.content else "Response generated."
#                     # Clean up any function call syntax that might have leaked through
#                     # Remove patterns like <function=name>{...}</function>
#                     content = re.sub(r'<function=[^>]+>.*?</function>', '', content, flags=re.DOTALL)
#                     content = re.sub(r'<function_calls>.*?</function_calls>', '', content, flags=re.DOTALL)
#                     message_content = content.strip() if content.strip() else "Response generated."
#                     break
        
#         return {
#             "messages": message_content,
#             "plots": result.get("plots", [])
#         }
#     except HTTPException:
#         raise  # Re-raise HTTP exceptions
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Unexpected error: {str(e)}"
#         )



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
