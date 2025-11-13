from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from threading import Lock

from starlette.responses import JSONResponse
from Pages.data_models import InputData
from Pages.backend import PythonChatbot
from Pages.graph.tools import convert_plotly_pickles_to_json
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder


app = FastAPI()
_chat_sessions: Dict[str, PythonChatbot] = {}
_session_lock = Lock()

# 1. Define the Pydantic Request Model
class ChatRequest(BaseModel):
    user_query: str
    input_data: Optional[List[InputData]] = None
    session_id: Optional[str] = Field(None, description="Identifier used to persist conversation state across requests.")
    reset_session: bool = Field(False, description="If true, resets the state for the provided session_id before handling the request.")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):

    # Print a safe summary of the request (avoid attempting to JSON-encode raw bytes)
    print(f"Received Chat Request: user_query={'(present)' if request.user_query else '(empty)'}")
    print(f"Session ID: {request.session_id}, Reset Session: {request.reset_session}")
    print("|n\nInput Data:", request.input_data if request.input_data else 0)
    # Summarize input_data without printing raw contents
    if request.input_data:
        summaries = []
        for item in request.input_data:
            # item may be a pydantic model; print variable name and whether content/path is present
            try:
                var = getattr(item, 'variable_name', None) or (item.get('variable_name') if isinstance(item, dict) else None)
                has_path = bool(getattr(item, 'data_path', None) or (item.get('data_path') if isinstance(item, dict) else None))
                has_content = bool(getattr(item, 'data_content', None) or (item.get('data_content') if isinstance(item, dict) else None))
                content_len = None
                if has_content:
                    raw = getattr(item, 'data_content', None) or (item.get('data_content') if isinstance(item, dict) else None)
                    try:
                        content_len = len(raw)
                    except Exception:
                        content_len = None
                summaries.append({
                    'variable_name': var,
                    'has_path': has_path,
                    'has_content': has_content,
                    'content_len': content_len,
                })
            except Exception:
                summaries.append({'variable_name': None})
        print('Input Data summary:', summaries)
    else:
        print('Input Data: None')
    
    try:
        chatbot: PythonChatbot
        if request.session_id:
            with _session_lock:
                if request.reset_session and request.session_id in _chat_sessions:
                    del _chat_sessions[request.session_id]
                chatbot = _chat_sessions.get(request.session_id)
                if chatbot is None:
                    chatbot = PythonChatbot()
                    _chat_sessions[request.session_id] = chatbot
        else:
            chatbot = PythonChatbot()
        
        # Get image paths BEFORE the run (to track only NEW ones)
        starting_image_paths_set = set(sum(chatbot.output_image_paths.values(), []))
        
        # 2. Invoke the core logic
        try:
            chatbot.user_sent_message(request.user_query, request.input_data)
        except RuntimeError as re_err:
            # This typically wraps a graph/tool invocation failure. Return a 400 with details.
            import logging
            logging.error("Graph execution error: %s", re_err, exc_info=True)
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": str(re_err),
                    "hint": "Tool or model execution failed. See server logs/intermediate_outputs for details."
                },
            )
        
        # Get the final AI message
        result = chatbot.chat_history[-1] 
        final_text_response = result.content 

        # 3. Identify and convert new charts
        final_image_paths_dict = chatbot.output_image_paths
        
        # Get the paths created during this specific run
        all_new_images = []
        for paths in final_image_paths_dict.values():
            all_new_images.extend(paths)
            
        # Filter for only those images generated in this specific message turn
        new_image_files = list(set(all_new_images) - starting_image_paths_set)

        # Convert new pickle files to JSON chart data
        chart_data_list = convert_plotly_pickles_to_json(new_image_files)
        
        # 4. Return the data
        print(
            f"\n\nChat Response: {final_text_response}, Charts Generated: {len(chart_data_list)}"
        )
        return {
            "success": True,
            "data": {
                "text": final_text_response,
                "charts": chart_data_list
            }
        }
    
    except Exception as e:
        import logging
        logging.error(f"Chat API Error: {e}", exc_info=True)
        # Return a 500 error with a JSON body
        return JSONResponse(
            status_code=500, 
            content={
                "success": False, 
                "error": f"An internal server error occurred: {str(e)}"
            }
        )


# Custom handler to avoid UnicodeDecodeError when validation errors include raw bytes
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    # exc.errors() may contain bytes inside values that jsonable_encoder will try to decode as utf-8
    # Use a custom encoder for bytes that decodes using latin1 with replacement to avoid crashes.
    try:
        safe = jsonable_encoder(exc.errors(), custom_encoder={bytes: lambda o: o.decode('latin1', errors='replace')})
    except Exception:
        # Fallback to stringifying the errors if encoding still fails
        safe = [dict(err) for err in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": safe})
