import tempfile
import os
import io
import time
from functools import wraps
from contextlib import redirect_stdout, redirect_stderr
from typing import Annotated

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.chat_agent import ChatAgent

# ============= ADD TIMING DECORATOR HERE =============
def track_execution_time(method_name):
    """Decorator to track and display execution time for ML methods"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Print start message (will be captured)
            print(f"Starting {method_name}...")
            
            # Execute the original method
            result = func(*args, **kwargs)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Format time display
            if execution_time < 60:
                time_str = f"{execution_time:.2f} seconds"
            else:
                minutes = int(execution_time // 60)
                seconds = execution_time % 60
                time_str = f"{minutes} min {seconds:.1f} sec"
            
            # Print execution time (will be captured by redirect_stdout)
            print(f"\nTotal execution time for {method_name}: {time_str}")
            print(f"{method_name} completed successfully!")
            
            return result
        return wrapper
    return decorator

# Monkey-patch the ChatAgent methods to add timing
if not hasattr(ChatAgent, "_timing_patched"):
    if hasattr(ChatAgent, '_train_classification'):
        original = ChatAgent._train_classification
        ChatAgent._train_classification = track_execution_time('Classification Training')(original)
    
    if hasattr(ChatAgent, '_train_regression'):
        original = ChatAgent._train_regression
        ChatAgent._train_regression = track_execution_time('Regression Training')(original)
    
    if hasattr(ChatAgent, '_optimize'):
        original = ChatAgent._optimize
        ChatAgent._optimize = track_execution_time('Model Optimization')(original)
    
    ChatAgent._timing_patched = True
# ============= END OF TIMING DECORATOR =============

app = FastAPI(
    title="ML Agent API",
    description="FastAPI backend for DataScienceMLAgent",
    version="1.0.0"
)

# Global instance for state management (equivalent to Streamlit's session_state for single user)
_ml_agent = ChatAgent()

def get_agent() -> ChatAgent:
    return _ml_agent

AgentDep = Annotated[ChatAgent, Depends(get_agent)]

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    paraphrased: str
    response: str
    execution_time: float

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, agent: AgentDep) -> ChatResponse:
    start_time = time.time()
    captured_output = io.StringIO()
    
    try:
        with redirect_stdout(captured_output), redirect_stderr(captured_output):
            paraphrased, response = agent.chat(request.message)
            
        terminal_output = captured_output.getvalue().strip()
        if terminal_output:
            full_response = response + "\n\n**Output:**\n```\n" + terminal_output + "\n```"
        else:
            full_response = response
            
    except Exception as e:
        try:
            # Fallback retry as in original code
            response_str = str(e)
            if "too many values to unpack" not in response_str and "not enough values to unpack" not in response_str:
                raise e
            with redirect_stdout(captured_output), redirect_stderr(captured_output):
                paraphrased, response = agent.chat(request.message)
            
            terminal_output = captured_output.getvalue().strip()
            if terminal_output:
                full_response = response + "\n\n**Output:**\n```\n" + terminal_output + "\n```"
            else:
                full_response = response
                
        except Exception as e2:
            paraphrased = request.message
            full_response = f"❌ Error: {str(e2)}"
            
    execution_time = time.time() - start_time
    
    # Format time display
    if execution_time < 60:
        time_str = f"{execution_time:.2f} seconds"
    else:
        minutes = int(execution_time // 60)
        seconds = execution_time % 60
        time_str = f"{minutes} min {seconds:.1f} sec"
    
    full_response += f"\n\n---\n⏱️ **Total processing time: {time_str}**"

    return ChatResponse(
        paraphrased=paraphrased,
        response=full_response,
        execution_time=execution_time
    )

@app.post("/upload")
def upload_files(
    files: Annotated[list[UploadFile], File(description="Multiple datasets to upload")], 
    agent: AgentDep
) -> dict:
    if not hasattr(agent, 'uploaded_files'):
        agent.uploaded_files = {}
        
    uploaded_info = {}
    for file in files:
        suffix = f".{file.filename.split('.')[-1]}" if '.' in file.filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = file.file.read()
            tmp.write(content)
            
        agent.uploaded_files[file.filename] = tmp.name
        uploaded_info[file.filename] = {
            "path": tmp.name,
            "size": len(content)
        }
            
    return {"message": f"Successfully uploaded {len(files)} files", "files": uploaded_info}

@app.get("/history")
def get_history(agent: AgentDep) -> dict:
    return {"history": agent._show_history()}

@app.post("/clear")
def clear_chat(agent: AgentDep) -> dict:
    agent.conversation = []
    return {"message": "Chat history cleared"}

@app.get("/download/model")
def download_model(agent: AgentDep) -> FileResponse:
    if os.path.exists(agent.champion_model_path):
        return FileResponse(
            path=agent.champion_model_path,
            filename="best_model.joblib",
            media_type="application/octet-stream"
        )
    raise HTTPException(status_code=404, detail="Model not found")

@app.get("/download/predictions")
def download_predictions(agent: AgentDep) -> FileResponse:
    # Get directory of champion model
    model_dir = os.path.dirname(agent.champion_model_path)
    predictions_path = os.path.join(model_dir, "predictions.csv")
    
    if os.path.exists(predictions_path):
        return FileResponse(
            path=predictions_path,
            filename="predictions.csv",
            media_type="text/csv"
        )
    raise HTTPException(status_code=404, detail="Predictions not found")

# Cleanup temp files on exit
import atexit

def cleanup_temp_files():
    # Clean up output files (model and predictions)
    temp_dir = tempfile.gettempdir()
    output_files_to_clean = [
        os.path.join(temp_dir, "best_model.joblib"),
        os.path.join(temp_dir, "predictions.csv")
    ]
    
    for file_path in output_files_to_clean:
        if os.path.exists(file_path):
            try:
                os.unlink(file_path)
                print(f"Cleaned up: {file_path}")
            except Exception as e:
                print(f"Could not delete {file_path}: {e}")

atexit.register(cleanup_temp_files)