import os
import shutil
import tempfile
import threading
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool

from analyzer_onnx import begin_session, end_session, SessionHandler, analyze_file
from voxalyzer import MODEL_VERSION

app = FastAPI()

# Lock to ensure thread safety for global models in main.py
analysis_lock = threading.Lock()

session: SessionHandler = None

def _analyze_safe(file_path: str, cleanup_file:bool = False):
    """
    Helper function to run the analysis safely with a lock and cleanup.
    """
    try:
        with analysis_lock:
            return analyze_file(file_path, session_handler=session, force=True)
    finally:
        if cleanup_file and os.path.exists(file_path):
            os.remove(file_path)

@app.get("/voxalyzer")
def voxalyzer():
    return MODEL_VERSION

@app.post("/analyze")
async def analyze_endpoint(request: Request):
    """
    Unified endpoint to analyze an MP3 file.
    Supports both multipart/form-data (field name 'file') and raw binary body.
    """
    content_type = request.headers.get("content-type", "")
    tmp_path = None
    cleanup_file = False
    try:
        if content_type.startswith("application/json"):
            request_data = await request.json()
            tmp_path = request_data["file"]
        elif content_type.startswith("multipart/form-data"):
            # Handle multipart upload
            form = await request.form()
            file_obj = form.get("file")
            
            if not file_obj:
                return {"error": "Multipart request must contain a 'file' field."}
            
            # Save uploaded file to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                shutil.copyfileobj(file_obj.file, tmp)
                tmp_path = tmp.name
                cleanup_file = True
        else:
            # Handle raw binary upload
            body = await request.body()
            if not body:
                return {"error": "Request body is empty."}
                
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(body)
                tmp_path = tmp.name
                cleanup_file = True

        # Run the blocking analysis in a separate thread
        return await run_in_threadpool(_analyze_safe, tmp_path, cleanup_file=cleanup_file)

    except Exception as e:
        # Cleanup if something failed before _analyze_safe was called
        # or if _analyze_safe failed but somehow didn't clean up (unlikely due to finally)
        if tmp_path and os.path.exists(tmp_path) and cleanup_file:
            os.remove(tmp_path)
        return {"error": str(e)}

def serve(port:int = 8000):
    global session
    import uvicorn

    session = begin_session()
    uvicorn.run(app, host="0.0.0.0", port=port)
    end_session(session)

if __name__ == "__main__":
    serve()
