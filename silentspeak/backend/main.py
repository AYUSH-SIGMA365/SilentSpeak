from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import shutil

from silentspeak.core.inference import LipReader

app = FastAPI(
    title="SilentSpeak API",
    description="Backend API for lip-reading visual speech recognition",
    version="1.0.0"
)

# Allow CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the LipReader singleton globally so it loads once at startup
try:
    lip_reader = LipReader()
except Exception as e:
    print(f"Failed to initialize LipReader on startup: {e}")
    lip_reader = None

class TranscribeResponse(BaseModel):
    transcript: str
    error: str | None = None

@app.get("/")
def read_root():
    return {"status": "ok", "message": "SilentSpeak API is running"}

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_video(video: UploadFile = File(...)):
    if not video.filename.endswith((".mp4", ".mpg", ".avi")):
        raise HTTPException(status_code=400, detail="Invalid video format")

    if lip_reader is None:
        raise HTTPException(status_code=500, detail="Model is not initialized")

    # Create a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        try:
            # Copy the uploaded file contents to the temp file
            shutil.copyfileobj(video.file, temp_file)
            temp_path = temp_file.name
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")
        finally:
            video.file.close()

    try:
        # Run inference
        transcript = lip_reader.transcribe(temp_path)
        return TranscribeResponse(transcript=transcript)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
