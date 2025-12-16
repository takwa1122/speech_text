# main.py
from fastapi import FastAPI, UploadFile, File
import whisper
import tempfile

app = FastAPI()

# Global variable to store the model instance
global_model = None

@app.post("/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    global global_model

    # 1. Load model only on the first request
    if global_model is None:
        # You might still need to use "tiny" here
        global_model = whisper.load_model("tiny") 

    # Save audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    # Transcribe
    # Use the cached model instance
    result = global_model.transcribe(tmp_path, fp16=False) 

    return {"text": result["text"]}
