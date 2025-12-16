from fastapi import FastAPI, UploadFile, File
import whisper
import tempfile

app = FastAPI()

# Load Whisper model
model = whisper.load_model("small")

@app.post("/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    # Save audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    # Transcribe
    result = model.transcribe(tmp_path, fp16=False)

    return {"text": result["text"]}
