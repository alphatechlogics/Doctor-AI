from fastapi import FastAPI, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import openai
import json
from utils import encode_image, process_image_analysis

# ---------------- SETUP OPENAI ----------------
from dotenv import load_dotenv
import os
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set it in .env file.")

# ---------------- FASTAPI ----------------
app = FastAPI(title="Doctor AI API")

class AnalyzeAndChatRequest(BaseModel):
    user_query: Optional[str] = None
    chat_history: Optional[List[dict]] = []

@app.post("/analyze_and_chat")
async def analyze_and_chat(
    file: Optional[UploadFile] = None,
    user_query: Optional[str] = Form(None),
    chat_history: Optional[str] = Form("[]"),
):
    try:
        parsed_chat_history = json.loads(chat_history)
        chat_history = parsed_chat_history if isinstance(parsed_chat_history, list) else []

        diagnosis = None
        if file:
            base64_image = encode_image(await file.read())
            diagnosis = process_image_analysis(base64_image)
            chat_history.append({"role": "assistant", "content": diagnosis})

        reply = None
        if user_query:
            chat_history.append({"role": "user", "content": user_query})
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=chat_history,
                max_tokens=500,
                temperature=0.2,
            )
            reply = response.choices[0].message.content.strip()

        return {"diagnosis": diagnosis, "reply": reply, "chat_history": chat_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
