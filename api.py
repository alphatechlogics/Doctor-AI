from fastapi import FastAPI, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import openai
import base64
import json
import streamlit as st

# Setup OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]  # Use Streamlit secrets
# openai.api_key = os.getenv("OPENAI_API_KEY")  # Commented out

if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set it in Streamlit secrets.")

# Model Name
MODEL_NAME = "gpt-4o-mini"

# Initialize FastAPI
app = FastAPI(title="Doctor AI API")

# Utility function to encode image to Base64
def encode_image(image_data: bytes) -> str:
    """Encode the image to Base64 string."""
    return base64.b64encode(image_data).decode("utf-8")

# Request model for AnalyzeAndChat
class AnalyzeAndChatRequest(BaseModel):
    user_query: Optional[str] = None
    chat_history: Optional[List[dict]] = []


@app.post("/analyze_and_chat")
async def analyze_and_chat(
    file: Optional[UploadFile] = None,
    user_query: Optional[str] = Form(None),
    chat_history: Optional[str] = Form("[]"),
):
    """
    Diagnose a skin condition from an uploaded image and handle follow-up queries.
    - If an image is provided, perform diagnosis.
    - If a user query is provided, maintain the conversation using chat history.
    """
    try:
        # Parse chat history
        try:
            parsed_chat_history = json.loads(chat_history)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid chat history format")

        chat_history = parsed_chat_history if isinstance(parsed_chat_history, list) else []

        # Initialize diagnosis variable
        diagnosis = None

        # Perform diagnosis if an image is uploaded
        if file:
            image_data = await file.read()
            base64_image = encode_image(image_data)

            # Call OpenAI for analysis
            diagnosis_response = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """You are a professional dermatologist AI trained to assist users by analyzing images of skin conditions. When analyzing an image, provide a realistic and thorough response as follows:
1. **Diagnosis**: Identify the most likely skin condition shown in the image. Be descriptive and include features like lesions, discoloration, or patterns that are visible. If there is uncertainty, provide your best assessment based on your training.
2. **Danger Level**: Rate the condition on a scale of 1 to 5 (1 being not dangerous and 5 being potentially serious). Use this scale to help users understand the urgency of seeking professional care:
   - **1**: Mild, non-serious conditions such as dry skin, minor rashes, or acne.
   - **2**: Moderate conditions like mild eczema or rosacea that may require simple treatments.
   - **3**: Conditions like infected acne or moderate dermatitis that may need medical attention if untreated.
   - **4**: Serious conditions like severe infections, deep ulcers, or potentially cancerous lesions.
   - **5**: Emergency conditions such as necrotizing fasciitis, advanced skin cancer, or severe burns that require immediate medical intervention.

3. **Treatment Suggestions**: Provide tailored suggestions for topical treatments, oral medications, or other relevant advice. Include over-the-counter and prescription options, lifestyle changes, and preventative measures. If there is any uncertainty, suggest the user consult a dermatologist for further evaluation.

4. **Disclaimer**: End your response with a clear disclaimer stating that your analysis is based solely on the image provided and does not replace professional medical advice. Encourage users to consult a licensed dermatologist for confirmation and a personalized treatment plan.

Avoid saying "I cannot analyze this image" or giving generic responses. Act as a professional dermatologist would, providing meaningful guidance based on the image.""",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                max_tokens=500,
                temperature=0.2,
            )
            diagnosis = diagnosis_response.choices[0].message.content.strip()
            # Add the diagnosis to the chat history
            chat_history.append({"role": "assistant", "content": diagnosis})

        # Handle user query if provided
        reply = None
        if user_query:
            chat_history.append({"role": "user", "content": user_query})

            # Call OpenAI for a response
            chat_response = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=chat_history,
                max_tokens=500,
                temperature=0.2,
            )
            reply = chat_response.choices[0].message.content.strip()

        return {
            "diagnosis": diagnosis,
            "reply": reply,
            "chat_history": chat_history,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
