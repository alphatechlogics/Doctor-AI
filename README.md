# 🩺 Doctor AI

Doctor AI is an intelligent tool for diagnosing skin conditions from images and providing follow-up recommendations through a conversational interface. This project includes a **Streamlit** app for end-user interaction and a **FastAPI** backend for programmatic access.

---

## ✨ Features

- 🖼️ **Image-based Diagnosis**: Upload an image of a skin condition to receive an AI-powered diagnosis and treatment suggestions.
- 💬 **Interactive Chat**: Ask follow-up questions based on the diagnosis or previous chat history.
- 🖥️ **Streamlit Frontend**: A user-friendly interface for image uploads and interactive conversations.
- 🌐 **FastAPI Backend**: A REST API to integrate diagnosis and chat functionality into other applications.

---

## 🛠️ Technologies

- 🐍 **Python**
- 🌈 **Streamlit**: Frontend interface.
- ⚡ **FastAPI**: Backend API service.
- 🤖 **OpenAI GPT-4**: Core AI model for image analysis and conversational replies.

---

## ✅ Prerequisites

1. Python 3.7 or higher.
2. An OpenAI API key.

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/doctor-ai.git
cd doctor-ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Secrets

- For Streamlit:

Create a secrets.toml file in the .streamlit directory:

```bash
OPENAI_API_KEY = "your-openai-api-key"
```

- For FastAPI:
  Set up the OPENAI_API_KEY using Streamlit secrets (already integrated into the code).

## 📖 Usage

### 1. Run the Streamlit App

- Start the app:

```bash
streamlit run app.py
```

### 2. Access the app at http://localhost:8501.

- Run the FastAPI Backend Start the API server:

```bash
uvicorn api:app --reload
```

- Access the API documentation at http://127.0.0.1:8000/docs.

## 🔗 API Endpoints

### 1. /analyze_and_chat (POST)

- Description: Diagnose a skin condition and/or respond to a follow-up query.
- Parameters:
  - 📂 file: Image file for diagnosis (optional).
  - ❓ user_query: Follow-up question (optional).
  - 📝 chat_history: Previous conversation history as JSON string (optional).
- Response:

```bash
{
  "diagnosis": "string",
  "reply": "string",
  "chat_history": [{"role": "user/assistant", "content": "string"}]
}
```

- Example Request:

```bash
curl -X POST \
  'http://127.0.0.1:8000/analyze_and_chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path_to_image.jpg' \
  -F 'user_query=What should I do now?' \
  -F 'chat_history=[{"role": "assistant", "content": "Your diagnosis is..."}, {"role": "user", "content": "What does this mean?"}]'
```
