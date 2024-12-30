import streamlit as st
import openai
from dotenv import load_dotenv
import os
from utils import encode_image, process_image_analysis

# ---------------- SET PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND) ----------------
st.set_page_config(page_title="Doctor AI", layout="wide")

# ---------------- LOAD ENVIRONMENT VARIABLES ----------------
load_dotenv()  # Load variables from .env file

# ---------------- SETUP OPENAI ----------------
# Prioritize .env and fallback to Streamlit secrets if .env is not available
openai_api_key = os.getenv("OPENAI_API_KEY")  # Use .env first
if not openai_api_key:
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]  # Use Streamlit secrets if available
    except Exception:
        openai_api_key = None

if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in .env file or Streamlit secrets.")
    st.stop()

openai.api_key = openai_api_key

# ---------------- MAIN APP ----------------
def main():
    st.title("Doctor AI")

    if "analysis_complete" not in st.session_state:
        st.session_state["analysis_complete"] = False
    if "static_analysis" not in st.session_state:
        st.session_state["static_analysis"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    uploaded_image = st.file_uploader(
        "Upload an image (jpg, jpeg, or png):",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", width=300)
        if not st.session_state["analysis_complete"]:
            with st.spinner("Analyzing the image..."):
                try:
                    base64_image = encode_image(uploaded_image.read())
                    analysis = process_image_analysis(base64_image)
                    st.session_state["static_analysis"] = analysis
                    st.session_state["analysis_complete"] = True
                    st.session_state["chat_history"].append({"role": "assistant", "content": analysis})
                except RuntimeError as e:
                    st.error(str(e))

    if st.session_state["static_analysis"]:
        st.markdown("### Analysis:")
        st.markdown(st.session_state["static_analysis"])

    user_input = st.text_area("Your question:", height=80, disabled=not st.session_state["analysis_complete"])
    if st.button("Send") and user_input.strip():
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        try:
            with st.spinner("Processing your query..."):
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state["chat_history"],
                    max_tokens=500,
                    temperature=0.2,
                )
                reply = response.choices[0].message.content.strip()
                st.session_state["chat_history"].append({"role": "assistant", "content": reply})
                st.markdown("### Response:")
                st.markdown(reply)
        except Exception as e:
            st.error(f"Error processing query: {e}")

if __name__ == "__main__":
    main()
