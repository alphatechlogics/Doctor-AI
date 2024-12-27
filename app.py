import streamlit as st
import openai
import base64
from dotenv import load_dotenv
import os

# ---------------- LOAD ENVIRONMENT VARIABLES ----------------
load_dotenv()  # Load variables from .env file

# ---------------- SETUP OPENAI ----------------
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    st.error("OpenAI API key not found. Please set it in the .env file.")
    st.stop()

MODEL_NAME = "gpt-4o-mini"  # Replace with your actual model name

# ---------------- UTILS ----------------
def encode_image(image):
    """Encode the uploaded image to Base64."""
    return base64.b64encode(image).decode("utf-8")

# ---------------- MAIN APP ----------------
def main():
    st.set_page_config(page_title="Doctor AI", layout="wide")
    st.title("Doctor AI")

    # Initialize session state
    if "analysis_complete" not in st.session_state:
        st.session_state["analysis_complete"] = False
    if "static_analysis" not in st.session_state:
        st.session_state["static_analysis"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Image upload
    uploaded_image = st.file_uploader(
        "Upload an image (jpg, jpeg, or png) of a skin condition:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    # Display uploaded image and perform analysis
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", width=300)

        # Perform static analysis if not already done
        if not st.session_state["analysis_complete"]:
            with st.spinner("Analyzing the image..."):
                try:
                    # Encode the image to Base64
                    base64_image = encode_image(uploaded_image.read())

                    # Send request to OpenAI
                    response = openai.chat.completions.create(
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
                    static_analysis = response.choices[0].message.content.strip()
                    st.session_state["static_analysis"] = static_analysis
                    st.session_state["analysis_complete"] = True
                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": static_analysis}
                    )
                except Exception as e:
                    st.error(f"Error analyzing image: {e}")

    # Display static analysis below the image
    if st.session_state["static_analysis"]:
        st.markdown("### Analysis:")
        st.markdown(st.session_state["static_analysis"])

    # Follow-up question input
    user_input = st.text_area(
        "Your question or message to the Doctor AI:",
        height=80,
        disabled=not st.session_state["analysis_complete"],
    )

    if st.button("Send") and st.session_state["analysis_complete"]:
        if user_input.strip():
            # Add user query to chat history
            st.session_state["chat_history"].append({"role": "user", "content": user_input.strip()})

            # Send follow-up query to OpenAI with context
            try:
                with st.spinner("Processing your query..."):
                    response = openai.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": "You are a dermatologist AI. Continue assisting the user based on the previous image analysis and their queries."},
                        ] + st.session_state["chat_history"],  # Include full chat history
                        max_tokens=500,
                        temperature=0.2,
                    )
                    reply = response.choices[0].message.content.strip()

                    # Add AI reply to chat history
                    st.session_state["chat_history"].append({"role": "assistant", "content": reply})

                    st.markdown("### Response:")
                    st.markdown(reply)
            except Exception as e:
                st.error(f"Error processing query: {e}")

if __name__ == "__main__":
    main()
