import streamlit as st
import base64
import json
import openai

# ---------------- SETUP OPENAI ----------------
# Replace with your actual OpenAI API key
openai.api_key = "sk-"

MODEL_NAME = "gpt-4o-mini"  # Ensure this model name is correct and accessible

# ---------------- UTILS ----------------
def encode_image(image_bytes: bytes) -> str:
    """
    Convert raw bytes of an image into a base64-encoded string.
    """
    return base64.b64encode(image_bytes).decode("utf-8")

def get_chat_history(chat_id: str) -> list:
    """
    Retrieve the conversation history for a given chat_id from session state.
    If it doesn't exist, create an empty list.
    """
    if "chats" not in st.session_state:
        st.session_state["chats"] = {}
    
    if chat_id not in st.session_state["chats"]:
        st.session_state["chats"][chat_id] = []
    
    return st.session_state["chats"][chat_id]

def add_message_to_chat_history(chat_id: str, role: str, content):
    """
    Append a message dict (role='user'/'assistant', content=any) to that chat's history.
    """
    st.session_state["chats"][chat_id].append({"role": role, "content": content})

def build_openai_messages(chat_history: list) -> list:
    """
    Convert stored chat history into the 'messages' format expected by OpenAI's ChatCompletion API.
    
    User messages can include both text and images.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a dermatologist AI. Analyze the user's images and questions. "
                "Provide a reasoned response and ask relevant follow-up questions. "
                "Do not provide medical diagnosis, but guide the user responsibly."
            )
        }
    ]
    
    for msg in chat_history:
        role = msg["role"]
        content = msg["content"]
        
        if isinstance(content, list):
            # Convert list of content blocks to JSON string
            str_content = json.dumps(content)
        else:
            # plain string
            str_content = str(content)
        
        messages.append({"role": role, "content": str_content})
    
    return messages

# ---------------- MAIN APP ----------------
def main():
    st.set_page_config(page_title="Dermatology Chatbot", layout="wide")
    st.title("Dermatology Chatbot (Image-based)")

    # ---------------- SIDEBAR ----------------
    st.sidebar.title("Chat Sessions")

    # Initialize available chats
    if "available_chats" not in st.session_state:
        st.session_state["available_chats"] = ["Chat_1"]

    # Select existing chat
    selected_chat_id = st.sidebar.selectbox(
        "Choose a chat session:",
        options=st.session_state["available_chats"],
    )

    # Button to create a new chat
    if st.sidebar.button("New Chat"):
        new_id = f"Chat_{len(st.session_state['available_chats']) + 1}"
        st.session_state["available_chats"].append(new_id)
        selected_chat_id = new_id  # Switch to the new chat immediately

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Current Chat:** {selected_chat_id}")

    # ---------------- CONVERSATION HISTORY ----------------
    chat_history = get_chat_history(selected_chat_id)

    if chat_history:
        st.markdown("### Conversation so far:")
        for idx, msg in enumerate(chat_history):
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                # Could be text or list of text+image
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            user_text = item.get("text", "")
                            st.markdown(f"**You:** {user_text}")
                        elif item.get("type") == "image_url":
                            data_url = item.get("image_url", {}).get("url", "")
                            if data_url.startswith("data:image"):
                                try:
                                    # Split the data URL to get MIME type and base64 string
                                    mime_type, b64_str = data_url.split(";base64,")
                                    image_data = base64.b64decode(b64_str)
                                    st.image(image_data, caption="Your Uploaded Image", use_container_width=True)
                                except Exception as e:
                                    st.error(f"Image decode error: {e}")
                else:
                    st.markdown(f"**You:** {content}")

            elif role == "assistant":
                st.markdown(f"**Assistant:** {content}")

            elif role == "system":
                # Optionally display system messages
                pass

        st.markdown("---")

    # ---------------- INPUT SECTION ----------------
    # Initialize session state for uploaded image
    if "uploaded_image_bytes" not in st.session_state:
        st.session_state["uploaded_image_bytes"] = None
    if "uploaded_image_mime" not in st.session_state:
        st.session_state["uploaded_image_mime"] = "image/jpeg"  # default MIME type

    uploaded_image = st.file_uploader(
        "Upload an image (jpg, jpeg, or png) of a skin disease:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )

    # Display the uploaded image immediately below uploader
    if uploaded_image:
        image_bytes = uploaded_image.read()
        st.session_state["uploaded_image_bytes"] = image_bytes
        st.session_state["uploaded_image_mime"] = uploaded_image.type  # e.g., 'image/jpeg', 'image/png'

        try:
            st.image(image_bytes, caption="Uploaded Image Preview", use_container_width=True)
        except Exception as e:
            st.error(f"Image display error: {e}")

    # Question input below the image
    user_input = st.text_area("Your question or message to the dermatologist AI:", height=80)

    # ---------------- SEND BUTTON ----------------
    if st.button("Send"):
        # Validate input
        if not user_input.strip() and not st.session_state["uploaded_image_bytes"]:
            st.warning("Please type a message or upload an image.")
            st.stop()

        # Build user content
        user_message_content = []
        if user_input.strip():
            user_message_content.append({
                "type": "text",
                "text": user_input.strip()
            })
        if st.session_state["uploaded_image_bytes"]:
            image_bytes = st.session_state["uploaded_image_bytes"]
            mime_type = st.session_state["uploaded_image_mime"]
            base64_str = encode_image(image_bytes)
            user_message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_str}"}
            })

        # Add user's message to the chat history
        add_message_to_chat_history(selected_chat_id, "user", user_message_content)

        # Clear the uploaded image after sending to prevent re-sending the same image
        st.session_state["uploaded_image_bytes"] = None
        st.session_state["uploaded_image_mime"] = "image/jpeg"  # reset to default

        # Build messages for OpenAI
        messages_for_api = build_openai_messages(get_chat_history(selected_chat_id))

        # Debug: Uncomment to see the messages sent to OpenAI
        # st.write("Messages sent to OpenAI:", messages_for_api)

        # Call OpenAI's ChatCompletion API
        with st.spinner("Analyzing..."):
            try:
                response = openai.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages_for_api,
                    max_tokens=300,
                    temperature=0.1,
                )
            except Exception as e:
                st.error(f"API Error: {e}")
                st.stop()

        # Extract assistant's reply
        assistant_reply = response.choices[0].message.content if response.choices else "No response."

        # Add assistant's reply to chat history
        add_message_to_chat_history(selected_chat_id, "assistant", assistant_reply)

        st.success("AI responded! Check above for the new message and image.")

    # Streamlit automatically reruns after any widget interaction,
    # so the updated conversation will be shown at the top.

if __name__ == "__main__":
    main()
