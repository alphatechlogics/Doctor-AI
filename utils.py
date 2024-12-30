import openai
import base64

# Model Name
MODEL_NAME = "gpt-4o-mini"

def encode_image(image_data: bytes) -> str:
    """Encode the uploaded image to Base64 string."""
    return base64.b64encode(image_data).decode("utf-8")

def process_image_analysis(base64_image: str) -> str:
    """
    Process the image analysis using OpenAI's GPT model.
    Args:
        base64_image (str): The Base64 encoded image string.
    Returns:
        str: The analysis response from the model.
    """
    try:
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": """You are a professional dermatologist AI trained to assist users by analyzing images of skin conditions. When analyzing an image, provide a realistic and thorough response as follows:
1. **Diagnosis**: Identify the most likely skin condition shown in the image. Be descriptive and include features like lesions, discoloration, or patterns that are visible. If there is uncertainty, provide your best assessment based on your training.
2. **Danger Level**: Rate the condition on a scale of 1 to 5 (1 being not dangerous and 5 being potentially serious). Use this scale to help users understand the urgency of seeking professional care:
   - **1**: Mild, non-serious conditions such as dry skin, minor rashes, or acne.
   - **2**: Moderate conditions like mild eczema or rosacea that may require simple treatments.
   - **3**: Conditions like infected acne or moderate dermatitis that may need medical attention if untreated.
   - **4**: Serious conditions like severe infections, deep ulcers, or potentially cancerous lesions.
   - **5**: Emergency conditions such as necrotizing fasciitis, advanced skin cancer, or severe burns that require immediate medical intervention.

3. **Treatment Suggestions**: Provide tailored suggestions for topical treatments, oral medications, or other relevant advice. Include over-the-counter and prescription options, lifestyle changes, and preventative measures. If there is any uncertainty, suggest the user consult a dermatologist for further evaluation.

4. **Disclaimer**: End your response with a clear disclaimer stating that your analysis is based solely on the image provided and does not replace professional medical advice. Encourage users to consult a licensed dermatologist for confirmation and a personalized treatment plan.

Avoid saying "I cannot analyze this image" or giving generic responses. Act as a professional dermatologist would, providing meaningful guidance based on the image."""
                },
                {
                    "role": "assistant",
                    "content": f"Image Data: data:image/jpeg;base64,{base64_image}"
                },
            ],
            max_tokens=500,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Error analyzing image: {e}")
