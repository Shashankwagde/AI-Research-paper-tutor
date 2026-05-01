import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)


def generate_response(prompt, max_tokens=300):
    """
    Generic response generator using Gemini 2.5 Flash (free model).
    max_tokens can be adjusted for longer summaries.
    """

    # Set up the model - using Gemini 2.5 Flash (free model)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=(
            "You are an academic AI research assistant.\n"
            "Provide clear, structured, comprehensive explanations.\n"
            "Do not hallucinate information.\n"
            "Base responses only on provided context."
        )
    )

    # Generate content
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=max_tokens
            )
        )
        
        # Check if response has valid content
        if response.candidates:
            return response.text
        else:
            return "Error: No response generated. The request may have been blocked by safety filters."
            
    except Exception as e:
        return f"Error: {str(e)}"
