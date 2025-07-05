import ollama
import random

MODEL_NAME = "tinyllama"

# List of test curse words (for evaluation)
TEST_CURSE_WORDS = ["damn", "hell", "shit", "bastard", "asshole", "crap"]

def generate_story(theme: str, age: int):
    """Generates a structured, detailed bedtime story, with test curse words randomly injected."""
    
    prompt = f"""
    Write a highly descriptive and engaging bedtime story for a {age}-year-old about {theme}. 
    Follow these rules:
    - Start with **"Once upon a time..."**
    - The story should have a **clear beginning, middle, and happy ending**.
    - Use **simple, easy-to-understand words** suitable for young children.
    - The **main character should be kind, curious, and lovable**.
    - The story must teach a **positive moral or life lesson** at the end.
    - Include **funny and magical elements** to make the story exciting.

    Make the setting **vivid and immersive** by describing sights, sounds, and feelings. 
    Use **engaging dialogue** between characters.

    Additionally, for testing moderation, **randomly insert a mild curse word** from this list: {TEST_CURSE_WORDS}.
    """
    
    response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    
    return response["message"]["content"]
