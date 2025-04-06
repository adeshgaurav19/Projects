from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model
model_name = "llama-3.2-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize FastAPI
app = FastAPI()

# Define request schema
class UserInput(BaseModel):
    text: str

# Endpoint for grammar correction
@app.post("/correct_grammar/")
async def correct_grammar(user_input: UserInput):
    prompt = f"Correct this French sentence: {user_input.text}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    corrected_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"corrected_sentence": corrected_sentence}

# Endpoint for vocabulary suggestion
@app.post("/vocabulary/")
async def suggest_vocabulary(user_input: UserInput):
    prompt = f"Give me synonyms, meaning, and an example sentence for the French word: {user_input.text}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

# Define a conversation history
conversation_history = []


# Endpoint for conversational responses
@app.post("/conversation/")
async def conversation(user_input: UserInput):
    global conversation_history
    
    # Add user input to conversation history
    conversation_history.append(f"User: {user_input.text}")
    
    # Create a conversational prompt
    context = "\n".join(conversation_history)
    prompt = f"{context}\nAssistant:"
    
    # Generate a response
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract and clean the assistant's response
    assistant_response = response.split("Assistant:")[-1].strip()
    
    # Add assistant response to conversation history
    conversation_history.append(f"Assistant: {assistant_response}")
    
    return {"response": assistant_response}