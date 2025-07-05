import streamlit as st
import ollama
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.schema import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import speech_recognition as sr
from gtts import gTTS
import tempfile
import base64
import wikipediaapi
import random

# Database Path
VECTOR_DB_PATH = "vector_db"

import wikipediaapi

# Define a proper User-Agent
wiki = wikipediaapi.Wikipedia(
    user_agent="AdeshGaurav-FrenchLearningBot/1.0 (contact: your-email@example.com)",
    language="fr"
)

# Example: Retrieve a French Wikipedia page
page = wiki.page("Grammaire_française")
print(page.summary)  # Print summary of "French Grammar"


# Load Learning Resources
def load_documents():
    docs = [
        Document(page_content="Bonjour! means Hello in French."),
        Document(page_content="Merci means Thank you in French."),
        Document(page_content="Comment ça va? means How are you?"),
        Document(page_content="Je m'appelle means My name is."),
        Document(page_content="Parlez-vous français? means Do you speak French?"),
        Document(page_content="Où est la gare? means Where is the train station?"),
    ]
    return docs

def create_vector_db():
    embedding_model = OllamaEmbeddings(model="tinyllama")
    docs = load_documents()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    texts = text_splitter.split_documents(docs)
    vector_db = Chroma.from_documents(texts, embedding_model, persist_directory=VECTOR_DB_PATH)
    vector_db.persist()
    return vector_db

# Initialize Vector DB
if not os.path.exists(VECTOR_DB_PATH):
    vector_db = create_vector_db()
else:
    vector_db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=OllamaEmbeddings(model="tinyllama"))

# Get Wikipedia Context
def get_wikipedia_page(topic):
    page = wiki_wiki.page(topic)
    return page.summary if page.exists() else "Information not found."

# Convert Text to Speech
def speak(text):
    tts = gTTS(text=text, lang="fr")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)

    with open(temp_file.name, "rb") as audio_file:
        audio_bytes = audio_file.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()

    os.remove(temp_file.name)
    return f'<audio controls><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>'

# Streamlit UI
st.title("🇫🇷 Conversational French Tutor - Learn French!")
st.write("Chat, listen, and learn French interactively!")

# Initialize Chat History & Progress
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_progress" not in st.session_state:
    st.session_state.user_progress = {"vocab": set(), "grammar": 0}

# Display Chat History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Speech Recognition
recognizer = sr.Recognizer()

st.write("🎙️ Click to speak (experimental)")
if st.button("Start Recording"):
    with sr.Microphone() as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        user_query = recognizer.recognize_google(audio, language="fr-FR")
        st.write(f"🗣️ You said: {user_query}")
    except sr.UnknownValueError:
        st.write("Sorry, I couldn't understand.")
    except sr.RequestError:
        st.write("Speech recognition service error.")

# User Input
if user_query := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Retrieve context from Vector DB and Wikipedia
    docs = vector_db.similarity_search(user_query, k=2)
    wiki_context = get_wikipedia_page(user_query)
    context = "\n".join([doc.page_content for doc in docs]) + "\n" + wiki_context

    # Conversational Memory
    history = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages[-5:]]
    history.append({"role": "user", "content": f"Context: {context}\nUser: {user_query}\tinyllama:"})

    # Generate Response
    response = ollama.chat(model="tinyllama", messages=history)
    chatbot_response = response['message']['content']

    # Track Vocabulary & Grammar Improvement
    if "new vocabulary" in chatbot_response:
        st.session_state.user_progress["vocab"].add(user_query)
    if "grammar correction" in chatbot_response:
        st.session_state.user_progress["grammar"] += 1

    # Append Response to Chat History
    st.session_state.messages.append({"role": "assistant", "content": chatbot_response})

    # Display Chatbot Response
    st.chat_message("assistant").write(chatbot_response)
    st.markdown(speak(chatbot_response), unsafe_allow_html=True)

# Sidebar - Learning Progress
st.sidebar.title("📚 Learning Progress")
st.sidebar.write(f"🔹 Vocabulary Learned: {len(st.session_state.user_progress['vocab'])}")
st.sidebar.write(f"🔹 Grammar Score: {st.session_state.user_progress['grammar']}")

# Flashcards for Vocabulary Practice
st.sidebar.title("🃏 Vocabulary Flashcards")
vocab_flashcards = {
    "chien": "dog",
    "chat": "cat",
    "maison": "house",
    "voiture": "car",
    "pomme": "apple",
}
word, meaning = random.choice(list(vocab_flashcards.items()))
st.sidebar.write(f"🔹 **{word}** → ???")
if st.sidebar.button("Show Answer"):
    st.sidebar.write(f"✔️ **{meaning}**")

# Grammar Correction
grammar_prompt = f"Correct this sentence in French:\n\n{user_query}"
grammar_correction = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": grammar_prompt}])
st.write("✅ **Grammar Suggestion:**", grammar_correction["message"]["content"])
