import streamlit as st
import requests
import feedparser
import os
import ollama
import chromadb
from chromadb.utils import embedding_functions
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://adesh:abcd1234@localhost/podcast_db")
engine = create_engine(DATABASE_URL, echo=False)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# Define Episode Model
class Episode(Base):
    __tablename__ = 'episodes'
    id = Column(Integer, primary_key=True)
    podcast_name = Column(String, nullable=False)
    episode_name = Column(String, nullable=False)
    url = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_function = embedding_functions.DefaultEmbeddingFunction()
podcast_collection = chroma_client.get_or_create_collection("podcast_summaries", embedding_function=embedding_function)

# Streamlit UI
st.set_page_config(page_title="Podcast AI", layout="wide")

st.markdown("""
    <style>
        .header { text-align: center; font-size: 40px; color: #0077b6; margin-bottom: 20px; }
        .sub-header { color: #023e8a; font-size: 22px; }
        .episode-card { background-color: #ffffff; padding: 15px; margin: 10px 0; border-radius: 10px; 
                        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="header">🎙️ Podcast Summarizer & AI Insights</p>', unsafe_allow_html=True)

podcast_name = st.text_input("Enter Podcast Name", placeholder="Search podcast...")

def get_podcast_feed_url(podcast_name):
    url = f"https://itunes.apple.com/search?term={podcast_name}&entity=podcast"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["results"][0]["feedUrl"] if data.get("resultCount", 0) > 0 else None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching podcast feed: {e}")
    return None

def get_episodes_from_feed(feed_url, limit=5):
    feed = feedparser.parse(feed_url)
    return feed.entries[:limit] if feed.entries else []

def get_audio_url(episode):
    for link in episode.get("links", []):
        if "audio" in link.get("type", ""):
            return link.get("href")
    return None

def lightweight_transcribe(audio_url):
    if not audio_url:
        return "No audio URL available for transcription."
    
    try:
        response = ollama.chat(model="tinyllama", messages=[
            {"role": "system", "content": "Transcribe the given podcast audio."},
            {"role": "user", "content": f"Audio URL: {audio_url}"}
        ])
        
        print("Transcription API Response:", response)
        
        if isinstance(response, dict):
            if "message" in response and "content" in response["message"]:
                return response["message"]["content"]
            elif "response" in response:
                return response["response"]
        
        return "Transcription not available."
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return "Transcription not available."

def generate_summary(text):
    if not text:
        return "No text available for summarization."
    
    try:
        response = ollama.chat(model="tinyllama", messages=[
            {"role": "system", "content": "Summarize the given text."},
            {"role": "user", "content": text}
        ])
        
        print("Summary API Response:", response)
        
        if isinstance(response, dict):
            if "message" in response and "content" in response["message"]:
                return response["message"]["content"]
            elif "response" in response:
                return response["response"]
        
        return "Summary not available."
    except Exception as e:
        print(f"Summary error: {str(e)}")
        return "Summary not available."

def display_episode_summary(episode, summary):
    if isinstance(summary, list):  # Handle list case
        summary = " ".join(summary)
    
    st.markdown(f"""
        <div class="episode-card">
            <h3>{episode.title}</h3>
            <p><strong>Summary:</strong> {summary}</p>
        </div>
    """, unsafe_allow_html=True)



def store_in_chroma(episode_name, summary):
    existing_ids = podcast_collection.get(ids=[episode_name])["ids"]
    if not existing_ids:
        podcast_collection.add(documents=[summary.strip()], ids=[episode_name])
    else:
        podcast_collection.update(ids=[episode_name], documents=[summary.strip()])


def query_summary(query):
    results = podcast_collection.query(query_texts=[query], n_results=1)
    if results["documents"]:
        summaries = results["documents"][0]  # Extract first document
        if isinstance(summaries, list) and summaries:
            return summaries[0]  # Ensure only the first result is returned as string
        return summaries  # In case it's already a string
    return "No relevant summary found."


# Main app logic
if podcast_name:
    feed_url = get_podcast_feed_url(podcast_name)
    if feed_url:
        episodes = get_episodes_from_feed(feed_url)
        st.markdown(f'## {podcast_name}')
        
        for episode in episodes:
            with st.expander(f"🎧 {episode.title}"):
                session = Session()
                existing_episode = session.query(Episode).filter(
                    Episode.episode_name == episode.title
                ).first()
                
                if not existing_episode:
                    audio_url = get_audio_url(episode)
                    transcription = lightweight_transcribe(audio_url)
                    
                    if transcription != "Transcription not available.":
                        summary = generate_summary(transcription)
                        store_in_chroma(episode.title, summary)
                        
                        new_episode = Episode(
                            podcast_name=podcast_name,
                            episode_name=episode.title,
                            url=audio_url
                        )
                        session.add(new_episode)
                        session.commit()
                        
                        display_episode_summary(episode, summary)
                    else:
                        st.error("Could not process this episode.")
                else:
                    summary = query_summary(episode.title)
                    display_episode_summary(episode, summary)
                    
                session.close()

# Chatbot section
st.markdown("### 💬 Ask About a Podcast Episode")
user_query = st.text_input("Ask a question:", key="user_query")

if user_query:
    with st.spinner("Finding answer..."):
        response = query_summary(user_query)
        st.markdown(f"""
            <div style='background-color: #ffffff; padding: 15px; 
                border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);'>
                {response}
            </div>
        """, unsafe_allow_html=True)