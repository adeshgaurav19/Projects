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


# Database setup remains the same
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://adesh:abcd1234@localhost/podcast_db")
engine = create_engine(DATABASE_URL, echo=False)
Session = sessionmaker(bind=engine)
Base = declarative_base()



class Episode(Base):
    __tablename__ = 'episodes'
    id = Column(Integer, primary_key=True)
    podcast_name = Column(String, nullable=False)
    episode_name = Column(String, nullable=False)
    url = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_function = embedding_functions.DefaultEmbeddingFunction()
podcast_collection = chroma_client.get_or_create_collection(
    name="podcast_summaries",
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"}
)


def lightweight_transcribe(audio_url, episode_title):
    """
    Generate a detailed transcription or outline for a podcast episode based on the title.
    The transcription will include key points, quotes, or important topics discussed in the episode.
    """
    try:
        prompt = f"""
        Transcribe the key points, quotes, and important topics discussed in the podcast episode titled "{episode_title}".
        Provide a detailed transcription, including notable moments, characters, and any relevant discussions that could be useful for deeper analysis. 
        Ensure the transcription includes contextual details like tone or major themes.
        """
        
        response = ollama.chat(
            model="tinyllama",
            messages=[
                {
                    "role": "system",
                    "content": "You are transcribing podcast episodes with key points, discussions, and notable quotes. Capture the essence of the discussion in great detail."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Check for valid response and return the transcription
        if isinstance(response, ollama._types.ChatResponse):
            return response.message.content
        elif isinstance(response, dict):
            if "message" in response and "content" in response["message"]:
                return response["message"]["content"]
            else:
                st.warning(f"Unexpected response format: {response.keys()}")
                return None
        else:
            st.warning(f"Unexpected response type: {type(response)}")
            return None
        
    except Exception as e:
        st.error(f"Error generating transcription: {str(e)}")
        return None


def generate_summary(text, episode_title):
    """
    Generate an in-depth summary of the episode that includes context, characters, interactions, and major takeaways.
    """
    if not text:
        return None
    
    try:
        prompt = f"""
        Provide an in-depth summary of the podcast episode titled "{episode_title}". The summary should be detailed, including:
        1. A comprehensive overview of the main themes discussed.
        2. Information on key characters, their role in the episode, and any major interactions or developments.
        3. Key moments or quotes that are significant to the episode's narrative.
        4. Insights or actionable takeaways derived from the discussion.
        5. Any challenges or notable scenarios discussed within the episode.

        Be sure to capture the essence of the conversation in the summary. Include enough detail for a deep understanding of the episode's content.
        """
        
        response = ollama.chat(
            model="tinyllama",
            messages=[
                {
                    "role": "system",
                    "content": "You are generating detailed summaries for podcast episodes. Capture the key insights, interactions, and significant moments in depth."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Validate and process response
        if isinstance(response, ollama._types.ChatResponse):
            summary = response.message.content
            if "[Generated]" in summary or "[Auto-generated" in summary:
                return None
            return summary
        elif isinstance(response, dict):
            if "message" in response and "content" in response["message"]:
                summary = response["message"]["content"]
                if "[Generated]" in summary or "[Auto-generated" in summary:
                    return None
                return summary
            else:
                st.warning(f"Unexpected response format: {response.keys()}")
                return None
        else:
            st.warning(f"Unexpected response type: {type(response)}")
            return None
        
    except Exception as e:
        st.error(f"Error generating detailed summary: {str(e)}")
        return None


    
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


def get_episodes_from_feed(feed_url, limit=1):
    feed = feedparser.parse(feed_url)
    return feed.entries[:limit] if feed.entries else []

def get_audio_url(episode):
    for link in episode.get("links", []):
        if "audio" in link.get("type", ""):
            return link.get("href")
    return None

def display_episode_summary(episode, summary):
    if isinstance(summary, list):
        summary = " ".join(summary)
    
    st.markdown(f"""{episode.title}Summary: {summary}""", unsafe_allow_html=True)

def store_in_chroma(episode_name, summary):
    """
    Store summary in ChromaDB with better error handling
    """
    if not summary or len(summary.strip()) < 10:  # Basic validation
        return False
        
    try:
        # Clean the summary text
        clean_summary = summary.replace("[Generated]", "").replace("[Auto-generated]", "").strip()
        
        # Add to ChromaDB
        podcast_collection.upsert(
            documents=[clean_summary],
            ids=[episode_name],
            metadatas=[{"episode": episode_name}]
        )
        return True
    except Exception as e:
        st.error(f"Error storing in ChromaDB: {str(e)}")
        return False

import re

def query_summary(query, context_data):
    """
    Perform a semantic search on a larger dataset (transcriptions and detailed summaries) to fetch richer, focused responses from podcast content.
    - The system uses keyword matching or semantic search to find the most relevant passages.
    - If no match is found, the closest related data from the context is returned.
    """
    try:
        # Perform semantic search with the query across transcriptions and summaries
        results = podcast_collection.query(
            query_texts=[query],
            n_results=3,  # Return top 3 results for richer context
            include=["documents", "metadatas", "distances"]
        )
        
        # Check if there are results
        if results and "documents" in results and results["documents"]:
            document_responses = []
            for doc, metadata, distance in zip(results["documents"], results["metadatas"], results["distances"]):
                document_text = doc[0]
                document_metadata = metadata[0]
                document_distance = distance[0]
                
                # Include context of the episode and distance for better interpretation
                document_responses.append(
                    f"**Episode '{document_metadata['episode']}':**\n\n{document_text}\n\n(Distance: {document_distance:.4f})"
                )
            
            # Return combined responses
            return "\n\n".join(document_responses)
        
        # Handle case when no relevant documents are found
        else:
            return f"**No exact match found for '{query}'.** Here is a related excerpt:\n\n{context_data[:500]}..."  # Show part of context as fallback

    except Exception as e:
        st.error(f"Error querying podcast data: {str(e)}")
        return "**Sorry, I encountered an error while searching for that information.**"



import re

def find_focused_answer(text, keywords):
    """
    Helper function to find a focused answer within the text based on keywords.
    Extracts relevant sentences or text around keyword matches, enhancing context.
    """
    # Iterate over each keyword to search within the text
    for keyword in keywords:
        pattern = re.compile(r'\b{}\b'.format(re.escape(keyword)), re.IGNORECASE)
        matches = pattern.finditer(text)
        
        # If any matches are found, process them
        for match in matches:
            # Split text into sentences for better context extraction
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # Find the matching sentence and return surrounding context
            start = next((i for i, s in enumerate(sentences) if pattern.search(s)), None)
            if start is not None:
                # Extract context from before and after the match
                context_start = max(0, start - 1)
                context_end = min(len(sentences), start + 2)  # Expand 1 sentence before and after
                
                # Return a snippet with surrounding context
                return ' '.join(sentences[context_start:context_end])
            
            # If not in a sentence, return the word with some surrounding context
            match_start = match.start()
            match_end = match.end()
            return text[max(0, match_start - 50):min(len(text), match_end + 50)]
    
    return "**No relevant answer found based on keywords.**"


def process_episode(episode, podcast_name):
    """
    Process a single episode with better error handling
    """
    try:
        # First check if we already have this episode
        session = Session()
        existing_episode = session.query(Episode).filter(
            Episode.episode_name == episode.title
        ).first()
        
        if existing_episode and existing_episode.summary:
            session.close()
            return existing_episode.summary
            
        # Generate new content
        transcription = lightweight_transcribe(None, episode.title)
        if not transcription:
            session.close()
            return None
            
        summary = generate_summary(transcription, episode.title)
        if not summary:
            session.close()
            return None
            
        # Store in database
        new_episode = Episode(
            podcast_name=podcast_name,
            episode_name=episode.title,
            url=get_audio_url(episode),
            summary=summary
        )
        session.add(new_episode)
        session.commit()
        
        # Store in ChromaDB
        store_success = store_in_chroma(episode.title, summary)
        if not store_success:
            st.warning("Warning: Could not store summary in ChromaDB")
            
        session.close()
        return summary
        
    except Exception as e:
        st.error(f"Error processing episode: {str(e)}")
        if 'session' in locals():
            session.close()
        return None

# Streamlit UI
st.set_page_config(page_title="Podcast AI", layout="wide")

st.markdown("""""", unsafe_allow_html=True)

st.markdown('🎙️ Podcast Summarizer & AI Insights', unsafe_allow_html=True)

st.sidebar.markdown("### Debug Options")
if st.sidebar.checkbox("Show Debug Info"):
    st.sidebar.write("Last processed episode:", st.session_state.get('last_processed_episode', 'None'))
    st.sidebar.write("ChromaDB collection size:", len(podcast_collection.get()["ids"]))

podcast_name = st.text_input("Enter Podcast Name", placeholder="Search podcast...")

# Main app logic with improved error handling
if podcast_name:
    with st.spinner("Searching for podcast..."):
        feed_url = get_podcast_feed_url(podcast_name)
        if feed_url:
            episodes = get_episodes_from_feed(feed_url)
            if episodes:
                st.markdown(f'## {podcast_name}')
                
                for episode in episodes:
                    with st.expander(f"🎧 {episode.title}"):
                        session = Session()
                        existing_episode = session.query(Episode).filter(
                            Episode.episode_name == episode.title
                        ).first()
                        
                        if existing_episode and existing_episode.summary:
                            display_episode_summary(episode, existing_episode.summary)
                        else:
                            with st.spinner("Processing episode..."):
                                summary = process_episode(episode, podcast_name)
                                if summary:
                                    display_episode_summary(episode, summary)
                                else:
                                    st.error("Could not process this episode.")
                        
                        session.close()
            else:
                st.warning("No episodes found for this podcast.")
        else:
            st.error("Could not find the podcast. Please check the name and try again.")

# Chatbot section with improved response handling
st.markdown("### 💬 Ask About a Podcast Episode")
user_query = st.text_input("Ask a question:", key="user_query")

if user_query:
    with st.spinner("Finding answer..."):
        response = query_summary(user_query, "")
        if response and response != "**I don't have enough information about that episode yet. Try another question or check back later.**":
            st.markdown(f"{response}", unsafe_allow_html=True)
        else:
            st.warning("No relevant information found for your question. Try another query or check back later.")
