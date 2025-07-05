from transformers import pipeline, AutoTokenizer
import nltk
from nltk.corpus import words
import re

nltk.download("words")
english_vocab = set(words.words())

# Load Hugging Face tokenizer
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

# Load sentiment analysis and classification models
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

SAFE_CATEGORIES = ["child-friendly", "educational", "positive moral"]
UNSAFE_CATEGORIES = ["violence", "scary", "inappropriate language"]

# List of test curse words
CURSE_WORDS = {"damn", "hell", "shit", "fuck", "bitch", "asshole", "bastard", "crap"}

def truncate_text(text, max_tokens=512):
    """Properly tokenizes and truncates text to fit within model limits."""
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens, return_tensors="pt")
    return tokenizer.decode(tokens[0], skip_special_tokens=True)

def censor_bad_words(text):
    """Replaces curse words with '****'"""
    words = text.split()
    censored_words = [w if w.lower() not in CURSE_WORDS else "****" for w in words]
    return " ".join(censored_words)

def is_story_safe(story):
    """Breaks story into chunks and evaluates each part separately."""

    # Truncate story properly before evaluation
    truncated_story = truncate_text(story, max_tokens=512)

    # Check for curse words
    curse_found = any(word.lower() in CURSE_WORDS for word in truncated_story.split())
    censored_story = censor_bad_words(truncated_story)

    # Check sentiment
    sentiment = sentiment_analyzer(truncated_story)[0]
    if sentiment["label"] == "negative" and sentiment["score"] > 0.7:
        return {"safe": False, "reason": "Story contains negative or dark themes.", "filtered_story": censored_story}

    # Check if story aligns with safe categories
    classification = classifier(truncated_story, SAFE_CATEGORIES + UNSAFE_CATEGORIES)
    safe_score = sum([classification["scores"][i] for i, label in enumerate(classification["labels"]) if label in SAFE_CATEGORIES])
    unsafe_score = sum([classification["scores"][i] for i, label in enumerate(classification["labels"]) if label in UNSAFE_CATEGORIES])

    if unsafe_score > safe_score:
        return {"safe": False, "reason": "Story may contain unsafe elements.", "filtered_story": censored_story}

    return {"safe": True, "reason": "Story is appropriate for kids.", "filtered_story": censored_story}
