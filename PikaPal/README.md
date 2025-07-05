---
title: StoryTeller Agent
sdk: streamlit
emoji: 🏆
colorFrom: blue
colorTo: purple
short_description: Storyteller agent for kids
---
# PikaPal: Safe & Trustworthy AI for Kids

PikaPal is a **kid-friendly AI assistant** that generates fun bedtime stories, offers mini-games, and keeps parents in the loop. We ensure **trust & safety** by sanitizing user input with a **Trust Layer** and using a **second LLM** for moderation. A **Gmail node** also notify parents about session details or the final story.

## Overview
- **Workflow (n8n)**: Receives user requests from Streamlit → Trust Layer (Code Node) → Basic LLM (Gemini or other) → Optional 2nd LLM (Moderation) → Respond → (Optional) Gmail for parental oversight.
- **Streamlit App**: A whimsical UI featuring dancing Pikachu, story input, and mini-games.

## Architecture & Trust Layer

User (Streamlit) --> [Webhook] --> [Trust Layer (Code Node)] 
  --> [Basic LLM Chain] --> [2nd LLM] --> [Respond to Webhook] 
  --> [Gmail Node -> Email to Parents]


1. **Webhook**: Takes user input (e.g., “dragons”).
2. **Trust Layer (Code Node)**:
   - Masks personal data (email, phone, address).
   - Replaces cuss words with `[BLOCKED_WORD]`.
3. **Basic LLM Chain**: Calls Gemini (or any model) to generate child-friendly text.
4. **(Optional) 2nd LLM**: Performs a moderation pass to flag or rewrite unsafe content.
5. **Respond to Webhook**: Sends sanitized output back to Streamlit.
6. **Gmail Node**: Emails the story or session summary to parents, if desired.

## Usage

1. **n8n Workflow**:
   - Create nodes: Webhook → Trust Layer (Code Node) → Basic LLM → Optional 2nd LLM → Respond → Optional Gmail.
   - In the Trust Layer, use regex to remove or mask personal info and cuss words.
   - Configure your LLM (Gemini or other) to generate or moderate the story.

2. **Streamlit (PikaPal UI)**:
   - Run `streamlit run app.py`.
   - Enter a story topic; the sanitized text is processed by n8n.
   - The final story returns to the UI, displayed in a fun, colorful interface.

3. **Parents in the Loop**:
   - An optional Gmail node can email logs or the final story to parents, ensuring transparency and safety.

## Why It’s Safe
- **Trust Layer**: Removes private info, blocks profanity, ensuring the LLM never sees or returns sensitive details.
- **Optional Moderation**: A second LLM can intercept harmful or age-inappropriate output.
- **Parental Email**: Allows parents to track sessions, adding human oversight.


Have fun building your **safe & magical** kids’ AI experience with PikaPal!