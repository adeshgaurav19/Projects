# 🧠 Mental Health Chatbot (Hugging Face Deployment)

### 🌟 Overview
This is a **Mental Health Chatbot** designed to provide emotional support and safe discussions about mental health. It:
- 💬 Uses **Mistral-7B** for AI-generated responses.
- 🛑 **Detects and flags high-risk messages** (e.g., "suicide", "self-harm").
- 📧 **Sends email alerts** if a message contains sensitive content.

---
### 🚀 How It Works
1. Users can **chat with the AI** about mental health.
2. The chatbot **processes messages** using **Mistral-7B**.
3. The **Safety Checker** scans for **high-risk words** (e.g., "suicide").
4. If flagged, **an email alert is sent**.

---
### 🏗️ Technology Used
- **Hugging Face Transformers** for chatbot responses.
- **Gradio** for the user interface.
- **Regex-based safety checker** for monitoring flagged words.
- **SMTP (Gmail)** for sending emergency alerts.

---
### 🖥️ Deployment on Hugging Face Spaces
1. **Create a Hugging Face Space** with `Gradio`.
2. **Push the code to GitHub or directly to Hugging Face**.
3. **Start the chatbot** and interact in real-time.

---
### ⚠️ Important Notes
- This chatbot **does not replace professional help**.
- If you or someone you know is in crisis, **please reach out to a professional**.
