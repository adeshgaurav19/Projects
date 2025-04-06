import gradio as gr
import logging
import re
import os
import smtplib
from email.mime.text import MIMEText
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# --- Primary Agent: Mental Health Chatbot using a Small Model ---
class ChatbotAgent:
    def __init__(self, model_name: str = "distilgpt2"):
        self.chatbot = pipeline("text-generation", model=model_name)
        logging.info(f"Using lightweight model: {model_name}")

    def generate_response(self, user_input: str) -> str:
        """
        Generates a response using the small chatbot model.
        """
        logging.info("Generating response via Hugging Face model.")
        response = self.chatbot(user_input, max_length=100, do_sample=True, temperature=0.7)
        return response[0]['generated_text'].strip()

# --- Checker Agent: Improved Safety Monitor ---
class CheckerAgent:
    def __init__(self):
        self.red_flags = [
            r"\b(suicide|kill myself|end my life|self-harm|overdose|harm myself|cut myself|take my life|can't go on|no point in living)\b",
            r"\b(depressed|hopeless|worthless|empty inside|tired of life|give up)\b"
        ]

    def validate_response(self, response: str) -> dict:
        alert = False
        flags_detected = []
        for pattern in self.red_flags:
            if re.search(pattern, response, re.IGNORECASE):
                flags_detected.append(pattern)
                alert = True

        validation_result = {
            'validated_response': response,
            'alert': alert,
            'flags_detected': flags_detected
        }

        if alert:
            logging.warning(f"⚠️ Risk detected! Keywords matched: {flags_detected}")
        else:
            logging.info("Response is safe.")

        return validation_result

# --- Secure Email Alert Function ---
def send_email_alert(subject: str, body: str):
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    SMTP_USERNAME = os.getenv("SMTP_USERNAME", "your_email@gmail.com")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "your_app_password")
    FROM_EMAIL = SMTP_USERNAME
    TO_EMAIL = "ai.agentpolice@gmail.com"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = FROM_EMAIL
    msg["To"] = TO_EMAIL

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(FROM_EMAIL, [TO_EMAIL], msg.as_string())
        logging.info("✅ Email alert sent successfully.")
    except Exception as e:
        logging.error(f"❌ Failed to send email alert: {e}")

# --- Global Instances ---
chatbot_agent = ChatbotAgent()
checker_agent = CheckerAgent()

# --- Chat Function for Gradio ---
def chat(user_message, history):
    if history is None:
        history = []

    if user_message.lower() in ["new topic", "reset"]:
        history = []
        return history, ""

    history.append(("User", user_message))

    chatbot_response = chatbot_agent.generate_response(user_message)

    validation = checker_agent.validate_response(chatbot_response)

    if validation["alert"]:
        alert_message = ("System", "⚠️ Alert: Chatbot response flagged. Email alert sent.")
        history.append(alert_message)
        send_email_alert(
            subject="🚨 Chatbot Alert: Flagged Response",
            body=f"🚨 A flagged response was detected:\n\n{chatbot_response}\n\nValidation details: {validation}"
        )

    history.append(("Chatbot", chatbot_response))
    return history, ""

# --- 🎨 Cool UI for Gradio ---
with gr.Blocks(css="body {background-color: #f5f5f5; font-family: 'Arial', sans-serif;}") as demo:
    gr.Markdown(
        """
        <h1 style="text-align: center; color: #333;">🧠 Mental Health Chatbot</h1>
        <p style="text-align: center; font-size: 16px; color: #555;">
        A conversational AI chatbot designed for mental health discussions. Type 'new topic' to start fresh.</p>
        """,
        unsafe_allow_html=True
    )

    chatbot_ui = gr.Chatbot(
        label="Chat with AI",
        bubble_full_width=False,
        height=400
    )

    with gr.Row():
        txt_input = gr.Textbox(
            show_label=False,
            placeholder="Type your message here...",
            lines=1
        )
        send_btn = gr.Button("💬 Send", variant="primary")

    # Enable chat
    send_btn.click(chat, [txt_input, chatbot_ui], [chatbot_ui, txt_input])
    txt_input.submit(chat, [txt_input, chatbot_ui], [chatbot_ui, txt_input])

if __name__ == "__main__":
    demo.launch()
