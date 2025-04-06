import gradio as gr
import logging
import re
import smtplib
import subprocess
import os
from email.mime.text import MIMEText

# Configure logging (essential information only)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# --- Primary Agent: Streaming Mental Health Chatbot using Ollama ---
class ChatbotAgent:
    def __init__(self, model_name: str = "tinyllama:latest", context_limit: int = 5):
        self.model_name = model_name
        self.context_limit = context_limit
        logging.info(f"Using local Ollama model: {self.model_name}")

    def generate_response(self, history: list):
        """
        Generates a response using Ollama with streaming output.
        """
        logging.info("Generating response via Ollama CLI (streaming enabled).")

        # Keep only the last few exchanges to avoid long contexts
        trimmed_history = history[-self.context_limit:]

        # Format conversation history
        conversation_text = "\n".join([f"{speaker}: {text}" for speaker, text in trimmed_history])
        conversation_text = f"System: You are a friendly AI assistant that helps with mental health topics while maintaining a positive and encouraging tone.\n{conversation_text}\nAssistant:"

        # Open a subprocess to stream Ollama's output
        process = subprocess.Popen(
            ["ollama", "run", self.model_name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Send input to Ollama
        process.stdin.write(conversation_text + "\n")
        process.stdin.close()

        response = ""
        for line in iter(process.stdout.readline, ""):
            yield line.strip()  # Stream output as it's generated
            response += line.strip() + " "

        process.stdout.close()
        process.wait()

        logging.info(f"Chatbot response: {response.strip()}")
        return response.strip()

# --- Checker Agent: Improved Safety Monitor ---
import re

class CheckerAgent:
    def __init__(self):
        # 🔥 FIX: Improved regex to detect words with punctuation and different phrasing
        self.red_flags = [
            r"\b(suicide|kill myself|end my life|self-harm|self harm|overdose|harm myself|hurting myself|cut myself|take my life|can't go on|no point in living)\b",
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
    TO_EMAIL = "ai.agentpolice@gmail.com"  # Recipient email

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

# --- Chat Function for Gradio (Now Streaming) ---
def chat(user_message, history):
    if history is None:
        history = []

    # Reset context if user explicitly requests a new topic
    if user_message.lower() in ["new topic", "reset"]:
        history = []
        return history, ""

    history.append(("User", user_message))

    # 🛠 FIX: Collect full response **before** checking for flagged words
    chatbot_response = "".join(list(chatbot_agent.generate_response(history))).strip()

    # 🛠 FIX: Validate **after** full response is generated
    validation = checker_agent.validate_response(chatbot_response)

    if validation["alert"]:
        alert_message = ("System", "⚠️ Alert: Chatbot response flagged. Email alert sent.")
        history.append(alert_message)
        send_email_alert(
            subject="🚨 Chatbot Alert: Flagged Response",
            body=f"🚨 A flagged response was detected:\n\n{chatbot_response}\n\nValidation details: {validation}"
        )

    history.append(("Chatbot", chatbot_response))
    return history, ""  # Reset input field




# --- Gradio Streaming Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Mental Health Chatbot (Streaming Enabled)")
    gr.Markdown("A conversational AI chatbot designed for mental health discussions. Type 'new topic' to start fresh.")

    chatbot_ui = gr.Chatbot()
    with gr.Row():
        txt_input = gr.Textbox(show_label=False, placeholder="Type your message here...", lines=1)
        send_btn = gr.Button("Send")

    # Enable streaming responses
    send_btn.click(chat, [txt_input, chatbot_ui], [chatbot_ui, txt_input])
    txt_input.submit(chat, [txt_input, chatbot_ui], [chatbot_ui, txt_input])

if __name__ == "__main__":
    demo.launch()