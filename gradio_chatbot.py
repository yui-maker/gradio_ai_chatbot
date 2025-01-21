import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from authentication import Authentication

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    print("OpenAI API Key not set")

openai = OpenAI()
MODEL = 'gpt-4o-mini'

system_message = (
    "You are a helpful assistant, trying your best to answer every question as accurately as possible. "
    "You are also free to say you do not know if you do not have the information to answer a question. "
    "You always respond in markdown."
)

def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

# Create and launch the interface
demo = gr.ChatInterface(
    fn=chat,
    title="AI chatbot",
    description="Please login to use the chat interface",
)

if __name__ == "__main__":
    demo.launch(share=True,
        auth=Authentication.auth,
        auth_message="Please enter your credentials to access the chat interface",
    )