import os
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import gradio as gr
from authentication import Authentication

# Load environment variables
load_dotenv()

# Initialize API clients
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API Key not set in environment variables")
if not anthropic_api_key:
    raise ValueError("Anthropic API Key not set in environment variables")

openai_client = OpenAI(api_key=openai_api_key)
claude_client = anthropic.Anthropic(api_key=anthropic_api_key)

# System message for AI assistants
SYSTEM_MESSAGE = (
    "You are a helpful assistant, trying your best to answer every question as accurately as possible. "
    "You are also free to say you do not know if you do not have the information to answer a question. "
    "You always respond in markdown."
)

# Conversation memory
conversation_history: Dict[str, List[Dict[str, str]]] = {"GPT": [], "Claude": []}

def stream_gpt(prompt: str) -> str:
    """Stream responses from GPT model."""
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        *conversation_history["GPT"],
        {"role": "user", "content": prompt}
    ]

    try:
        stream = openai_client.chat.completions.create(
            model='gpt-4',
            messages=messages,
            stream=True
        )
        result = ""
        for chunk in stream:
            delta_content = chunk.choices[0].delta.content
            if delta_content is not None:
                result += delta_content
                yield result
        conversation_history["GPT"].extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": result}
        ])
    except Exception as e:
        yield f"Error: {str(e)}"

def stream_claude(prompt: str) -> str:
    """Stream responses from Claude model."""
    messages = [
        *conversation_history["Claude"],
        {"role": "user", "content": prompt},
    ]

    try:
        result = claude_client.messages.stream(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0.7,
            system=SYSTEM_MESSAGE,
            messages=messages,
        )
        response = ""
        with result as stream:
            for text in stream.text_stream:
                response += text
                yield response
        conversation_history["Claude"].extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ])
    except Exception as e:
        yield f"Error: {str(e)}"

def stream_model(prompt: str, model: str) -> str:
    """Stream responses based on selected model."""
    if model == "GPT":
        yield from stream_gpt(prompt)
    elif model == "Claude":
        yield from stream_claude(prompt)
    else:
        raise ValueError("Unknown model. Please select either 'GPT' or 'Claude'.")

# Define Gradio interface with dark mode
iface = gr.Interface(
    fn=stream_model,
    inputs=[
        gr.Textbox(label="Your message:"),
        gr.Dropdown(["GPT", "Claude"], label="Select model", value="Claude")
    ],
    outputs=gr.Markdown(label="Response:"),
    title="AI Assistant",
    description="Chat with GPT or Claude AI models",
    allow_flagging="never"
)

def main():
    """Run the Gradio interface."""
    iface.launch(share=True, 
                auth=Authentication.auth, 
                auth_message="Please enter your credentials to access the chat interface",)

if __name__ == "__main__":
    main()
