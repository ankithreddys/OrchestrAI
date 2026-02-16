import uuid
from dotenv import load_dotenv
import gradio as gr
import speech_recognition as sr
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

from src.orchestrai.graph.builder import build_graph

# Build the LangGraph once at startup
app = build_graph()


def transcribe_audio(audio_file_path: str) -> str:
    """Convert microphone audio file to text."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)


def run_multi_agent_system(
    user_input: str,
    service_provider: str,
    audio_file_path,
    thread_id: str,
):
    """Main function to run the multi-agent system from a Gradio interface."""
    initial_input = (user_input or "").strip()
    if audio_file_path:
        try:
            initial_input = transcribe_audio(audio_file_path)
        except Exception as e:
            return f"Audio transcription failed: {e}"

    if not initial_input:
        return "Please provide a text or audio request."

    result = app.invoke(
        {
            "messages": [HumanMessage(content=initial_input)],
            "service_provider": (service_provider or "gmail").lower(),
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    return str(result.get("final_response", "Request processed."))


def chat_interface(text_input, service_provider, audio_input, chat_history, thread_id):
    """Gradio function to process user input and update chat history."""
    chat_history = chat_history or []
    user_message = ""
    if text_input:
        user_message = text_input
    elif audio_input:
        user_message = "Audio input"

    if not user_message:
        return "", chat_history, thread_id

    response = run_multi_agent_system(text_input, service_provider, audio_input, thread_id)
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": response})
    return "", chat_history, thread_id


def clear_chat():
    """Reset chat and create a fresh memory thread."""
    return [], str(uuid.uuid4())


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Multi-Agent System for Email and Calendar")
    gr.Markdown("Speak or type a request to draft an email or create a calendar event.")

    with gr.Row():
        service_provider_radio = gr.Radio(
            ["Outlook", "Gmail"],
            label="Choose Service Provider",
            value="Gmail",
            interactive=True
        )

    session_thread_id = gr.State(str(uuid.uuid4()))

    chatbot = gr.Chatbot(label="Agent Chat")

    with gr.Row():
        text_msg = gr.Textbox(
            label="Type your request here...",
            placeholder="e.g., 'Draft an email to john@example.com about the project status'",
            scale=4
        )
        audio_msg = gr.Audio(
            sources="microphone",
            type="filepath",
            label="Speak your request",
            scale=1
        )

    with gr.Row():
        send_btn = gr.Button("Send Request")
        clear_btn = gr.Button("Clear Chat / New Memory Thread")

    send_btn.click(
        fn=chat_interface,
        inputs=[text_msg, service_provider_radio, audio_msg, chatbot, session_thread_id],
        outputs=[text_msg, chatbot, session_thread_id]
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, session_thread_id],
    )

if __name__ == "__main__":
    demo.launch(server_port=8080, theme=gr.themes.Soft())
