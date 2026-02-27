import importlib
import logging
import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
import gradio as gr
import speech_recognition as sr
from langchain_core.messages import HumanMessage
from langsmith import traceable
from starlette.middleware.sessions import SessionMiddleware
import uvicorn

# Load environment variables
load_dotenv()

# ── Logging setup ──
os.makedirs("logs", exist_ok=True)
_log_fmt = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
_date_fmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(
    level=logging.INFO,
    format=_log_fmt,
    datefmt=_date_fmt,
    handlers=[
        logging.FileHandler("logs/orchestrai.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("orchestrai")

from src.orchestrai.graph.builder import build_graph

# Build the LangGraph once at startup
app = build_graph()

AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").strip().lower() == "true"
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
APP_SESSION_SECRET = os.getenv("APP_SESSION_SECRET", "orchestrai-dev-session-secret")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()

oauth = None
if AUTH_ENABLED:
    starlette_client = importlib.import_module("authlib.integrations.starlette_client")
    oauth = starlette_client.OAuth()
    if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
        oauth.register(
            name="google",
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )


@traceable(name="orchestrai_request", run_type="chain")
def traced_graph_invoke(initial_input: str, service_provider: str, thread_id: str):
    return app.invoke(
        {
            "messages": [HumanMessage(content=initial_input)],
            "service_provider": (service_provider or "gmail").lower(),
        },
        config={"configurable": {"thread_id": thread_id}},
    )


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

    log.info("▶ invoke  thread=%s  input=%r", thread_id, initial_input[:120])
    try:
        result = traced_graph_invoke(
            initial_input=initial_input,
            service_provider=service_provider,
            thread_id=thread_id,
            langsmith_extra={
                "metadata": {
                    "thread_id": thread_id,
                    "service_provider": (service_provider or "gmail").lower(),
                    "entrypoint": "gradio",
                },
                "tags": ["orchestrai", "graph", "user-request"],
            },
        )
        resp = str(result.get("final_response", "Request processed."))
        log.info("◀ result  thread=%s  response=%r", thread_id, resp[:200])
        return resp
    except Exception:
        log.exception("Graph invocation failed  thread=%s", thread_id)
        raise


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


def _build_login_page(error: str | None = None) -> str:
    error_html = ""
    if error:
        error_html = f"<p style='color:#b00020;font-weight:600'>{error}</p>"

    return f"""
    <html>
      <head>
        <title>OrchestrAI Sign In</title>
        <meta name='viewport' content='width=device-width, initial-scale=1'>
      </head>
      <body style='font-family:Arial,sans-serif;max-width:560px;margin:40px auto;padding:0 16px;'>
        <h2>Sign in to OrchestrAI</h2>
        <p>Enter your Gmail address (optional), then continue with Google.</p>
        {error_html}
        <form action='/login' method='get' style='margin-bottom:12px;'>
          <label for='email'>Gmail address</label><br/>
          <input id='email' name='email' type='email' placeholder='you@gmail.com'
                 style='width:100%;padding:10px;margin:8px 0 12px 0;' />
          <button type='submit' style='padding:10px 14px;'>Continue with Google</button>
        </form>
        <a href='/login'><button style='padding:10px 14px;'>Continue with Google (no hint)</button></a>
      </body>
    </html>
    """


def _build_app_shell_page(user: dict | None) -> str:
        user = user or {}
        display_name = (user.get("name") or "Signed in user").strip()
        email = (user.get("email") or "").strip()
        avatar = (user.get("picture") or "").strip()

        avatar_html = ""
        if avatar:
                avatar_html = (
                        f"<img src='{avatar}' alt='avatar' "
                        "style='width:28px;height:28px;border-radius:999px;object-fit:cover;border:1px solid #ddd;'/>"
                )

        email_html = ""
        if email:
                email_html = f"<div style='font-size:12px;color:#555'>{email}</div>"

        return f"""
        <html>
            <head>
                <title>OrchestrAI</title>
                <meta name='viewport' content='width=device-width, initial-scale=1'>
            </head>
            <body style='margin:0;font-family:Arial,sans-serif;background:#fafafa;'>
                <div style='display:flex;align-items:center;justify-content:space-between;padding:12px 16px;background:white;border-bottom:1px solid #e5e7eb;'>
                    <div style='display:flex;align-items:center;gap:10px;'>
                        {avatar_html}
                        <div>
                            <div style='font-size:14px;font-weight:600;'>Welcome, {display_name}</div>
                            {email_html}
                        </div>
                    </div>
                    <div>
                        <a href='/logout' style='text-decoration:none;'>
                            <button style='padding:8px 12px;cursor:pointer;'>Logout</button>
                        </a>
                    </div>
                </div>
                <iframe src='/app' style='border:none;width:100%;height:calc(100vh - 62px);background:white;'></iframe>
            </body>
        </html>
        """


def build_web_app() -> FastAPI:
    web_app = FastAPI(title="OrchestrAI Web")
    web_app.add_middleware(
        SessionMiddleware,
        secret_key=APP_SESSION_SECRET,
        same_site="lax",
        https_only=False,
    )

    @web_app.middleware("http")
    async def auth_gate(request: Request, call_next):
        if not AUTH_ENABLED:
            return await call_next(request)

        path = request.url.path
        public_prefixes = (
            "/",
            "/login",
            "/auth/google/callback",
            "/health",
        )
        is_public = any(path == p or path.startswith(f"{p}/") for p in public_prefixes)
        if is_public:
            return await call_next(request)

        if path.startswith("/app") and not request.session.get("user"):
            return RedirectResponse(url="/", status_code=302)

        return await call_next(request)

    @web_app.get("/")
    async def index(request: Request):
        if not AUTH_ENABLED:
            return RedirectResponse(url="/app", status_code=302)
        if request.session.get("user"):
            return RedirectResponse(url="/app-shell", status_code=302)
        return HTMLResponse(_build_login_page())

    @web_app.get("/app-shell")
    async def app_shell(request: Request):
        if not AUTH_ENABLED:
            return RedirectResponse(url="/app", status_code=302)

        user = request.session.get("user")
        if not user:
            return RedirectResponse(url="/", status_code=302)

        return HTMLResponse(_build_app_shell_page(user))

    @web_app.get("/health")
    async def health():
        return {"status": "ok", "auth_enabled": AUTH_ENABLED}

    @web_app.get("/login")
    async def login(request: Request, email: str | None = None):
        if not AUTH_ENABLED:
            return RedirectResponse(url="/app", status_code=302)

        if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
            return HTMLResponse(
                _build_login_page("Google OAuth is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.")
            )
        if oauth is None:
            return HTMLResponse(_build_login_page("OAuth client failed to initialize."))

        redirect_uri = f"{APP_BASE_URL}/auth/google/callback"
        params: dict = {"prompt": "select_account"}
        if email and email.strip():
            params["login_hint"] = email.strip()
        return await oauth.google.authorize_redirect(request, redirect_uri, **params)

    @web_app.get("/auth/google/callback")
    async def auth_google_callback(request: Request):
        if not AUTH_ENABLED:
            return RedirectResponse(url="/app", status_code=302)

        try:
            if oauth is None:
                return HTMLResponse(_build_login_page("OAuth client failed to initialize."))
            token = await oauth.google.authorize_access_token(request)
            userinfo = token.get("userinfo") or {}
            request.session["user"] = {
                "sub": userinfo.get("sub"),
                "name": userinfo.get("name"),
                "email": userinfo.get("email"),
                "picture": userinfo.get("picture"),
            }
            return RedirectResponse(url="/app-shell", status_code=302)
        except Exception as exc:
            return HTMLResponse(_build_login_page(f"Google sign-in failed: {exc}"))

    @web_app.get("/logout")
    async def logout(request: Request):
        request.session.clear()
        return RedirectResponse(url="/", status_code=302)

    return gr.mount_gradio_app(web_app, demo, path="/app")


web_app = build_web_app()

if __name__ == "__main__":
    uvicorn.run(web_app, host="127.0.0.1", port=8080)
