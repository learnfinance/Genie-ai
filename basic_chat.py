"""
Basic LLM Chat App Tutorial Demos
---------------------------------
Three Streamlit chat examples:
- Echo bot (session state + chat UI)
- Simple random-responder with streaming emulator
- ChatGPT-like clone with OpenAI streaming
"""

import random
import time
from typing import List, Dict, Any

import streamlit as st
from openai import OpenAI
import openai as openai_legacy


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def get_openai_client(api_key: str):
    """Create an OpenAI client that works for both new and legacy SDKs."""
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except TypeError:
        openai_legacy.api_key = api_key
        return openai_legacy


def stream_chat_completion(client, messages: List[Dict[str, str]], model: str):
    """Call OpenAI chat completion with streaming across SDK versions."""
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        return client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
    if hasattr(client, "ChatCompletion"):
        return client.ChatCompletion.create(
            model=model,
            messages=messages,
            stream=True,
        )
    raise RuntimeError("OpenAI client is not initialized correctly.")


def delta_from_chunk(chunk: Any) -> str:
    """Extract text delta from streamed chunk across SDK versions."""
    # New SDK object style
    if hasattr(chunk, "choices"):
        choice = chunk.choices[0]
        delta = getattr(choice, "delta", None)
        if delta and getattr(delta, "content", None):
            return delta.content
        if getattr(choice, "message", None) and getattr(choice.message, "content", None):
            return choice.message.content
    # Legacy dict style
    if isinstance(chunk, dict):
        choice = chunk.get("choices", [{}])[0]
        if "delta" in choice and choice["delta"].get("content"):
            return choice["delta"]["content"]
        if "message" in choice and choice["message"].get("content"):
            return choice["message"]["content"]
    return ""


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Basic LLM Chat Apps", page_icon="💬", layout="wide")
st.title("Basic LLM Chat Apps")
st.caption("Echo bot, simple streamer, and ChatGPT-like clone — all in Streamlit.")

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("Use tabs below to try each example. Tab 3 needs an API key.")

tabs = st.tabs(["Echo Bot", "Simple Streamer", "ChatGPT-like", "File-aware Chat"])


# -----------------------------------------------------------------------------
# Tab 1: Echo bot
# -----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Echo bot (session state + chat UI)")
    if "echo_messages" not in st.session_state:
        st.session_state.echo_messages = []

    for msg in st.session_state.echo_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Say something"):
        st.session_state.echo_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = f"Echo: {prompt}"
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.echo_messages.append({"role": "assistant", "content": response})


# -----------------------------------------------------------------------------
# Tab 2: Simple streamer
# -----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Simple chatbot with streamed response (toy)")
    if "simple_messages" not in st.session_state:
        st.session_state.simple_messages = []

    for msg in st.session_state.simple_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    RESPONSES = [
        "Hello there! How can I assist you today?",
        "Hi, human! Anything I can help with?",
        "Do you need help with something?",
    ]

    def response_generator():
        response = random.choice(RESPONSES)
        for word in response.split():
            yield word + " "
            time.sleep(0.05)

    if prompt := st.chat_input("Ask something"):
        st.session_state.simple_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            streamed = st.write_stream(response_generator())
        st.session_state.simple_messages.append({"role": "assistant", "content": streamed})


# -----------------------------------------------------------------------------
# Tab 3: ChatGPT-like clone
# -----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("ChatGPT-like (OpenAI streaming)")
    if "llm_messages" not in st.session_state:
        st.session_state.llm_messages = []
    if "openai_model" not in st.session_state:
        st.session_state.openai_model = "gpt-4o-mini"

    client = get_openai_client(api_key)
    if not client:
        st.info("Enter your OpenAI API key in the sidebar to enable this tab.")

    for msg in st.session_state.llm_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Chat with the model"):
        st.session_state.llm_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if client:
            with st.chat_message("assistant"):
                stream = stream_chat_completion(
                    client,
                    [{"role": m["role"], "content": m["content"]} for m in st.session_state.llm_messages],
                    model=st.session_state.openai_model,
                )
                full_response = ""
                for chunk in stream:
                    token = delta_from_chunk(chunk)
                    if not token:
                        continue
                    full_response += token
                    st.write(token)
            st.session_state.llm_messages.append({"role": "assistant", "content": full_response})
        else:
            with st.chat_message("assistant"):
                st.markdown("⚠️ OpenAI key required for this tab.")


# -----------------------------------------------------------------------------
# Tab 4: File-aware chat (accepts attachments)
# -----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("File-aware chat (attachments + text responses)")
    st.caption("Attach files via chat input; assistant acknowledges and replies.")

    if "filechat_messages" not in st.session_state:
        st.session_state.filechat_messages = []

    for msg in st.session_state.filechat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    submission = st.chat_input(
        "Ask a question or attach files",
        accept_file="multiple",
        file_type=["txt", "pdf", "docx"],
    )

    if submission:
        user_text = submission.text if hasattr(submission, "text") else submission.get("text", "")
        user_files = submission.files if hasattr(submission, "files") else submission.get("files", [])

        # Render user message
        with st.chat_message("user"):
            st.markdown(user_text or "(no text, files only)")
            if user_files:
                st.write(f"Attached files ({len(user_files)}):")
                for f in user_files:
                    st.write(f"- {f.name} ({f.size} bytes)")

        st.session_state.filechat_messages.append({"role": "user", "content": user_text or "(files attached)"})

        # Simple assistant reply
        with st.chat_message("assistant"):
            file_summary = ""
            if user_files:
                file_summary = "I received these files: " + ", ".join([f.name for f in user_files])
            response = f"Got it! {file_summary or 'No files attached.'} How can I help with them?"
            st.markdown(response)

        st.session_state.filechat_messages.append({"role": "assistant", "content": response})
