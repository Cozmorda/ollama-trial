import json
import subprocess
import shlex
import requests
import streamlit as st
from typing import Generator


OLLAMA_BASE = "http://localhost:11434"
OLLAMA_HTTP = OLLAMA_BASE + "/api/generate"


def get_available_models() -> list:
    urls = [OLLAMA_BASE + "/api/tags"]
    for url in urls:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code != 200:
                continue
            data = r.json()

            # If it's a simple list of strings
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data

            # If it's a list of dicts with names
            if isinstance(data, list):
                names = []
                for item in data:
                    if isinstance(item, dict):
                        name = item.get("name") or item.get("model") or item.get("id")
                        if name:
                            names.append(name)
                if names:
                    return names

            # If it's a dict containing a models list (e.g., {"models": [{"name": ...}, ...]})
            if isinstance(data, dict):
                inner = data.get("models") or data.get("results") or data.get("available")
                if isinstance(inner, list):
                    out = []
                    for m in inner:
                        if isinstance(m, str):
                            out.append(m)
                        elif isinstance(m, dict):
                            name = m.get("name") or m.get("model") or m.get("id")
                            if name:
                                out.append(name)
                    if out:
                        return out
        except Exception:
            continue
    return ["ollama"]


def stream_from_ollama_http(prompt: str, model: str = "ollama") -> Generator[str, None, None]:
    payload = {"model": model, "prompt": prompt, "stream": True}
    try:
        resp = requests.post(OLLAMA_HTTP, json=payload, stream=True, timeout=5)
    except Exception:
        return

    if resp.status_code != 200:
        return

    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        # Ollama may stream lines prefixed with 'data: '
        if isinstance(raw, bytes):
            line = raw.decode("utf-8", errors="ignore")
        else:
            line = raw
        if not line:
            continue
        if line.startswith("data:"):
            line = line[len("data:") :].strip()
        if not line or line == "[DONE]":
            continue
        try:
            obj = json.loads(line)
        except Exception:
            yield line
            continue
        # Try to extract text from a couple of possible shapes
        text = None
        if isinstance(obj, dict):
            # openai-like streaming pieces
            choices = obj.get("choices")
            if choices and isinstance(choices, list):
                delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
                if isinstance(delta, dict):
                    text = delta.get("content") or delta.get("text")
                else:
                    text = choices[0].get("text")
            # generic field
            if not text:
                text = obj.get("text") or obj.get("content")
        if text:
            yield text
        else:
            # last-resort: stringify
            yield line


def stream_from_ollama_cli(prompt: str, model: str = "ollama") -> Generator[str, None, None]:
    # Fallback: try calling the local `ollama` CLI if available.
    cmd = ["ollama", "generate", model, "--prompt", prompt, "--stream"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception:
        return

    assert proc.stdout
    for line in proc.stdout:
        yield line


def stream_response(prompt: str, model: str = "ollama") -> Generator[str, None, None]:
    # Try HTTP stream first, then CLI fallback
    for chunk in stream_from_ollama_http(prompt, model=model) or []:
        yield chunk
    for chunk in stream_from_ollama_cli(prompt, model=model) or []:
        yield chunk


def app():
    st.set_page_config(page_title="Ollama Chat", page_icon="🤖")
    st.title("Ollama Chat — Streamlit")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    available_models = get_available_models()

    with st.form(key="chat_form", clear_on_submit=True):
        sel = st.selectbox("Available models", options=available_models + ["custom..."], index=0)
        if sel == "custom...":
            model = st.text_input("Custom model name", value="ollama")
        else:
            model = sel

        user_input = st.text_area("You", value="", height=100)
        submit = st.form_submit_button("Send")

    if submit and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        placeholder = st.empty()
        assistant_content = ""
        prompt = "\n".join([m["content"] for m in st.session_state.messages])
        for chunk in stream_response(prompt, model=model):
            assistant_content += chunk
            placeholder.markdown(assistant_content)

        st.session_state.messages.append({"role": "assistant", "content": assistant_content})

    # show history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")


if __name__ == "__main__":
    app()
