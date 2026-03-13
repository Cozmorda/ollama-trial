import json
import time
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

    decoder = json.JSONDecoder()
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

        # Some servers send multiple JSON objects on one line. Decode in a loop.
        buffer = line.strip()
        while buffer:
            try:
                obj, idx = decoder.raw_decode(buffer)
                buffer = buffer[idx:].lstrip()
            except Exception:
                # If we cannot parse, fall back to raw text
                yield buffer
                break

            # Try to extract text from known shapes
            text = None
            if isinstance(obj, dict):
                # Ollama streaming uses "response" for tokens
                if "response" in obj and isinstance(obj["response"], str):
                    text = obj["response"]

            if text:
                yield text


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


def stream_response(prompt: str, model: str) -> Generator[str, None, None]:
    # Try HTTP stream first, then CLI fallback
    yielded_any = False
    for chunk in stream_from_ollama_http(prompt, model=model) or []:
        yielded_any = True
        yield chunk
    if yielded_any:
        return
    for chunk in stream_from_ollama_cli(prompt, model=model) or []:
        yield chunk


def app():
    st.set_page_config(page_title="Ollama Chat", page_icon="🤖")
    st.title("Ollama Chat — Streamlit")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    available_models = get_available_models()

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = available_models[0] if available_models else "ollama"

    options = available_models + ["custom..."]
    if st.session_state.selected_model not in options:
        options = [st.session_state.selected_model] + options

    sel = st.selectbox(
        "Available models",
        options=options,
        index=options.index(st.session_state.selected_model),
    )
    if sel == "custom...":
        model = st.text_input("Custom model name", value=st.session_state.selected_model)
    else:
        model = sel

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("You", value="", height=100)
        submit = st.form_submit_button("Send")

    if submit and user_input:
        st.session_state.selected_model = model
        st.session_state.messages.append({"role": "user", "content": user_input})

        timer_placeholder = st.empty()
        status_placeholder = st.empty()
        user_prompt_placeholder = st.empty()

        user_prompt_placeholder.markdown(f"**You:** {user_input}")
        status_placeholder.markdown("⏳ Waiting for response...")
        
        prompt = user_input
        start_time = time.perf_counter()        
        assistant_content = st.write_stream(stream_response(prompt, model=model))
        elapsed = time.perf_counter() - start_time
        
        timer_placeholder.markdown(f"⏱️ {elapsed:.2f}s")

        st.session_state.messages.append({"role": "assistant", "content": assistant_content})
        status_placeholder.markdown("✅ Response complete")


if __name__ == "__main__":
    app()
