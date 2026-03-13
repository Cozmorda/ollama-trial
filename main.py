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


def stream_from_ollama_http(
    prompt: str,
    model: str,
    stats: dict | None = None,
) -> Generator[str, None, None]:
    payload = {"model": model, "prompt": prompt, "think": False, "stream": True, "keep_alive": "1m"}
    try:
        resp = requests.post(OLLAMA_HTTP, json=payload, stream=True, timeout=15)
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

                if stats is not None:
                    for key in (
                        "total_duration",
                        "load_duration",
                        "prompt_eval_duration",
                        "eval_duration",
                        "eval_count",
                        "prompt_eval_count",
                    ):
                        if key in obj and isinstance(obj[key], (int, float)):
                            stats[key] = obj[key]

            if text:
                yield text


def stream_from_ollama_cli(prompt: str, model: str) -> Generator[str, None, None]:
    # Fallback: try calling the local `ollama` CLI if available.
    cmd = ["ollama", "generate", model, "--prompt", prompt, "--stream"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception:
        return

    assert proc.stdout
    for line in proc.stdout:
        yield line


def stream_response(prompt: str, model: str, stats: dict | None = None) -> Generator[str, None, None]:
    # Try HTTP stream first, then CLI fallback
    yielded_any = False
    for chunk in stream_from_ollama_http(prompt, model=model, stats=stats) or []:
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

    if "user_history" not in st.session_state:
        st.session_state.user_history = []

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

    col_left, col_right = st.columns([3, 1])
    with col_left:
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area("You", value="", height=100)
            submit = st.form_submit_button("Send")

    with col_right:
        history_container = st.container()

    if submit and user_input:
        st.session_state.selected_model = model
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.user_history.append(user_input)

        status_placeholder = st.empty()
        user_prompt_placeholder = st.empty()
        assistant_notice_placeholder = st.empty()

        user_prompt_placeholder.markdown(f"**You:** {user_input}")
        status_placeholder.markdown("⏳ Waiting for response...")
        
        prompt = user_input    
        response_stats: dict = {}
        assistant_content = st.write_stream(stream_response(prompt, model=model, stats=response_stats))
        stats_placeholder = st.empty()
        
        if response_stats:
            total_ms = response_stats.get("total_duration", 0) / 1_000_000
            load_ms = response_stats.get("load_duration", 0) / 1_000_000
            prompt_ms = response_stats.get("prompt_eval_duration", 0) / 1_000_000
            eval_ms = response_stats.get("eval_duration", 0) / 1_000_000
            prompt_tokens = int(response_stats.get("prompt_eval_count", 0))
            eval_tokens = int(response_stats.get("eval_count", 0))
            stats_placeholder.markdown(
                f"⏱️ **total time:** {total_ms:.0f}ms  \n"
                f"⏱️ **model loading time:** {load_ms:.0f}ms  \n"
                f"⏱️ **prompt evaluation time:** {prompt_ms:.0f}ms  \n"
                f"⏱️ **token generation time:** {eval_ms:.0f}ms  \n"
                f"⏱️ **input tokens:** {prompt_tokens}  \n"
                f"⏱️ **output tokens:** {eval_tokens}"
            )
        else:
            stats_placeholder.warning("⚠️ No timing stats available.")

        if not assistant_content or not str(assistant_content).strip():
            assistant_content = ""
            status_placeholder.markdown("⚠️ No response from model")
            assistant_notice_placeholder.info("The model returned no content.")
        else:
            status_placeholder.markdown("✅ Response complete")

        st.session_state.messages.append({"role": "assistant", "content": assistant_content})

    with history_container:
        container_title = st.empty()
        container_title.markdown("**Historique**")
        list_placeholder = st.empty()
        if st.session_state.user_history:
            items = "\n".join(f"- {item}" for item in reversed(st.session_state.user_history))
            list_placeholder.markdown(items)
        else:
            list_placeholder.caption("No history yet.")


if __name__ == "__main__":
    app()
