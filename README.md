Ollama Streamlit Chat
=====================

Quickstart
----------

- Ensure you have a local Ollama installation running (the app expects the HTTP API at `http://localhost:11434`).
- Install Python dependencies (recommended using a venv):

```bash
pip install -r requirements.txt
# or with pyproject-based tooling, use your preferred installer
```

- Run the Streamlit app:

```bash
streamlit run main.py
```

Notes
-----
- The app will try to stream responses from the Ollama HTTP API and will fall back to the `ollama` CLI if available.
- If your Ollama server is on a different host/port or uses a different model name, update `main.py` accordingly.
