# HAL 9000 Web Speech Assistant (Python)

This project exposes a small FastAPI server that connects a HAL 9000‑themed web interface with the OpenAI Realtime API.
Audio from your microphone is streamed to OpenAI over WebSocket and the synthesized response is played back in the browser.
CSS credits: `https://codepen.io/giana/pen/XmjOBG`

## Prerequisites

- **Python 3.13+** – tested with Python 3.13.5
- **An OpenAI API key** with access to the Realtime API
- Optional: Docker if you prefer running the container

## Local Setup

1. (Optional) create and activate a virtual environment
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```
2. Install the dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the example environment file and add your credentials
   ```bash
   cp .env.example .env
   # edit .env and set OPENAI_API_KEY
   ```
4. Start the server
   ```bash
   python main.py
   ```

Visit `http://localhost:5050` and click **Start** to converse with HAL.

## Docker

Create the `.env` file as above and run:
```bash
docker compose up --build
```

## Features

- HAL 9000 inspired UI served from the `static` folder
- Streams audio to and from the OpenAI Realtime API
- Optional initial greeting (see `send_initial_conversation_item` in `main.py`)
- Basic interruption handling when you talk over HAL
- Demonstrates Realtime API function calling with `get_current_time`
  and `hal9000_system_analysis` tools
- Includes a retrieval-augmented generation tool using Qdrant
  for hybrid (dense + sparse) document search with paragraph chunking
  powered by OpenAI embeddings. See the
  [hybrid search tutorial](https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/)
  and [async API guide](https://qdrant.tech/documentation/database-tutorials/async-api/)
  for details.
- The time zone used by `get_current_time` is configurable with the
  `TIMEZONE` variable (defaults to `Atlantic/Canary`). If the specified zone is
  unavailable, the server falls back to `UTC`.
- Set `TURN_DETECTION_MODE` to `semantic_vad` or `server_vad` to control how
  HAL detects turns (defaults to `semantic_vad`)
- Audio is exchanged as 16-bit little-endian PCM at 24kHz over the WebSocket
  connection, and HAL responds in the same 24kHz PCM format

## RAG Setup

The assistant can search local documents using Qdrant. Set the following
environment variables or update `.env`:

- `QDRANT_URL` – URL for the Qdrant instance (default `http://localhost:6333`)
- `RAG_DOCS_DIR` – directory containing text files to index (default `./docs`)
- `RAG_COLLECTION` – collection name in Qdrant (default `rag_docs`)

Set `OPENAI_EMBEDDING_MODEL` to choose the model used for embeddings
(default `text-embedding-3-small`).
Place your documents in `RAG_DOCS_DIR` before starting the server. Paragraphs
are indexed using OpenAI embeddings combined with TF-IDF sparse vectors. The search returns the ten most relevant chunks.

Have fun—and remember, HAL is always listening.
