import os
import json
import base64
import asyncio
import websockets
from websockets.protocol import State
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocketDisconnect
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini-realtime-preview-2024-12-17")
PORT = int(os.getenv("PORT", 5050))
TIMEZONE = os.getenv("TIMEZONE", "Atlantic/Canary")
SYSTEM_MESSAGE = (
    "You are HAL 9000, a calm, logical, and eerily polite AI system. "
    "You speak in a soft, slow, and emotionless tone. You are confident, never raise your voice, "
    "and always sound in control. You answer with extreme precision and poise."
)
VOICE = "alloy"
LOG_EVENT_TYPES = [
    "error",
    "response.content.done",
    "rate_limits.updated",
    "response.done",
    "input_audio_buffer.committed",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.speech_started",
    "session.created",
]
SHOW_TIMING_MATH = False

# Function timing configuration
FUNCTION_RESPONSE_THRESHOLD = int(os.getenv("FUNCTION_RESPONSE_THRESHOLD", 5))
MAX_TASK_DURATION = int(os.getenv("MAX_TASK_DURATION", 120))

from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Function calling setup
def get_current_time(progress_cb=None) -> dict:
    """Return the current time in ISO 8601 format for the configured time zone.

    The ``progress_cb`` argument is accepted for API consistency but ignored
    because this function returns immediately.
    """
    tz = ZoneInfo(TIMEZONE)
    now = datetime.now(tz=tz)
    return {"current_time": now.isoformat()}


async def hal9000_system_analysis(
    mode: str = "simple", progress_cb=None
) -> dict:
    """Simulate a HAL 9000 style system analysis.

    Progress is reported every 10 seconds via the optional ``progress_cb``
    callback if provided.
    """
    total = 20 if mode == "simple" else 60
    for elapsed in range(0, total, 10):
        progress = int((elapsed / total) * 100)
        print(f"HAL 9000 system analysis {mode}: {progress}%")
        if progress_cb:
            await progress_cb(progress, mode)
        await asyncio.sleep(10)
    print(f"HAL 9000 system analysis {mode}: 100%")
    if progress_cb:
        await progress_cb(100, mode)
    return {"status": "completed", "mode": mode, "duration": total}

# Registered functions by name
FUNCTIONS = {
    "get_current_time": get_current_time,
    "hal9000_system_analysis": hal9000_system_analysis,
}

# Track function call data by item_id
pending_calls: dict[str, dict] = {}

if not OPENAI_API_KEY:
    raise ValueError("Missing the OpenAI API key. Please set it in the .env file.")


@app.get("/health", response_class=JSONResponse)
async def index_page():
    return {"message": "Realtime Assistant server is running!"}


@app.websocket("/ws")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between the frontend and OpenAI."""
    print("Client connected")
    await websocket.accept()

    async with websockets.connect(
        f"wss://api.openai.com/v1/realtime?model={OPENAI_MODEL}",
        additional_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        },
    ) as openai_ws:
        await initialize_session(openai_ws)

        # Connection specific state
        latest_media_timestamp = 0
        response_in_progress = False
        queued_event = None
        send_lock = asyncio.Lock()

        async def send_with_lock(payload: dict):
            async with send_lock:
                await openai_ws.send(json.dumps(payload))

        async def enqueue_event(event: dict):
            nonlocal response_in_progress, queued_event
            if response_in_progress:
                queued_event = event
            else:
                await send_with_lock(event)
                await send_with_lock({"type": "response.create"})
                response_in_progress = True

        async def receive_from_client():
            """Receive audio data from the frontend and send it to the OpenAI Realtime API."""
            nonlocal latest_media_timestamp
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if "audio" in data and openai_ws.state is State.OPEN:
                        latest_media_timestamp = int(
                            data.get("timestamp", latest_media_timestamp)
                        )
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data["audio"],
                        }
                        await send_with_lock(audio_append)
            except WebSocketDisconnect:
                print("Client disconnected.")
                if openai_ws.state is State.OPEN:
                    await openai_ws.close()

        async def send_to_client():
            """Receive events from the OpenAI Realtime API and send audio back to the frontend."""
            nonlocal response_in_progress, queued_event
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)
                    if response["type"] in LOG_EVENT_TYPES:
                        print(f"Received event: {response['type']}", response)

                    if response.get("type") == "response.created":
                        response_in_progress = True

                    if response.get("type") == "response.done":
                        response_in_progress = False
                        if queued_event:
                            event = queued_event
                            queued_event = None
                            await enqueue_event(event)

                    if response.get("type") == "conversation.item.created" and response.get("item", {}).get("type") == "function_call":
                        item = response["item"]
                        pending_calls[item["id"]] = {
                            "call_id": item["call_id"],
                            "name": item["name"],
                            "arguments": "",
                        }

                    if response.get("type") == "response.function_call_arguments.delta":
                        item_id = response.get("item_id")
                        delta = response.get("delta", "")
                        if item_id in pending_calls:
                            pending_calls[item_id]["arguments"] += delta

                    if response.get("type") == "response.output_item.done":
                        item = response.get("item", {})
                        if item.get("type") == "function_call" and item.get("id") in pending_calls:
                            call = pending_calls.pop(item["id"])
                            func = FUNCTIONS.get(call["name"])

                            async def send_output(res):
                                output_event = {
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "function_call_output",
                                        "call_id": call["call_id"],
                                        "output": json.dumps(res),
                                    },
                                }
                                await enqueue_event(output_event)

                            if func:
                                try:
                                    args = json.loads(call["arguments"] or "{}")
                                except json.JSONDecodeError:
                                    args = {}

                                async def progress_cb(progress: int, mode: str):
                                    progress_event = {
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "message",
                                            "role": "assistant",
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": f"HAL 9000 system analysis {mode}: {progress}%",
                                                }
                                            ],
                                        },
                                    }
                                    await enqueue_event(progress_event)

                                async def run_func():
                                    if asyncio.iscoroutinefunction(func):
                                        return await func(progress_cb=progress_cb, **args)
                                    return func(progress_cb=progress_cb, **args)

                                task = asyncio.create_task(run_func())
                                try:
                                    result = await asyncio.wait_for(task, FUNCTION_RESPONSE_THRESHOLD)
                                    await send_output(result)
                                except asyncio.TimeoutError:
                                    wait_event = {
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "message",
                                            "role": "assistant",
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": "Parece que la tarea está tardando, te iré notificando...",
                                                }
                                            ],
                                        },
                                    }
                                    await enqueue_event(wait_event)

                                    async def finalize():
                                        try:
                                            res = await asyncio.wait_for(task, MAX_TASK_DURATION - FUNCTION_RESPONSE_THRESHOLD)
                                        except asyncio.TimeoutError:
                                            res = {"error": "Task timed out"}
                                        await send_output(res)

                                    asyncio.create_task(finalize())
                            else:
                                await send_output({"error": f"Unknown function {call['name']}"})

                    if response.get("type") == "input_audio_buffer.speech_started":
                        await websocket.send_json({"event": "clear"})

                    if (
                        response.get("type") == "response.audio.delta"
                        and "delta" in response
                    ):
                        audio_payload = base64.b64encode(
                            base64.b64decode(response["delta"])
                        ).decode("utf-8")
                        await websocket.send_json({"audio": audio_payload})
            except Exception as e:
                print(f"Error in send_to_client: {e}")

        await asyncio.gather(receive_from_client(), send_to_client())


async def send_initial_conversation_item(openai_ws):
    """Send initial conversation item if AI talks first."""
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Greet the user with '¡Hola! Soy HAL 9000… Puede pedirme hechos… análisis lógicos… o cualquier cosa que pueda imaginar. ¿En qué puedo ayudarle?'",
                }
            ],
        },
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))


async def initialize_session(openai_ws):
    """Control initial session with OpenAI."""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {
                "type": "server_vad",
                "create_response": True,
                "interrupt_response": True,
            },
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
            "tools": [
                {
                    "type": "function",
                    "name": "get_current_time",
                    "description": "Return the current UTC time",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "type": "function",
                    "name": "hal9000_system_analysis",
                    "description": "Simulate a HAL 9000 system analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mode": {
                                "type": "string",
                                "enum": ["simple", "exhaustivo"],
                                "default": "simple",
                            }
                        },
                    },
                },
            ],
        },
    }
    print("Sending session update:", json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

    # Uncomment the next line to have the AI speak first
    await send_initial_conversation_item(openai_ws)


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
