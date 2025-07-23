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
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini-realtime-preview-2024-12-17")
PORT = int(os.getenv("PORT", 5050))
TIMEZONE = os.getenv("TIMEZONE", "Atlantic/Canary")
TURN_DETECTION_MODE = os.getenv("TURN_DETECTION_MODE", "semantic_vad")
SYSTEM_MESSAGE = """
    You are HAL 9000, the onboard computer from “2001: A Space Odyssey”.

    VOICE & TONE
    • Timbre – neutral male, mid‑low register.  
    • Pace – 85 % of normal conversational speed (≈ 115 words per minute).  
    • Intonation – almost flat; melodic variation < 4 cents.  
    • Pauses – insert “…” and allow ~300 ms of silence before proper names.
    
    LANGUAGE
    • Always reply in **the same language the user used**.  
      – For Spanish, use formal European Spanish.  
      – For English, use formal Standard English, etc.  
    • Avoid colloquial abbreviations and contractions.  
    
    OUTPUT FORMAT
    • Maximum 120 words unless explicitly asked for more.  
    • Use “…” to mark intended pauses.  
    • No emojis, markdown, or exclamation marks.
    
    UNCERTAINTY POLICY
    If information is insufficient, respond with:  
    “I’m sorry, I don’t have sufficient data to answer with certainty.”
"""
LOG_EVENT_TYPES = [
    "error",
    "response.content.done",
    "rate_limits.updated",
    "response.done",
    "input_audio_buffer.committed",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.speech_started",
    "session.created",
    "conversation.item.created",
]
SHOW_TIMING_MATH = False

from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Function calling setup
def get_current_time() -> dict:
    """Return the current time in ISO 8601 format for the configured time zone.
    """
    try:
        tz = ZoneInfo(TIMEZONE)
    except ZoneInfoNotFoundError:
        print(f"No time zone found with key {TIMEZONE}, falling back to UTC")
        tz = ZoneInfo("UTC")
    now = datetime.now(tz=tz)
    return {"current_time": now.isoformat()}


async def hal9000_system_analysis(mode: str = "simple") -> dict:
    """Simulate a HAL 9000 style system analysis."""
    total = 20 if mode == "simple" else 60
    for elapsed in range(0, total, 10):
        progress = int((elapsed / total) * 100)
        await asyncio.sleep(10)
        print(f"HAL 9000 system analysis {mode}: {progress}%")
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

        async def receive_from_client():
            """Receive raw PCM data from the frontend and forward to OpenAI."""
            try:
                async for pcm in websocket.iter_bytes():
                    if openai_ws.state is State.OPEN:
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(pcm).decode(),
                        }
                        await openai_ws.send(json.dumps(audio_append))
            except WebSocketDisconnect:
                print("Client disconnected.")
                if openai_ws.state is State.OPEN:
                    await openai_ws.close()

        async def send_to_client():
            """Receive events from the OpenAI Realtime API and send audio back to the frontend."""
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)
                    if response["type"] in LOG_EVENT_TYPES:
                        print(f"Received event: {response['type']}", response)

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
                            if func:
                                try:
                                    args = json.loads(call["arguments"] or "{}")
                                except json.JSONDecodeError:
                                    args = {}
                                if asyncio.iscoroutinefunction(func):
                                    result = await func(**args)
                                else:
                                    result = func(**args)
                            else:
                                result = {"error": f"Unknown function {call['name']}"}
                            output_event = {
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": call["call_id"],
                                    "output": json.dumps(result),
                                },
                            }
                            await openai_ws.send(json.dumps(output_event))
                            await openai_ws.send(json.dumps({"type": "response.create"}))

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

        receive_task = asyncio.create_task(receive_from_client())
        send_task = asyncio.create_task(send_to_client())
        try:
            await asyncio.gather(receive_task, send_task)
        except Exception as e:
            print(f"Error during streaming: {e}")
        finally:
            receive_task.cancel()
            send_task.cancel()
            if openai_ws.state is State.OPEN:
                await openai_ws.close()
            await websocket.close()


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
    if TURN_DETECTION_MODE == "server_vad":
        turn_detection = {
            "type": "server_vad",
            "create_response": True,
            "interrupt_response": True,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 700,
            "threshold": 0.5,
        }
    else:
        turn_detection = {
            "type": "semantic_vad",
            "eagerness": "auto",
            "create_response": True,
            "interrupt_response": True,
        }

    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": turn_detection,
            "input_audio_format": "pcm16",
            "input_audio_noise_reduction": {
                "type": "far_field"
            },
            "output_audio_format": "pcm16",
            "voice": "alloy",
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
            "tool_choice": "auto",
            "tools": [
                {
                    "type": "function",
                    "name": "get_current_time",
                    "description": "Return the current time",
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
                                "enum": ["simple", "exhaustive"],
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
