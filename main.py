import asyncio
import websockets
import sys
import json
import base64
import time
from openai import OpenAI
from dotenv import load_dotenv
import os
import requests 
import uuid
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from io import BytesIO
from pydub import AudioSegment

# Load environment variables from .env file
load_dotenv()

# Fetch API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
# openai.api_key = OPENAI_API_KEY

elevenlabsclient = ElevenLabs(api_key=ELEVENLABS_API_KEY)


def text_to_speech_base64(text: str) -> str:
    # Text-to-speech conversion with detailed parameters
    response = elevenlabsclient.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2_5",
        voice_settings=VoiceSettings(stability=0.0, similarity_boost=1.0, style=0.0, use_speaker_boost=True),
    )

    audio_data = BytesIO()
    for chunk in response:
        if chunk:
            audio_data.write(chunk)

    audio_data.seek(0)
    audio_segment = AudioSegment.from_mp3(audio_data)
    audio_segment = audio_segment.set_frame_rate(8000).set_sample_width(1).set_channels(1)

    ulaw_buffer = BytesIO()
    audio_segment.export(ulaw_buffer, format="wav", codec="pcm_mulaw")

    return base64.b64encode(ulaw_buffer.getvalue()).decode("utf-8")


def deepgram_connect():
    extra_headers = {'Authorization': f'Token {DEEPGRAM_API_KEY}'}
    return websockets.connect("wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000&endpointing=true", extra_headers=extra_headers)


conversation_history_map = {}

async def get_openai_response(transcript, streamSid):
    try:
        conversation_history_map[streamSid].append({"role": "user", "content": transcript})
        stream = client.chat.completions.create(model="gpt-4o-mini", messages=conversation_history_map[streamSid], stream=True)

        response, mid_response, mid_response_count = "", "", 0
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                delta = chunk.choices[0].delta.content
                response += delta
                mid_response += delta
                mid_response_count += 1
                if mid_response_count > 19:
                    mid_response_count = 0
                    yield mid_response
                    mid_response = ""

        conversation_history_map[streamSid].append({"role": "assistant", "content": response})
        
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        yield "Sorry, I couldn't process the response."


async def proxy(client_ws, path):
    outbox = asyncio.Queue()
    audio_cursor, conn_start, prompt_count = 0.0, time.time(), 0
    streamSid = ""
    
    async with deepgram_connect() as deepgram_ws1, deepgram_connect() as deepgram_ws2:
        async def deepgram_sender(deepgram_ws):
            while True:
                chunk = await outbox.get()
                await deepgram_ws.send(chunk)

        async def deepgram_receiver(deepgram_ws):
            nonlocal audio_cursor, prompt_count
            async for message in deepgram_ws:
                try:
                    dg_json = json.loads(message)
                    transcript = dg_json["channel"]["alternatives"][0]["transcript"]
                    print(f"transcipt here: {transcript}")
                    if transcript:
                        prompt_count += 1
                        if prompt_count > 1:
                            await client_ws.send(json.dumps({"event": "clear", "streamSid": streamSid}))
                        async for chunk in get_openai_response(transcript, streamSid):  
                            payload = text_to_speech_base64(chunk)
                            await client_ws.send(json.dumps({
                                "event": "media",
                                "streamSid": streamSid,
                                "media": {"payload": payload}
                            }))
                        await client_ws.send(json.dumps({
                            "event": "mark",
                            "streamSid": streamSid,
                            "mark": {"name": "prompt_ends"}
                        }))
                except json.JSONDecodeError:
                    print('Error parsing Deepgram response as JSON.')
                    continue

        async def client_receiver(client_ws):
            nonlocal streamSid, audio_cursor, prompt_count
            BUFFER_SIZE = 20 * 160
            buffer, empty_byte_received = bytearray(b''), False

            async for message in client_ws:
                try:
                    data = json.loads(message)
                    if data["event"] in ("connected", "start"):
                        if data["event"] == "start":
                            streamSid = data["streamSid"]
                            conversation_history_map[streamSid] = [{"role": "system", "content": "You are a helpful assistant simulating a natural conversation."}]
                        continue
                    if data["event"] == "media":
                        media = data["media"]
                        chunk = base64.b64decode(media["payload"])
                        audio_cursor += len(chunk) / 8000.0
                        buffer.extend(chunk)
                        if chunk == b'':
                            empty_byte_received = True
                    if data["event"] == "mark":
                        if data["mark"]["name"] == "prompt_ends":
                            prompt_count -= 1
                    if data["event"] == "stop":
                        streamSid = data["streamSid"]
                        del conversation_history_map[streamSid]
                        break
                    if len(buffer) >= BUFFER_SIZE or empty_byte_received:
                        outbox.put_nowait(buffer)
                        buffer = bytearray(b'')
                except json.JSONDecodeError:
                    print('Message from client not formatted correctly.')
                    break

            outbox.put_nowait(b'')

        await asyncio.wait([
            asyncio.ensure_future(deepgram_sender(deepgram_ws1)),
            asyncio.ensure_future(deepgram_receiver(deepgram_ws1)),
            asyncio.ensure_future(deepgram_receiver(deepgram_ws2)),  # Second receiver
            asyncio.ensure_future(client_receiver(client_ws))
        ])

        await client_ws.close()
        del conversation_history_map[streamSid]
        print('Proxy session finished')


async def main():
    port = int(os.environ.get("PORT", 5000))
    proxy_server = websockets.serve(proxy, '0.0.0.0', port)
    await proxy_server
    await asyncio.Future()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped")
