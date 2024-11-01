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
elevenlabsclient = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def text_to_speech_base64(text: str) -> str:
    response = elevenlabsclient.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB",  # Adam pre-made voice
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2_5",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
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
    ulaw_base64 = base64.b64encode(ulaw_buffer.getvalue()).decode("utf-8")
    return ulaw_base64

def deepgram_connect():
    extra_headers = {'Authorization': f'Token {DEEPGRAM_API_KEY}'}
    deepgram_ws = websockets.connect(
        "wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000&endpointing=true", 
        extra_headers=extra_headers
    )
    return deepgram_ws

conversation_history_map = {}

async def get_openai_response(transcript, streamSid):
    try:
        conversation_history_map[streamSid].append({"role": "user", "content": transcript})
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation_history_map[streamSid],
            stream=True,
        )
        response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
        
        conversation_history_map[streamSid].append({"role": "assistant", "content": response})      
        return response     
    except Exception as e:
        print(f"Error in OpenAI API call:  {e}")
        return "Sorry, I couldn't process the response."

async def deepgram_receiver(deepgram_ws, client_ws, streamSid):
    async for message in deepgram_ws:
        try:
            dg_json = json.loads(message)
            transcript = dg_json["channel"]["alternatives"][0]["transcript"]
            if transcript:
                print(f"[Deepgram Receiver] Transcript: {transcript}")
                response = await get_openai_response(transcript, streamSid)
                payload = text_to_speech_base64(response)
                await client_ws.send(json.dumps({
                    "event": "media",
                    "streamSid": streamSid,
                    "media": {
                        "payload": payload
                    }
                }))
                await client_ws.send(json.dumps({
                    "event": "mark",
                    "streamSid": streamSid,
                    "mark": {
                        "name": "response_ends"
                    }
                }))
        except json.JSONDecodeError:
            print('[Deepgram Receiver] Was not able to parse Deepgram response as JSON.')

async def deepgram_receiver_log(deepgram_ws):
    async for message in deepgram_ws:
        try:
            dg_json = json.loads(message)
            transcript = dg_json["channel"]["alternatives"][0]["transcript"]
            if transcript:
                print(f"[Deepgram Logging Receiver] Transcript: {transcript}")
        except json.JSONDecodeError:
            print('[Deepgram Logging Receiver] Was not able to parse Deepgram response as JSON.')

async def proxy(client_ws, path):
    outbox = asyncio.Queue()
    audio_cursor = 0.0
    conn_start = time.time()
    streamSid = ""
    prompt_count = 0
    
    async with deepgram_connect() as deepgram_ws:
        await asyncio.gather(
            deepgram_receiver(deepgram_ws, client_ws, streamSid),
            deepgram_receiver_log(deepgram_ws),
        )

        async def deepgram_sender(deepgram_ws):
            while True:
                chunk = await outbox.get()
                await deepgram_ws.send(chunk)

        async def client_receiver(client_ws):
            nonlocal streamSid
            nonlocal audio_cursor
            nonlocal prompt_count
            BUFFER_SIZE = 20 * 160
            buffer = bytearray(b'')
            empty_byte_received = False
            async for message in client_ws:
                try:
                    data = json.loads(message)
                    if data["event"] in ("connected", "start"):
                        if data['event'] in ("start"):
                            streamSid = data['streamSid']
                            conversation_history_map[streamSid] = [
                                {"role": "system", "content": "You are a helpful assistant simulating a natural conversation."}
                            ]
                        continue
                    if data["event"] == "media":
                        media = data["media"]
                        chunk = base64.b64decode(media["payload"])
                        time_increment = len(chunk) / 8000.0
                        audio_cursor += time_increment
                        buffer.extend(chunk)
                        if chunk == b'':
                            empty_byte_received = True
                    if data["event"] == "stop":
                        streamSid = data['streamSid']
                        del conversation_history_map[streamSid]
                        break
                    if len(buffer) >= BUFFER_SIZE or empty_byte_received:
                        outbox.put_nowait(buffer)
                        buffer = bytearray(b'')
                except json.JSONDecodeError:
                    print('Message from client not formatted correctly, bailing')
                    break

            outbox.put_nowait(b'')

        await asyncio.gather(
            deepgram_sender(deepgram_ws),
            client_receiver(client_ws),
        )

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
