
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
from pydub import AudioSegment, silence
import threading
import queue
import re  # Import regex module to detect sentence-ending punctuation
import boto3
import audioop




# Load environment variables from .env file
load_dotenv()

# Fetch API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

AWS_ACCESS_KEY_ID= os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

client = OpenAI(api_key=OPENAI_API_KEY)
# openai.api_key = OPENAI_API_KEY


elevenlabsclient = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)

def text_to_speech_base64(text: str) -> str:
    # Call the Eleven Labs text-to-speech API
    response = elevenlabsclient.text_to_speech.convert(
        voice_id="pMsXgVXv3BLzUgSXRplE",  # Adam pre-made voice
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2_5",  # Use the turbo model for low latency
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    # Collect audio data chunks
    audio_data = BytesIO()
    for chunk in response:
        if chunk:
            audio_data.write(chunk)

    # Convert MP3 audio data to AudioSegment, set to 8000 Hz, 1 channel, 16-bit PCM
    audio_data.seek(0)
    audio_segment = AudioSegment.from_mp3(audio_data)
    audio_segment = audio_segment.set_frame_rate(8000).set_sample_width(2).set_channels(1)

    # Split audio on silence and recombine to remove long pauses
    chunks = silence.split_on_silence(
        audio_segment,
        min_silence_len=500,  # Minimum length of silence in ms to consider as a break
        silence_thresh=-50,   # Silence threshold in dB
        keep_silence=200      # Keep 200 ms of silence around each chunk for smooth transitions
    )
    
    # Combine chunks back together
    trimmed_audio = AudioSegment.empty()
    for chunk in chunks:
        trimmed_audio += chunk

    # Export audio as raw PCM (s16le) without headers
    pcm_buffer = BytesIO()
    trimmed_audio.export(pcm_buffer, format="s16le")

    # Convert PCM to μ-law encoding
    pcm_data = pcm_buffer.getvalue()
    ulaw_data = audioop.lin2ulaw(pcm_data, 2)  # 2 bytes for 16-bit PCM

    # Encode μ-law data in base64
    ulaw_base64 = base64.b64encode(ulaw_data).decode("utf-8")

    return ulaw_base64


def text_to_speech_base64_poly(text: str) -> str:
    # Initialize the Polly client with credentials
    polly_client = boto3.client(
        'polly',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

    # Request text-to-speech conversion from Polly
    response = polly_client.synthesize_speech(
        Text=text,
        VoiceId='Matthew',  # You can use any other Polly voice
        OutputFormat='mp3',
        Engine='standard'  # 'neural' engine can also be used if available
    )

    # Read audio data from the response stream
    audio_data = BytesIO(response['AudioStream'].read())

    # Convert to AudioSegment and set frame rate to 8000 Hz with μ-law encoding
    audio_segment = AudioSegment.from_mp3(audio_data)
    audio_segment = audio_segment.set_frame_rate(8000).set_sample_width(1).set_channels(1)

    # Export audio to a μ-law encoded BytesIO buffer
    ulaw_buffer = BytesIO()
    audio_segment.export(ulaw_buffer, format="wav", codec="pcm_mulaw")

    # Encode to base64
    ulaw_base64 = base64.b64encode(ulaw_buffer.getvalue()).decode("utf-8")

    # Return the base64-encoded string
    return ulaw_base64




def deepgram_connect():
    extra_headers = {'Authorization': f'Token {DEEPGRAM_API_KEY}'}
    deepgram_ws = websockets.connect(
        "wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000&endpointing=true", 
        extra_headers=extra_headers
    )
    return deepgram_ws


conversation_history_map = {}
stop_event = threading.Event() 


async def process_chunk(chunk, streamSid, client_ws):
    # Your asynchronous processing logic here
    # For example, you might want to perform an I/O operation or a database call
    print(f"processing: {chunk}")
    payload =  text_to_speech_base64(chunk)
    try:
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
                    "name": chunk
                    }
                    }))
    except Exception as e:
                print("Error sending message: ", e)
                
            
    print("sent message.")
    


async def get_openai_response(transcript, streamSid, client_ws):
    global stop_event 
    try:
        # Update the conversation history for the user
        if streamSid not in conversation_history_map:
            conversation_history_map[streamSid] = []  # Initialize if not present

        conversation_history_map[streamSid].append({"role": "user", "content": transcript})

        # Create the chat completion stream
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation_history_map[streamSid],
            stream=True,
        )

        chunk_buffer = []

        # Process the stream and collect chunks
        for chunk in stream:  # Use a regular for loop since stream is not async
            if stop_event.is_set():  # Check if the stop signal has been set
                print("Stopping OpenAI request processing.")
                break 
            
            if chunk.choices[0].delta.content is not None:
                chunk_buffer.append(chunk.choices[0].delta.content)
                combined_chunk = ''.join(chunk_buffer)
                
                # Check if `combined_chunk` ends with a sentence-ending punctuation mark
                if re.search(r'[.,!?;:]$', combined_chunk):
                    print(f"Sent chunk: {combined_chunk}")
                    await process_chunk(combined_chunk, streamSid, client_ws)  # Call your async function here
                    chunk_buffer = []  # Reset the buffer

        print("___________Came out of for loop____________")
        # After finishing the stream, enqueue any remaining chunks
        if chunk_buffer and not stop_event.is_set():
            combined_chunk = ''.join(chunk_buffer)
            print(f"Sent chunk: {combined_chunk}")
            await process_chunk(combined_chunk, streamSid, client_ws)  # Call your async function for the last chunk

        # Append the full response to the conversation history
        full_response = ''.join(
            [chunk.choices[0].delta.content for chunk in stream if chunk.choices[0].delta.content is not None]
        )
        conversation_history_map[streamSid].append({"role": "assistant", "content": full_response})
        return await client_ws.send(json.dumps({ 
            "event": "mark",
            "streamSid": streamSid,
            "mark": {
                "name": "ends"
            }
        }))

    except Exception as e:
        print(f"Error in OpenAI API call: {e}")


def run_openai_response(transcript, streamSid, client_ws):
    """Run the OpenAI response function in an asyncio loop."""
    asyncio.run(get_openai_response(transcript, streamSid, client_ws))


async def proxy(client_ws, path):
    outbox = asyncio.Queue()

    audio_cursor = 0.0
    conn_start = time.time()

    streamSid = ""
    prompt_count = 0
    
    # Initialize global variables for buffering
    transcript_buffer = []
    buffer_lock = threading.Lock()
    last_update_time = None
    
    
    async with deepgram_connect() as deepgram_ws:
        async def deepgram_sender(deepgram_ws):
            while True:
                chunk = await outbox.get()
                await deepgram_ws.send(chunk)

        async def deepgram_receiver(deepgram_ws):
            nonlocal audio_cursor
            nonlocal prompt_count
            async for message in deepgram_ws:
                try:
                    dg_json = json.loads(message)
                    transcript = dg_json["channel"]["alternatives"][0]["transcript"]
                    print(f"transcript : {transcript}")
                    if transcript:
                        asyncio.create_task(handle_response(transcript=transcript))
                        
                    print("receiving ends here")
                except json.JSONDecodeError:
                    print('Was not able to parse Deepgram response as JSON.')
                    continue


        # Function to process and clear the buffer
        async def process_buffer():
            nonlocal transcript_buffer, last_update_time
            
            nonlocal prompt_count
              # Get response from OpenAI API
            prompt_count += 1
            if prompt_count > 1:
                print(f"stopppinnnnnnngggg   :  {prompt_count}")
                stop_event.set()
                time.sleep(3)
                stop_event.clear()
                prompt_count = 1
                await client_ws.send(json.dumps({ 
                        "event": "clear",
                        "streamSid": streamSid,
                    }))
                
                
            # Concatenate all transcripts
            concatenated_transcript = " ".join(transcript_buffer)
            
            # Start the OpenAI response function in a new thread
            openai_thread = threading.Thread(target=lambda: asyncio.run(run_openai_response(concatenated_transcript, streamSid, client_ws)))
            openai_thread.start()
            
            # Clear the buffer and reset last update time
            with buffer_lock:
                transcript_buffer = []
            last_update_time = None


        # Background task to monitor the buffer and process it if no new transcripts arrive within 2 seconds
        async def monitor_buffer():
            nonlocal last_update_time

            while True:
                await asyncio.sleep(2)  # Wait for 2 seconds to check for new transcripts
                
                # Check if 2 seconds have passed since the last update
                if last_update_time is not None and time.monotonic() - last_update_time >= 2:
                    with buffer_lock:
                        if transcript_buffer:  # Only process if there's something in the buffer
                            await process_buffer()
                    break  # Exit the loop after processing the buffer
                
                
        # The main handler function for incoming transcripts
        async def handle_response(transcript):
            nonlocal transcript_buffer, last_update_time

            # Add the transcript to the buffer
            with buffer_lock:
                transcript_buffer.append(transcript)
                last_update_time = time.monotonic()  # Update the timestamp of the last addition
            
            # Check if we need to start the buffer monitor task
            if not hasattr(handle_response, "monitor_task") or handle_response.monitor_task.done():
                handle_response.monitor_task = asyncio.create_task(monitor_buffer())


        # async def handle_response(transcript):
                  
             
        #     print(f"Active threads before creating a new one: {threading.active_count()}")
        #     # Start a new thread for the OpenAI response function
        #     openai_thread = threading.Thread(target=lambda: asyncio.run(run_openai_response(transcript, streamSid, client_ws)))
        #     openai_thread.start()

            


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
                            conversation_history_map[streamSid] = [ {"role": "system", "content": "You are a helpful assistant simulating a natural conversation."}]
                        continue
                    if data["event"] == "media":
                        media = data["media"]
                        chunk = base64.b64decode(media["payload"])
                        time_increment = len(chunk) / 8000.0
                        audio_cursor += time_increment
                        buffer.extend(chunk)
                        if chunk == b'':
                            empty_byte_received = True
                                
                    if data["event"] == "mark": 
                        try: 
                            if data['mark']['name'] == "ends":
                                prompt_count -= 1
                            print(f"mark : {data['mark']['name']}")
                        except Exception as e:
                            print(e)
                        
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

        
        print("start")
        await asyncio.wait([
            asyncio.ensure_future(deepgram_sender(deepgram_ws)),
            asyncio.ensure_future(deepgram_receiver(deepgram_ws)),
            asyncio.ensure_future(client_receiver(client_ws))
        ])
        print("ends")

        await client_ws.close()
        del conversation_history_map[streamSid]
        print('Finished running the proxy')

# async def main():
#     proxy_server = websockets.serve(proxy, 'localhost', 5000)
#     await proxy_server

#     await asyncio.Future()  # Keep the server running indefinitely

async def main():
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    proxy_server = websockets.serve(proxy, '0.0.0.0', port)  # Bind to all interfaces
    await proxy_server

    await asyncio.Future()  # Keep the server running indefinitely


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped")



