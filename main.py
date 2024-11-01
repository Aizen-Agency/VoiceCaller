
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
import threading
import queue
import time


# Load environment variables from .env file
load_dotenv()

# Fetch API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
# openai.api_key = OPENAI_API_KEY


elevenlabsclient = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)


def text_to_speech_base64(text: str) -> str:
    # Calling the text_to_speech conversion API with detailed parameters
    response = elevenlabsclient.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB",  # Adam pre-made voice
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2_5",  # use the turbo model for low latency
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    # Collect all chunks of audio data
    audio_data = BytesIO()
    for chunk in response:
        if chunk:
            audio_data.write(chunk)
    
    # Reset the buffer position to the start
    audio_data.seek(0)

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

# Assuming conversation_history_map and client are defined somewhere in your code
# For demonstration purposes, let's define them minimally # Replace with your OpenAI client initialization

# A simple function to process chunks in a separate thread
async def chunk_processor(chunk_queue):
    while True:
        chunk = chunk_queue.get()
        if chunk is None:  # A sentinel value to exit the loop
            break
        # Simulate processing the chunk
        print(f"Processing chunk: {chunk}")
        
        payload =  text_to_speech_base64(chunk)
        time.sleep(2)
        # try:
        #     await client_ws.send(json.dumps({
        #         "event": "media",
        #         "streamSid": streamSid,
        #         "media": {
        #             "payload": payload
        #         }
        #     }))
        #     await client_ws.send(json.dumps({ 
        #             "event": "mark",
        #             "streamSid": streamSid,
        #             "mark": {
        #             "name": transcript
        #             }
        #             }))
        # except Exception as e:
        #         print("Error sending message:", e)
                
            
        print("sent message")
        
        
        chunk_queue.task_done()

# Initialize the chunk queue
chunk_queue = queue.Queue()

# Start the chunk processing thread
processing_thread = threading.Thread(target=chunk_processor, args=(chunk_queue,))
processing_thread.daemon = True  # Make the thread a daemon thread
processing_thread.start()

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
                print("Error sending message:", e)
                
            
    print("sent message")
    

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
        chunk_count = 0

        # Process the stream and collect chunks
        for chunk in stream:  # Use a regular for loop since stream is not async
            if stop_event.is_set():  # Check if the stop signal has been set
                print("Stopping OpenAI request processing.")
                break 
            
            if chunk.choices[0].delta.content is not None:
                chunk_buffer.append(chunk.choices[0].delta.content)
                chunk_count += 1

                # Enqueue and print 20 chunks at a time
                if chunk_count == 20:
                    combined_chunk = ''.join(chunk_buffer)
                    print(f"Sent chunk: {combined_chunk}")
                    await process_chunk(combined_chunk, streamSid, client_ws)  # Call your async function here
                     # Print the sent message immediately
                    chunk_buffer = []  # Reset the buffer
                    chunk_count = 0  # Reset the chunk count

        print("___________Came out of for loop_____________")
        # After finishing the stream, enqueue any remaining chunks
        if chunk_buffer and stop_event.is_set():
            combined_chunk = ''.join(chunk_buffer)
            print(f"Sent chunk: {combined_chunk}")
            process_chunk(combined_chunk, streamSid, client_ws)  # Call your async function for the last chunk
            #print(f"Sent chunk: {combined_chunk}")  # Print the last sent message

        # Append the full response to the conversation history
        full_response = ''.join(
            [chunk.choices[0].delta.content for chunk in stream if chunk.choices[0].delta.content is not None]
        )
        conversation_history_map[streamSid].append({"role": "assistant", "content": full_response})
        await client_ws.send(json.dumps({ 
                    "event": "mark",
                    "streamSid": streamSid,
                    "mark": {
                    "name": "ends"
                    }
                    }))

    except Exception as e:
        print(f"Error in OpenAI API call: {e}")


def run_openai_response(transcript, streamSid, client_ws, loop):
    """Schedules the OpenAI response function to run within the given event loop."""
    asyncio.run_coroutine_threadsafe(get_openai_response(transcript, streamSid, client_ws), loop)



async def proxy(client_ws, path):
    outbox = asyncio.Queue()

    audio_cursor = 0.0
    conn_start = time.time()

    streamSid = ""
    prompt_count = 0
    
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


        async def handle_response(transcript):
            nonlocal prompt_count
              # Get response from OpenAI API
            prompt_count += 1
            if prompt_count > 1:
                print(f"stopppinnnnnnnggg   :  {prompt_count}")
                stop_event.set()
                time.sleep(2)
                stop_event.clear()
                await client_ws.send(json.dumps({ 
                        "event": "clear",
                        "streamSid": streamSid,
                    }))
                prompt_count = 1
                   
                    
            # Start a new thread for the OpenAI response function
            # Start a new thread for the OpenAI response function, passing the event loop
            loop = asyncio.get_event_loop()  # Get the current event loop
            openai_thread = threading.Thread(target=lambda: run_openai_response(transcript, streamSid, client_ws, loop))
            openai_thread.start()


            


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
                            print(f"mark : {data["mark"]["name"]}")
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



