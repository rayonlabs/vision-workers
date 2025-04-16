#!/usr/bin/env python3
import json
import aiohttp
import asyncio
import re
import time
import random
import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Configuration
MINER_BASE_URL = "http://185.141.218.194:6919"
VALIDATOR_BASE_URL = "http://185.141.218.194:6920"
TASK_TYPE = "chat"  # "chat" or "completion"
TASK_ID = "chat-llama-3-2-3b"
N_DATA_POINTS = 10  # Default for one-shot test; set higher for multiple iterations

# Text generation parameters
MIN_WORDS_PER_MESSAGE = 20
MAX_WORDS_PER_MESSAGE = 100
MIN_CHAT_MESSAGES = 3  # Including system message
MAX_CHAT_MESSAGES = 6  # Including system message
COMPLETION_MIN_WORDS = 50
COMPLETION_MAX_WORDS = 200

class ServerType(str, Enum):
    LLM = "llm_server"
    IMAGE = "image_server"

MODEL_CONFIG = {
    "model": "unsloth/Llama-3.2-3B-Instruct",
    "half_precision": True,
    "tokenizer": "tau-vision/llama-tokenizer-fix",
    "max_model_len": 20_000,
    "gpu_memory_utilization": 0.5,
    "tensor_parallel_size": 1,
    "eos_token_id": 128009
}

async def fetch_random_text(n_paragraphs=2, n_sentences=3) -> str:
    """Fetch random text from metaphorpsum.com."""
    url = f'http://metaphorpsum.com/paragraphs/{n_paragraphs}/{n_sentences}'
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    text = await response.text()
                    return text
                else:
                    return "The quick brown fox jumps over the lazy dog. " * 10
        except Exception:
            return "The quick brown fox jumps over the lazy dog. " * 10

def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    try:
        return sent_tokenize(text)
    except:
        return re.split(r'(?<=[.!?])\s+', text)

async def generate_random_text(n_words: int) -> str:
    """Generate random text with specified number of words."""
    text = await fetch_random_text(
        n_paragraphs=random.randint(1, 3), 
        n_sentences=random.randint(2, 5)
    )
    
    words = text.split()
    if len(words) > n_words:
        return " ".join(words[:n_words])
    
    while len(words) < n_words:
        additional_text = await fetch_random_text(
            n_paragraphs=1, 
            n_sentences=random.randint(1, 3)
        )
        words.extend(additional_text.split())
    
    return " ".join(words[:n_words])

async def generate_chat_messages(n_messages: int, min_words: int, max_words: int, session: aiohttp.ClientSession) -> List[Dict]:
    """Generate a list of chat messages with alternating roles, using LLM for assistant messages."""
    messages = [
        {
            "role": "system",
            "content": await generate_random_text(random.randint(min_words, max_words))
        }
    ]
    
    # Always start with a user message after system
    messages.append({
        "role": "user",
        "content": await generate_random_text(random.randint(min_words, max_words))
    })
    
    # We'll have (n_messages - 2) messages after system and first user
    # These should alternate between assistant and user
    for i in range(n_messages - 2):
        if i % 2 == 0:  # Assistant's turn
            # Get all messages up to this point to send to the LLM
            current_messages = messages.copy()
            
            # Create a payload for the LLM to generate the assistant's response
            llm_payload = {
                "messages": current_messages,
                "temperature": 0.7,  # Using a moderate temperature for assistant responses
                "max_tokens": random.randint(100, 300),
                "model": MODEL_CONFIG['model'],
                "top_p": 1,
                "stream": False,  # We don't need streaming for this
                "logprobs": False  # We don't need logprobs for this
            }
            
            # Call the LLM to generate the assistant's response
            miner_endpoint = f"{MINER_BASE_URL}/v1/chat/completions"
            try:
                async with session.post(miner_endpoint, json=llm_payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        assistant_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
                        # If we got a valid response, use it; otherwise generate random text
                        if assistant_content:
                            messages.append({
                                "role": "assistant",
                                "content": assistant_content
                            })
                        else:
                            messages.append({
                                "role": "assistant",
                                "content": await generate_random_text(random.randint(min_words, max_words))
                            })
                    else:
                        # Fallback to random text if the LLM call fails
                        messages.append({
                            "role": "assistant",
                            "content": await generate_random_text(random.randint(min_words, max_words))
                        })
            except Exception as e:
                print(f"Error generating assistant message: {e}")
                # Fallback to random text if there's an exception
                messages.append({
                    "role": "assistant",
                    "content": await generate_random_text(random.randint(min_words, max_words))
                })
        else:  # User's turn
            messages.append({
                "role": "user",
                "content": await generate_random_text(random.randint(min_words, max_words))
            })
    
    # Ensure the last message is from the user
    if messages[-1]["role"] != "user":
        messages.append({
            "role": "user",
            "content": await generate_random_text(random.randint(min_words, max_words))
        })
    
    return messages

async def generate_payload(task_type: str, temperature: float = 0.5, session: Optional[aiohttp.ClientSession] = None) -> Dict:
    """Generate a payload based on task type and temperature."""
    if session is None:
        # Create a session if none was provided
        async with aiohttp.ClientSession() as temp_session:
            return await generate_payload(task_type, temperature, temp_session)
    
    if task_type == "chat":
        n_messages = random.randint(MIN_CHAT_MESSAGES, MAX_CHAT_MESSAGES)
        messages = await generate_chat_messages(
            n_messages, 
            MIN_WORDS_PER_MESSAGE, 
            MAX_WORDS_PER_MESSAGE,
            session
        )
        
        return {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": random.randint(300, 500),
            "model": MODEL_CONFIG['model'],
            "top_p": 1,
            "stream": True,
            "logprobs": True,
            "seed": random.randint(1, 10000)
        }
    else:  # completion
        text = await generate_random_text(
            random.randint(COMPLETION_MIN_WORDS, COMPLETION_MAX_WORDS)
        )
        
        return {
            "prompt": text,
            "temperature": temperature,
            "seed": random.randint(1, 10000),
            "model": MODEL_CONFIG['model'],
            "stream": True,
            "logprobs": True,
            "top_p": 1.0,
            "max_tokens": random.randint(300, 500)
        }
    
def load_sse_jsons(chunk: str) -> List[Dict[str, Any]]:
    """Parse Server-Sent Events (SSE) format into JSON objects."""
    jsons = []
    pattern = r'data: ([\s\S]*?)(?=\n\ndata: |\n\n$|$)'
    matches = re.findall(pattern, chunk)
    
    for match in matches:
        data = match.strip()
        if data == "[DONE]":
            break
            
        try:
            data_escaped = data.replace('\n', '\\n')
            loaded_chunk = json.loads(data_escaped)
            jsons.append(loaded_chunk)
        except json.JSONDecodeError:
            try:
                for c, esc in [('\n', '\\n'), ('\r', '\\r'), ('\t', '\\t'), 
                               ('\b', '\\b'), ('\f', '\\f')]:
                    data = data.replace(c, esc)
                loaded_chunk = json.loads(data)
                jsons.append(loaded_chunk)
            except json.JSONDecodeError:
                print('error parsing JSON')
                exit(0)
    
    return jsons

async def query_miner(session: aiohttp.ClientSession, endpoint: str, payload: Dict) -> Dict:
    """Query the miner endpoint and collect streaming responses."""
    print(f"Querying miner at {endpoint}...")
    start_time = time.time()
    
    async with session.post(endpoint, json=payload) as response:
        if response.status != 200:
            error_text = await response.text()
            raise Exception(f"Error from miner: {response.status} - {error_text}")
        
        collected_chunks = []
        async for chunk in response.content:
            chunk_str = chunk.decode('utf-8')
            collected_chunks.append(chunk_str)
        
        full_response = "".join(collected_chunks)
        
    response_time = time.time() - start_time
    text_jsons = load_sse_jsons(full_response)
    
    return {
        "text_jsons": text_jsons,
        "response_time": response_time,
        "success": len(text_jsons) > 0
    }

async def construct_query_result(miner_result: Dict, task: str) -> Dict:
    """Construct a QueryResult object from the miner's response."""
    return {
        "formatted_response": miner_result["text_jsons"] if miner_result["text_jsons"] else None,
        "node_id": 1,
        "node_hotkey": "sample_hotkey",
        "response_time": miner_result["response_time"],
        "stream_time": miner_result["response_time"],
        "task": task,
        "success": miner_result["success"],
        "status_code": 200 if miner_result["success"] else 500
    }

async def submit_to_validator(session: aiohttp.ClientSession, base_url: str, 
                             query_result: Dict, server_config: Dict, payload: Dict) -> str:
    """Submit the miner's response to the validator for checking."""
    print(f"Submitting to validator...")
    endpoint = f"{base_url}/check-result"
    
    check_payload = {
        "server_config": server_config,
        "result": query_result,
        "payload": payload
    }
    
    async with session.post(endpoint, json=check_payload) as response:
        if response.status != 200:
            error_text = await response.text()
            raise Exception(f"Error from validator: {response.status} - {error_text}")
        
        result = await response.json()
        return result.get("task_id")

async def poll_validator_task(session: aiohttp.ClientSession, base_url: str, task_id: str) -> Dict:
    """Poll the validator for the task result."""
    print(f"Polling validator for task {task_id}...")
    endpoint = f"{base_url}/check-task/{task_id}"
    
    while True:
        async with session.get(endpoint) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Error polling validator: {response.status} - {error_text}")
            
            result = await response.json()
            status = result.get("status")
            
            if status == "Processing":
                print("Task still processing, waiting...")
                await asyncio.sleep(1)
                continue
            
            if status == "Success":
                return result.get("result")
            
            if status == "Failed":
                raise Exception(f"Validator task failed: {result}")
            
            raise Exception(f"Unknown task status: {status}")

async def run_single_test(session: aiohttp.ClientSession, temperature: float = 0.5) -> Dict:
    """Run a single test iteration."""
    payload = await generate_payload(TASK_TYPE, temperature)
    
    miner_endpoint = f"{MINER_BASE_URL}/v1/{'chat/' if TASK_TYPE == 'chat' else ''}completions"
    
    server_config = {
        "server_needed": ServerType.LLM,
        "load_model_config": MODEL_CONFIG,
        "checking_function": "check_text_result",
        "task": TASK_ID,
        "endpoint": "/v1/chat/completions" if TASK_TYPE == "chat" else "/v1/completions"
    }
    
    miner_result = await query_miner(session, miner_endpoint, payload)
    print(f"Miner response received. Success: {miner_result['success']}")
    
    query_result = await construct_query_result(miner_result, TASK_ID)
    
    task_id = await submit_to_validator(session, VALIDATOR_BASE_URL, query_result, server_config, payload)
    print(f"Validator task submitted. Task ID: {task_id}")
    
    validator_result = await poll_validator_task(session, VALIDATOR_BASE_URL, task_id)
    
    score = None
    if validator_result and validator_result.get("node_scores"):
        score = list(validator_result["node_scores"].values())[0]
    
    return {
        "payload": payload,
        "temperature": temperature,
        "miner_response": miner_result,
        "score": score,
        "validator_result": validator_result
    }

async def run_full_test(n_data_points: int):
    """Run a full test with multiple data points."""
    # Generate temperatures following a normal distribution between 0 and 1
    # We'll use a truncated normal distribution centered at 0.5 with std=0.25
    # and reject samples outside [0,1]
    temps = []
    while len(temps) < n_data_points:
        t = np.random.normal(0.5, 0.25)
        if 0 <= t <= 1:
            temps.append(t)
    
    results = []
    async with aiohttp.ClientSession() as session:
        for i, temperature in enumerate(temps):
            print(f"\n--- Running test {i+1}/{n_data_points} with temperature {temperature:.4f} ---")
            try:
                result = await run_single_test(session, temperature)
                
                if TASK_TYPE == "chat":
                    message_count = len(result["payload"]["messages"])
                    content_info = f"{message_count} messages"
                else:
                    word_count = len(result["payload"]["prompt"].split())
                    content_info = f"{word_count} words"
                
                results.append({
                    "temperature": temperature,
                    "score": result["score"],
                    "content_info": content_info,
                    "payload": json.dumps(result["payload"]),
                    "response": json.dumps(result["miner_response"]),
                })
                print(f"Test {i+1} completed. Score: {result['score']} | {content_info}")
            except Exception as e:
                print(f"Error in test {i+1}: {e}")
                results.append({
                    "temperature": temperature,
                    "score": None,
                    "content_info": None,
                    "payload": None,
                    "response": None,
                    "error": str(e)
                })
    
    df = pd.DataFrame(results)
    filename = f"test_results_{TASK_ID}_{int(time.time())}.csv"
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

async def main():
    """Main function to run tests based on configuration."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')
    
    if N_DATA_POINTS <= 1:
        async with aiohttp.ClientSession() as session:
            try:
                result = await run_single_test(session)
                print("\n=== TEST RESULTS ===")
                print(f"Temperature: {result['temperature']}")
                print(f"Score: {result['score']}")
                print("\nValidator Result:")
                print(json.dumps(result['validator_result'], indent=2))
            except Exception as e:
                print(f"Error: {e}")
    else:
        await run_full_test(N_DATA_POINTS)

if __name__ == "__main__":
    asyncio.run(main())