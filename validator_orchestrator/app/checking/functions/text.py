from app.core import models
import json
import random
from loguru import logger
import httpx
from typing import List, Dict, Any, Tuple, Union
import math
import copy

BASE_URL = "http://llm_server:6919".rstrip("/")

PROMPT_KEY = "prompt"
MESSAGES_KEY = "messages"

BOTTOM_TEXT_THRESHOLD = 0.125
TOP_TEXT_THRESHOLD = 0.25

HTTP_TIMEOUT = 60
SHORT_TIMEOUT = 20

MAX_FAILED_TOKENS = 5
MAX_CHECKS = 5

DEFAULT_NUM_LOGPROBS = 20
DEFAULT_PROMPT_LOGPROBS = 1 #vllm 0.8.3 breaks at some arbitrary prompt_logprobs > 1: https://github.com/vllm-project/vllm/issues/16836 

MODELS_FOR_EOT_HANDLING = ["llama-3", "deepseek-r1", "qwq-32b", "qwen2.5-7b", "qwen3", "mistral-nemo"]


def _score_average_distance(average_distance: float) -> float:
    if average_distance <= BOTTOM_TEXT_THRESHOLD:
        return 1.0
    elif average_distance <= TOP_TEXT_THRESHOLD:
        return 1.0 - 0.5 * (average_distance - BOTTOM_TEXT_THRESHOLD) / (TOP_TEXT_THRESHOLD - BOTTOM_TEXT_THRESHOLD)
    return 0.0


def _payload_is_completions(payload: dict) -> bool:
    return PROMPT_KEY in payload


def _extract_completions_message(idx: int, response: dict) -> models.MessageResponse:
    content = response["choices"][0]["text"]
    logprobs = response["choices"][0]["logprobs"]

    if idx == 0 and content == "" and logprobs is None:
        return None

    text_token = logprobs["tokens"][0]
    logprob = logprobs["token_logprobs"][0]
    if content in logprobs["top_logprobs"][0]:
        logprob = logprobs["top_logprobs"][0][content]

    return models.MessageResponse(content=content, logits=models.LogitResponse(text_token=text_token, logprob=logprob))


def _extract_chat_message(idx: int, response: dict) -> Union[models.MessageResponse, None]:
    content = response["choices"][0]["delta"]["content"]
    logprobs = response["choices"][0]["logprobs"]
    
    if idx == 0 and content == "" and logprobs is None:
        role = response["choices"][0]["delta"]["role"]
        if role == "assistant":
            return None
            
    text_token = logprobs["content"][0]["token"]
    logprob = logprobs["content"][0]["logprob"]
    return models.MessageResponse(content=content, logits=models.LogitResponse(text_token=text_token, logprob=logprob))


async def _tokenize(prompt: str, model: str, add_special_tokens: bool) -> list[int]:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            url=f"{BASE_URL}/tokenize", 
            json={"model": model, "prompt": prompt, "add_special_tokens": add_special_tokens}
        )
        r.raise_for_status()
        return r.json()["tokens"]


async def _detokenize(tokens: list[int], model: str) -> str:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            url=f"{BASE_URL}/detokenize", 
            json={"tokens": tokens, "model": model}
        )
        r.raise_for_status()
        return r.json()["prompt"]


async def _tokenize_and_detokenize(input_payload: dict, model_name: str, eos_token_id: int, add_generation_prompt: bool = True) -> tuple[str, int]:
    async with httpx.AsyncClient() as http_client:
        logger.info(f"Tokenizing at: {BASE_URL}/tokenize")
        tokenize_response = await http_client.post(url=f"{BASE_URL}/tokenize", json=input_payload)
        tokenize_response.raise_for_status()
        token_list: list[int] = tokenize_response.json()["tokens"]

        if any(model in model_name.lower() for model in MODELS_FOR_EOT_HANDLING) and not add_generation_prompt:
            last_eot_index = max((index for index, value in enumerate(token_list) if value == eos_token_id), default=None)
            if last_eot_index is not None:
                token_list = token_list[:last_eot_index]

        detokenize_response = await http_client.post(
            url=f"{BASE_URL}/detokenize", 
            json={"tokens": token_list, "model": model_name}
        )
        detokenize_response.raise_for_status()
        prompt = detokenize_response.json()["prompt"]
        return prompt, len(token_list)


async def _chat_to_prompt(messages: list[dict], model_name: str, eos_token_id: int, add_generation_prompt: bool = True) -> tuple[str, int]:
    input_payload = {"model": model_name, "messages": messages, "add_special_tokens": False}
    return await _tokenize_and_detokenize(input_payload, model_name, eos_token_id, add_generation_prompt)


async def _completions_to_prompt(prompt: str, model_name: str, eos_token_id: int, add_generation_prompt: bool = True) -> tuple[str, int]:
    input_payload = {"model": model_name, "prompt": prompt}
    return await _tokenize_and_detokenize(input_payload, model_name, eos_token_id, add_generation_prompt)


async def make_api_call(payload: dict, endpoint: str) -> dict:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.post(endpoint, json=payload)
        return response.json()


async def query_endpoint_with_status(
    endpoint: str,
    data: Dict[str, Any],
    base_url: str = BASE_URL
) -> Tuple[Union[Dict[str, Any], None], int]:
    url = f"{base_url.rstrip('/')}/v1/{endpoint.lstrip('/')}"
    
    async with httpx.AsyncClient(timeout=SHORT_TIMEOUT) as client:
        logger.info(f"Querying: {url}")
        try:
            response = await client.post(url, json=data)
            logger.info(f"Status code: {response.status_code}")
            
            if response.status_code >= 400:
                return None, response.status_code
                
            try:
                response_data = response.json()
                if isinstance(response_data, str):
                    response_data = json.loads(response_data)
                return response_data, response.status_code
                
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON response: {e}")
                return None, response.status_code
                
        except httpx.HTTPError as e:
            status = getattr(e, 'response', None)
            status_code = status.status_code if status else 500
            logger.error(f"HTTP error occurred: {e}")
            return None, status_code


async def _process_think_tags_deepseek(prompt: str, messages: list[dict]) -> str:
    assistant_message = next((m for m in messages if m['role'] == 'assistant'), None)
    if not assistant_message:
        return prompt
        
    content = assistant_message['content']
    think_start = content.find('<think>')
    think_end = content.find('</think>')
    
    if think_start == -1 or think_end == -1:
        return prompt
        
    think_content = content[think_start:think_end + 8]
    
    assistant_token = '<｜Assistant｜>'
    insert_pos = prompt.find(assistant_token) + len(assistant_token)
    
    return prompt[:insert_pos] + think_content + prompt[insert_pos:]


async def calculate_distance_for_token_vlm(
    task_config: models.OrchestratorServerConfig,
    llm_request: models.ChatRequestModel,
    chat_responses: List[models.MessageResponse],
    index: int,
    starting_assistant_message: bool,
    is_last_token: bool,
    must_output_eos: bool
) -> float:

    messages = llm_request.messages
    messages = [msg.model_dump() if hasattr(msg, 'model_dump') else msg for msg in messages]

    chat_completions_payload = {
        "messages": messages,
        "model": task_config.load_model_config["model"],
        "temperature": llm_request.temperature,
        "top_p": 1,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20,
        "add_generation_prompt": starting_assistant_message,
        "continue_final_message": not starting_assistant_message,
        "add_special_tokens": False
    }
    try:
        validator_checking_response = await make_api_call(chat_completions_payload, endpoint=f"{BASE_URL}/v1/chat/completions")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON in calculate_distance_for_token_vlm: {e}. Response: {validator_checking_response}")
        return 1
    except httpx.RequestError as e:
        logger.error(f"Request failed in calculate_distance_for_token_vlm: {e}")
        return 1


    # tweaked when upgrading vllm 0.6.4.post1 to 0.8.5.post1 due to `\u0120` being present in upgraded version, https://gist.github.com/tripathiarpan20/f742eacd9f462656c10ce202bbb86dde
    # text = chat_responses[index].content
    text = chat_responses[index].logits.text_token

    # VLLM_USE_V1=0
    validator_log_probs_for_token = validator_checking_response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]

    # VLLM_USE_V1=0
    found_entry = next((entry for entry in validator_log_probs_for_token if entry["token"] == text), None)

    # Handle case of prompt_logprobs outputting |eot| token but miner response including "" output
    if is_last_token and must_output_eos:
        if chat_responses[index].content != "":
            logger.info(f"token: `{text}` at last index doesn't correspond to eos "" output by miners")
            logger.info(f"validator_log_probs_for_token: {validator_log_probs_for_token}")
            return 1
        eos_token = validator_checking_response["choices"][0]["logprobs"]["content"][0]["token"]
        found_entry = next((entry for entry in validator_log_probs_for_token if entry["token"] == eos_token), None)

    if not found_entry:
        logger.info(f"token: {text} - not found in vali logprobs")
        logger.info(f"validator_log_probs_for_token: {validator_log_probs_for_token}")
        return 1
    else:
        distance = min(abs(math.exp(found_entry["logprob"]) - math.exp(chat_responses[index].logits.logprob)), 1)

    return distance

async def calculate_distance_for_token(
    task_config: models.OrchestratorServerConfig,
    llm_request: Union[models.ChatRequestModel, models.CompletionRequestModel],
    chat_responses: List[models.MessageResponse],
    index: int,
    starting_assistant_message: bool,
) -> float:
    model_name = task_config.load_model_config["model"]
    eos_token_id = task_config.load_model_config["eos_token_id"]
    
    if isinstance(llm_request, models.ChatRequestModel):
        messages = [elm.model_dump() for elm in llm_request.messages]
        prompt, _ = await _chat_to_prompt(
            messages=messages,
            model_name=model_name,
            eos_token_id=eos_token_id,
            add_generation_prompt=starting_assistant_message,
        )
        if 'deepseek-r1' in model_name.lower():
            prompt = await _process_think_tags_deepseek(prompt, messages)
            
    elif isinstance(llm_request, models.CompletionRequestModel):
        prompt = llm_request.prompt

    completions_payload = {
        "prompt": prompt,
        "model": model_name,
        "temperature": llm_request.temperature,
        "top_p": llm_request.top_p,
        "top_k": -1,
        "max_tokens": 1,
        "logprobs": DEFAULT_NUM_LOGPROBS,
        "add_special_tokens": False
    }
    
    try:
        validator_checking_response = await make_api_call(completions_payload, endpoint=f"{BASE_URL}/v1/completions")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON in calculate_distance_for_token: {e}. Response: {validator_checking_response}")
        return 1
    except httpx.RequestError as e:
        logger.error(f"Request failed in calculate_distance_for_token: {e}")
        return 1


    # tweaked when upgrading vllm 0.6.4.post1 to 0.8.5.post1 due to `\u0120` being present in upgraded version, https://gist.github.com/tripathiarpan20/f742eacd9f462656c10ce202bbb86dde
    #text = chat_responses[index].content
    text = chat_responses[index].logits.text_token

    validator_log_probs_for_token = validator_checking_response["choices"][0]["logprobs"]["top_logprobs"][0]

    if text not in validator_log_probs_for_token:
        logger.info(f"token: {chat_responses[index].content} (parsed as {text}) - not found in vali logprobs")
        logger.info(f"validator_log_probs_for_token: {validator_log_probs_for_token}")
        return 1
    else:
        distance = min(abs(math.exp(validator_log_probs_for_token[text]) - math.exp(chat_responses[index].logits.logprob)), 1)

    return distance


async def _extract_messages(formatted_response, is_completions_payload) -> List[models.MessageResponse]:
    messages = []
    for idx, response in enumerate(formatted_response):
        try:
            if is_completions_payload:
                message = _extract_completions_message(idx, response)
            else:
                message = _extract_chat_message(idx, response)

            if message is not None:
                messages.append(message)
        except Exception as e:
            logger.error(f"Error with logprob: {e}. Response: {response}")
            logger.exception(e)
            return None
    return messages


async def _get_full_prompt(is_completions_payload, payload, messages, task_config, full_response_content):
    model_name = task_config.load_model_config["model"]
    eos_token_id = task_config.load_model_config['eos_token_id']
    number_of_output_tokens = len(messages)
    
    if is_completions_payload:
        input_completions_content = payload[PROMPT_KEY]
        input_tokens = await _tokenize(input_completions_content, model_name, add_special_tokens=False)
        num_input_tokens = len(input_tokens)
        eos_token = await _detokenize([eos_token_id], model_name)

        if number_of_output_tokens != payload["max_tokens"] and messages[-1] != eos_token:
            full_prompt = input_completions_content + full_response_content + eos_token
        else:    
            full_prompt = input_completions_content + full_response_content

        all_tokens = await _tokenize(full_prompt, model_name, add_special_tokens=False)
        return full_prompt, all_tokens, num_input_tokens

    else:
        input_chat_content = payload[MESSAGES_KEY]
        input_content, num_input_tokens = await _chat_to_prompt(
            input_chat_content, 
            model_name, 
            eos_token_id, 
            add_generation_prompt=True
        )
        full_prompt_before_eos = input_content + full_response_content
        
        if 'deepseek-r1' in model_name.lower():
            all_tokens = await _tokenize(full_prompt_before_eos, model_name, add_special_tokens=False)
        else:    
            all_tokens = await _tokenize(full_prompt_before_eos, model_name, add_special_tokens=True)

        if number_of_output_tokens != payload["max_tokens"] and all_tokens[-1] != eos_token_id:
            all_tokens.append(eos_token_id)

        full_prompt = await _detokenize(all_tokens, model_name)
        return full_prompt, all_tokens, num_input_tokens



#----

async def _detect_repetitive_patterns(messages: List[models.MessageResponse], min_pattern_length: int = 10, max_repetitions: int = 3) -> tuple[bool, str]:
    """
    Detect repetitive patterns in the response content.
    
    Args:
        messages: List of message responses
        min_pattern_length: Minimum length of pattern to consider
        max_repetitions: Maximum allowed repetitions before flagging
        
    Returns:
        (is_repetitive, reason)
    """
    full_content = "".join([msg.content for msg in messages])
    
    # Check for exact substring repetitions
    for pattern_len in range(min_pattern_length, min(len(full_content) // 2, 100)):
        for start_pos in range(len(full_content) - pattern_len):
            pattern = full_content[start_pos:start_pos + pattern_len]
            
            # Skip patterns that are just whitespace or single characters
            if len(set(pattern.strip())) <= 1:
                continue
                
            count = 0
            pos = 0
            while True:
                pos = full_content.find(pattern, pos)
                if pos == -1:
                    break
                count += 1
                pos += pattern_len
                
            if count > max_repetitions:
                return True, f"Pattern '{pattern[:50]}...' repeated {count} times (max allowed: {max_repetitions})"
    
    # Check for similar sentence-level repetitions
    sentences = full_content.split('.')
    if len(sentences) > 5:
        sentence_counts = {}
        for sentence in sentences:
            cleaned = sentence.strip().lower()
            if len(cleaned) > 20:  # Only check substantial sentences
                sentence_counts[cleaned] = sentence_counts.get(cleaned, 0) + 1
                
        for sentence, count in sentence_counts.items():
            if count > max_repetitions:
                return True, f"Sentence repeated {count} times: '{sentence[:100]}...'"
    
    return False, ""


async def _analyze_token_distribution(messages: List[models.MessageResponse], prompt_logprobs: List[Dict[str, Any]]) -> tuple[bool, str]:
    """
    Analyze the distribution of token probabilities to detect anomalous patterns.
    
    Returns:
        (is_anomalous, reason)
    """
    if len(prompt_logprobs) < 10:  # Need sufficient tokens for analysis
        return False, ""
    
    # Extract ranks and logprobs
    ranks = []
    logprobs = []
    
    for i, (message, logprob_data) in enumerate(zip(messages, prompt_logprobs)):
        token = message.logits.text_token
        
        # Find the token in logprobs
        token_found = False
        for token_id, token_info in logprob_data.items():
            if token_info.get("decoded_token") == token or str(token_id) == str(token):
                ranks.append(token_info["rank"])
                logprobs.append(token_info["logprob"])
                token_found = True
                break
        
        if not token_found:
            ranks.append(float('inf'))  # Worst possible rank
            logprobs.append(float('-inf'))  # Worst possible logprob
    
    # Check for too many high-rank (unlikely) tokens
    high_rank_count = sum(1 for rank in ranks if rank > 100)
    high_rank_ratio = high_rank_count / len(ranks)
    
    if high_rank_ratio > 0.3:  # More than 30% of tokens are very unlikely
        return True, f"Too many unlikely tokens: {high_rank_count}/{len(ranks)} ({high_rank_ratio:.2%}) have rank > 100"
    
    # Check for too many very low probability tokens
    very_low_prob_count = sum(1 for lp in logprobs if lp < -10)
    very_low_prob_ratio = very_low_prob_count / len(logprobs)
    
    if very_low_prob_ratio > 0.4:  # More than 40% of tokens have very low probability
        return True, f"Too many very low probability tokens: {very_low_prob_count}/{len(logprobs)} ({very_low_prob_ratio:.2%}) have logprob < -10"
    
    # Check for suspicious patterns in consecutive tokens
    consecutive_bad_tokens = 0
    max_consecutive_bad = 0
    
    for rank in ranks:
        if rank > 50:  # Consider rank > 50 as "bad"
            consecutive_bad_tokens += 1
            max_consecutive_bad = max(max_consecutive_bad, consecutive_bad_tokens)
        else:
            consecutive_bad_tokens = 0
    
    if max_consecutive_bad > 20:  # Too many consecutive unlikely tokens
        return True, f"Too many consecutive unlikely tokens: {max_consecutive_bad} tokens in a row with rank > 50"
    
    return False, ""


async def _validate_tokens(
    out_tokens: List, 
    messages: List[models.MessageResponse], 
    prompt_logprobs: List[Dict[str, Any]], 
    payload: Dict, 
    eos_token_id: int
) -> tuple[bool, str, List[int]]:
    """
    Enhanced token validation with repetition detection and distribution analysis.
    """
    failed_tokens_idx = []
    failed_tokens_details = []
    max_acceptable_rank = 5
    
    # First, check for repetitive patterns
    is_repetitive, repetition_reason = await _detect_repetitive_patterns(messages)
    if is_repetitive:
        return True, f"Repetitive content detected: {repetition_reason}", []
    
    # Analyze token distribution for anomalies
    is_anomalous, anomaly_reason = await _analyze_token_distribution(messages, prompt_logprobs)
    if is_anomalous:
        return True, f"Anomalous token distribution: {anomaly_reason}", []
    
    logger.info(f"prompt_logprobs : {json.dumps(prompt_logprobs[:2], indent=2)}")
    
    # Original token validation logic with enhanced criteria
    suspicious_token_count = 0
    total_rank_penalty = 0
    
    for idx, response_token, logprobs in zip(range(len(out_tokens)), out_tokens, prompt_logprobs):
        nice_logprobs = json.dumps(logprobs, indent=2, sort_keys=True, ensure_ascii=False)
        additional_log = f" (decoded: '{messages[idx].content}', logprob: {messages[idx].logits.logprob})" if idx <= len(messages) - 1 else ""

        if str(response_token) in logprobs:
            logprob = logprobs[str(response_token)]["logprob"]
            rank = logprobs[str(response_token)]["rank"]
            
            # Enhanced criteria for token acceptance
            is_token_acceptable = True
            failure_reasons = []
            
            # Original rank and logprob checks
            if rank >= max_acceptable_rank:
                is_token_acceptable = False
                failure_reasons.append(f"rank too high ({rank} >= {max_acceptable_rank})")
                
            if logprob <= float("-inf"):
                is_token_acceptable = False
                failure_reasons.append("logprob is -inf")
            
            # Additional checks for suspicious tokens
            if rank > 100:  # Very unlikely token
                suspicious_token_count += 1
                total_rank_penalty += rank
                
            if rank > 1000:  # Extremely unlikely token
                is_token_acceptable = False
                failure_reasons.append(f"extremely unlikely token (rank {rank})")
                
            if logprob < -15:  # Very low probability
                is_token_acceptable = False
                failure_reasons.append(f"very low probability (logprob {logprob})")
            
            if not is_token_acceptable:
                logger.error(f"Token {response_token} {additional_log} failed validation: {', '.join(failure_reasons)} ❌")
                failed_tokens_idx.append(idx)
                failed_tokens_details.append((response_token, rank, logprob, additional_log, failure_reasons))

                if len(failed_tokens_idx) > MAX_FAILED_TOKENS:
                    failed_tokens_details_str = json.dumps(failed_tokens_details, indent=2, sort_keys=True, ensure_ascii=False)
                    fail_reason = f"Too many bad tokens found ('response_token', 'rank', 'logprob', 'additional_log', 'reasons'):\n{failed_tokens_details_str}"
                    return True, fail_reason, failed_tokens_idx
        else:
            logger.error(f"Token {response_token} {additional_log} not found in logprobs :(")
            return True, "Token not found in logprobs", failed_tokens_idx

        # Enhanced EOS token handling
        if str(eos_token_id) in logprobs and str(response_token) != str(eos_token_id):
            eos_logprob = logprobs[str(eos_token_id)]["logprob"]
            response_logprob = logprobs[str(response_token)]["logprob"]
            
            if response_logprob == float("-inf"):
                return True, "Token with -inf probability chosen over EOS", failed_tokens_idx
            
            if eos_logprob > float("-inf"):
                try:
                    if response_logprob > float("-inf"):
                        prob_ratio = math.exp(eos_logprob) / math.exp(response_logprob)
                        # More strict threshold for stopping avoidance
                        if prob_ratio > 50:  # Reduced from 100
                            return True, f"Avoided stopping when EOS was {prob_ratio:.2f}x more likely", failed_tokens_idx
                    else:
                        pass

                except (ZeroDivisionError, OverflowError):
                    logger.warning("Math error when comparing token probabilities")
                    return True, "Math error in probability comparison", failed_tokens_idx
    
    # Check overall suspicious token ratio
    if len(messages) > 20:  # Only for longer responses
        suspicious_ratio = suspicious_token_count / len(messages)
        avg_rank_penalty = total_rank_penalty / len(messages) if len(messages) > 0 else 0
        
        if suspicious_ratio > 0.5:  # More than 50% suspicious tokens
            return True, f"Too many suspicious tokens: {suspicious_token_count}/{len(messages)} ({suspicious_ratio:.2%}) have rank > 100", failed_tokens_idx
            
        if avg_rank_penalty > 200:  # Average rank penalty too high
            return True, f"Average token rank too high: {avg_rank_penalty:.1f} (indicating poor overall quality)", failed_tokens_idx

    return False, "", failed_tokens_idx



#----



async def _perform_token_checks(task_config, payload, messages, indices_to_check):
    total_distance = 0
    checks = 0
    is_completions_payload = _payload_is_completions(payload)

    if is_completions_payload:
        llm_request = models.CompletionRequestModel(**payload)
        llm_request.max_tokens = 1
    else:
        llm_request = models.ChatRequestModel(**payload)
        llm_request.max_tokens = 1

    for i, index in enumerate(indices_to_check):
        if checks >= MAX_CHECKS:
            break

        if is_completions_payload:
            text_to_inject_for_checking = "".join([i.content for i in messages[:index]])
            llm_request.prompt += text_to_inject_for_checking
            starting_assistant_message = False
        else:
            starting_assistant_message = i == 0
            if index > 0:
                text_to_inject_into_assistant_message = "".join([i.content for i in messages[:index]])
                llm_request.messages.append(
                    models.Message(
                        **{
                            "role": "assistant",
                            "content": text_to_inject_into_assistant_message,
                        }
                    )
                )
                
        logger.info(f"index : {index} - token : {messages[index].content}")
        distance = await calculate_distance_for_token(
            task_config, 
            llm_request, 
            messages, 
            index, 
            starting_assistant_message
        )
        checks += 1
        total_distance += distance
        
        if index != 0 and is_completions_payload:
            llm_request.prompt = llm_request.prompt[: (len(llm_request.prompt) - len(text_to_inject_for_checking))]
        elif index != 0 and not is_completions_payload:
            llm_request.messages = llm_request.messages[:-1]

    try:
        average_distance = total_distance / checks
    except Exception as e:
        logger.error(f"Error with average distance: {e}. Total distance: {total_distance}. Checks: {checks}")
        return 0
        
    return _score_average_distance(average_distance)


async def check_text_result(result: models.QueryResult, payload: dict, task_config: models.OrchestratorServerConfig) -> Union[float, None]:
    if result.formatted_response is None:
        miner_status_code = result.status_code
        payload["model"] = task_config.load_model_config["model"]
        _, vali_status_code = await query_endpoint_with_status(task_config.endpoint, payload)
        logger.info(f"miner status code: {miner_status_code} - vali status code : {vali_status_code}")
        if type(vali_status_code) is int:
            return 1 if str(vali_status_code) == str(miner_status_code) else -3
        else:
            return 1 if str(vali_status_code[0]) == str(miner_status_code[0]) else -3
    
    formatted_response = json.loads(result.formatted_response) if isinstance(result.formatted_response, str) else result.formatted_response
    is_completions_payload = _payload_is_completions(payload)
    
    messages = await _extract_messages(formatted_response, is_completions_payload)
    if not messages:
        logger.error("No valid messages in response.")
        return 0.0

    if len(messages) > payload["max_tokens"]:
        logger.error("Number of messages is greater than max_tokens, skipping logprob check, returning 0")
        return 0.0

    full_response_content = "".join([message.content for message in messages])
    
    full_prompt, all_tokens, num_input_tokens = await _get_full_prompt(
        is_completions_payload, 
        payload, 
        messages, 
        task_config, 
        full_response_content
    )

    completions_payload = {
        "prompt": full_prompt,
        "model": task_config.load_model_config["model"],
        "temperature": payload["temperature"],
        "max_tokens": 1,
        "logprobs": True,
        "prompt_logprobs": DEFAULT_PROMPT_LOGPROBS,
        "add_special_tokens": False,
        "top_k": -1,
        "top_p": 1,
        "stream": False
    }

    logger.info(f"completions_payload for checks: \n{json.dumps(completions_payload, indent=2)}\n")

    try:
        result = await make_api_call(completions_payload, endpoint=f"{BASE_URL}/v1/completions")
    except Exception as e:
        logger.exception(e)
        logger.error(f"API call failed: {e}")
        raise e

    prompt_logprobs = result["choices"][0]["prompt_logprobs"][num_input_tokens:]
    
    bad_token_found, fail_reason, failed_tokens_idx = await _validate_tokens(
        all_tokens[num_input_tokens:],
        messages, 
        prompt_logprobs, 
        payload,
        task_config.load_model_config['eos_token_id']
    )
    
    if bad_token_found:
        nice_logprobs = json.dumps(prompt_logprobs, indent=2, sort_keys=True, ensure_ascii=False)
        logger.error(f"Bad token (s) found at indexes {failed_tokens_idx}." f" Prompt logprobs: {nice_logprobs}" f" Reason: {fail_reason}")
        return 0.0

    logger.info("All tokens found in prompt_logprobs! ✅")

    indices_to_check = [0, len(messages) - 1] if len(messages) > 1 else [0]
    
    if failed_tokens_idx:
        indices_to_check += failed_tokens_idx[:3]

    remaining_indexes = list(set(range(0, len(messages) - 1)) - set(indices_to_check))
    number_of_additional_indices_to_check = min(5 - len(indices_to_check), len(messages) - 2)
    
    if remaining_indexes and number_of_additional_indices_to_check > 0:
        additional_indices_to_check = random.sample(
            remaining_indexes,
            number_of_additional_indices_to_check,
        )
        indices_to_check.extend(additional_indices_to_check)

    logger.info(f"failed token indexes : {failed_tokens_idx}")
    logger.info(f"logprobs indexes to check : {indices_to_check}")

    payload["starting_assistant_message"] = True
    
    return await _perform_token_checks(task_config, payload, messages, indices_to_check)

async def check_vlm_result(result: models.QueryResult, payload: dict, task_config: models.OrchestratorServerConfig) -> Union[float, None]:
    if result.formatted_response is None:
        miner_status_code = result.status_code
        payload["model"] = task_config.load_model_config["model"]
        _, vali_status_code = await query_endpoint_with_status(task_config.endpoint, payload)
        logger.info(f"miner status code: {miner_status_code} - vali status code : {vali_status_code}")
        if type(vali_status_code) is int:
            return 1 if str(vali_status_code) == str(miner_status_code) else -3
        else:
            return 1 if str(vali_status_code[0]) == str(miner_status_code[0]) else -3
    
    formatted_response = json.loads(result.formatted_response) if isinstance(result.formatted_response, str) else result.formatted_response
    
    messages = await _extract_messages(formatted_response, is_completions_payload=False)
    if not messages:
        logger.error("No valid messages in response.")
        return 0.0

    if len(messages) > payload["max_tokens"]:
        logger.error("Number of messages is greater than max_tokens, skipping logprob check, returning 0")
        return 0.0

    full_response_content = "".join([message.content for message in messages])
    eos_token_id = task_config.load_model_config.get("eos_token_id", 128009)
    
    input_chat_content = payload[MESSAGES_KEY]
    must_output_eos = False
    eos_token = await _detokenize([eos_token_id], task_config.load_model_config["model"])
    
    if len(messages) != payload["max_tokens"] and full_response_content[-len(eos_token):] != eos_token:
        full_response_content += eos_token
        must_output_eos = True

    out_tokens = await _tokenize(full_response_content, task_config.load_model_config["model"], add_special_tokens=False)

    input_chat_content_w_response = input_chat_content.copy()
    input_chat_content_w_response.append({"role": "assistant", "content": full_response_content})

    chat_completions_payload = {
        "messages": input_chat_content_w_response,
        "model": task_config.load_model_config["model"],
        "temperature": payload["temperature"],
        "max_tokens": 1,
        "add_generation_prompt": False,
        "continue_final_message": True,
        "prompt_logprobs": DEFAULT_PROMPT_LOGPROBS,
        "add_special_tokens": False,
        "logprobs": True,
        "top_k": -1,
        "top_p": 1,
        "stream": False
    }

    logger.info(f"chat_completions_payload for checks: \n{json.dumps(chat_completions_payload, indent=2)}\n")

    try:
        result = await make_api_call(chat_completions_payload, endpoint=f"{BASE_URL}/v1/chat/completions")
    except Exception as e:
        logger.exception(e)
        logger.error(f"API call failed: {e}")
        raise e
    
    n_generated_tokens = len(messages)
    
    prompt_logprobs = result["prompt_logprobs"][-n_generated_tokens:]
    logger.info(f"prompt_logprobs : {json.dumps(prompt_logprobs[:2], indent=2)}")
    
    bad_token_found, fail_reason, failed_tokens_idx = await _validate_tokens(
        out_tokens,
        messages, 
        prompt_logprobs, 
        payload,
        task_config.load_model_config['eos_token_id']
    )
    
    if bad_token_found:
        nice_logprobs = json.dumps(prompt_logprobs, indent=2, sort_keys=True, ensure_ascii=False)
        logger.error(f"Bad token (s) found at indexes {failed_tokens_idx}." f" Prompt logprobs: {nice_logprobs}" f" Reason: {fail_reason}")
        return 0.0

    logger.info("All tokens found in prompt_logprobs! ✅")
    
    indices_to_check = [0, len(messages) - 1] if len(messages) > 1 else [0]
    
    # if failed_tokens_idx:
    #     indices_to_check += failed_tokens_idx[:3]
        
    remaining_indexes = list(set(range(0, len(messages) - 1)) - set(indices_to_check))
    number_of_additional_indices_to_check = min(MAX_CHECKS - len(indices_to_check), len(messages) - 2)
    
    if remaining_indexes and number_of_additional_indices_to_check > 0:
        additional_indices_to_check = random.sample(
            remaining_indexes,
            number_of_additional_indices_to_check,
        )
        indices_to_check.extend(additional_indices_to_check)
        
    logger.info(f"failed token indexes: {failed_tokens_idx}")
    logger.info(f"logprobs indexes to check: {indices_to_check}")
    
    total_distance = 0
    checks = 0
    
    llm_request = models.ChatRequestModel(**payload)
    llm_request.max_tokens = 1
    
    for i, index in enumerate(indices_to_check):
        if checks >= MAX_CHECKS:
            break
            
        starting_assistant_message = i == 0
        if index > 0:
            text_to_inject_into_assistant_message = "".join([i.content for i in messages[:index]])
            llm_request.messages.append(
                models.Message(
                    **{
                        "role": "assistant",
                        "content": text_to_inject_into_assistant_message,
                    }
                )
            )
            
        logger.info(f"index: {index} - token: {messages[index].content}")
        distance = await calculate_distance_for_token_vlm(
            task_config, 
            llm_request, 
            messages, 
            index, 
            starting_assistant_message,
            index == (len(messages) - 1),
            must_output_eos
        )
        checks += 1
        total_distance += distance
        
        if index != 0:
            llm_request.messages = llm_request.messages[:-1]
            
    try:
        average_distance = total_distance / checks
    except Exception as e:
        logger.error(f"Error with average distance: {e}. Total distance: {total_distance}. Checks: {checks}")
        return 0
        
    return _score_average_distance(average_distance)
