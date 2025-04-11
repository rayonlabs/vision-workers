from app.core import models
import json
import random
from loguru import logger
import httpx
from typing import List
import math
from typing import Dict, Any, Tuple, Union
import copy


PROMPT_KEY = "prompt"
MESSAGES_KEY = "messages"

# TODO: Eventually change to chutes
BASE_URL = "http://llm_server:6919".rstrip("/")

BOTTOM_TEXT_THRESHOLD = 0.125
TOP_TEXT_THRESHOLD = 0.25


def _score_average_distance(average_distance: float) -> float:
    if average_distance <= BOTTOM_TEXT_THRESHOLD:
        return 1.0
    elif average_distance <= TOP_TEXT_THRESHOLD:
        return 1.0 - 0.5 * (average_distance - BOTTOM_TEXT_THRESHOLD) / (TOP_TEXT_THRESHOLD - BOTTOM_TEXT_THRESHOLD)
    return 0.0


def _payload_is_completions(payload: dict) -> bool:
    return PROMPT_KEY in payload


def _extract_completions_message(idx: int, response: dict) -> str:
    content = response["choices"][0]["text"]
    logprobs = response["choices"][0]["logprobs"]

    # Sometimes the first message is empty and doesn't have logprobs, skip it
    if idx == 0 and content == "" and logprobs is None:
        return None

    logprob = logprobs["token_logprobs"][0]
    return models.MessageResponse(content=content, logprob=logprob)


def _extract_chat_message(idx: int, response: dict) -> models.MessageResponse | None:
    content = response["choices"][0]["delta"]["content"]
    logprobs = response["choices"][0]["logprobs"]
    # Below is a fix for the first message not having logprobs
    if idx == 0 and content == "" and logprobs is None:
        role = response["choices"][0]["delta"]["role"]
        if role == "assistant":
            return None
    logprob = logprobs["content"][0]["logprob"]
    return models.MessageResponse(content=content, logprob=logprob)


async def _tokenize(prompt: str, model: str, add_special_tokens: bool) -> list[int]:
    async with httpx.AsyncClient() as client:
        r = await client.post(url=f"{BASE_URL}/tokenize", json={"model": model, "prompt": prompt, "add_special_tokens": add_special_tokens})
        r.raise_for_status()  # raise an exception for 4xx or 5xx status codes
        return r.json()["tokens"]


async def _detokenize(tokens: list[int], model: str):
    async with httpx.AsyncClient() as client:
        r = await client.post(url=f"{BASE_URL}/detokenize", json={"tokens": tokens, "model": model})
        r.raise_for_status()  # raise an exception for 4xx or 5xx status codes
        return r.json()["prompt"]


async def _tokenize_and_detokenize(input_payload: dict, model_name: str, eos_token_id: int = 128009, add_generation_prompt: bool = True) -> tuple[str, int]:
    async with httpx.AsyncClient() as http_client:
        logger.info(f"Tokenizing at: {BASE_URL}/tokenize")
        tokenize_response = await http_client.post(url=f"{BASE_URL}/tokenize", json=input_payload)
        tokenize_response.raise_for_status()
        token_list: list[int] = tokenize_response.json()["tokens"]

        if ("llama-3" in model_name.lower() or 'deepseek-r1' in model_name.lower() or 'qwq-32b' in model_name.lower() or 'qwen2.5-7b' in model_name.lower()) and not add_generation_prompt:
            last_eot_index = max((index for index, value in enumerate(token_list) if value == eos_token_id), default=None)
            if last_eot_index is not None:
                token_list = token_list[:last_eot_index]

        detokenize_response = await http_client.post(url=f"{BASE_URL}/detokenize", json={"tokens": token_list, "model": model_name})
        detokenize_response.raise_for_status()
        prompt = detokenize_response.json()["prompt"]
        return prompt, len(token_list)


async def _chat_to_prompt(messages: list[dict], model_name: str, eos_token_id: int = 128009, add_generation_prompt: bool = True) -> tuple[str, int]:
    input_payload = {"model": model_name, "messages": messages, "add_special_tokens": False}
    return await _tokenize_and_detokenize(input_payload, model_name, eos_token_id, add_generation_prompt)


async def _completions_to_prompt(prompt: str, model_name: str, eos_token_id: int = 128009, add_generation_prompt: bool = True) -> tuple[str, int]:
    input_payload = {"model": model_name, "prompt": prompt}
    return await _tokenize_and_detokenize(input_payload, model_name, eos_token_id, add_generation_prompt)


async def make_api_call(
    payload: dict,
    endpoint: str,
) -> dict:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(endpoint, json=payload)
        return response.json()


async def query_endpoint_with_status(
    endpoint: str,
    data: Dict[str, Any],
    base_url: str = "http://llm_server:6919"
) -> Tuple[Union[Dict[str, Any], None], int]:
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    
    async with httpx.AsyncClient(timeout=20) as client:
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
        
    # TODO: in future if upgrading from vllm 0.6.3, remember to set `add_special_tokens = False` due to "second bos" issue
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

    logger.info(f"chat completion payload: \n{json.dumps(chat_completions_payload, indent=2)}\n")
    logger.info(f"validator_checking_response: \n{json.dumps(validator_checking_response, indent=2)}\n")
    logger.info(f"chat_responses: \n{json.dumps([response.dict() for response in chat_responses[max(0, index-5):index+3]], indent=2)}\n")
    logger.info(f"focus token in response: \n{json.dumps(chat_responses[index].dict(), indent=2)}\n")

    text = chat_responses[index].content

    #updated acc to output from /chat/completions with VLLM_USE_V1=0
    validator_log_probs_for_token = validator_checking_response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]

    # find entry in top_logprobs where `token` == text, updated acc to output from /chat/completions with VLLM_USE_V1=0
    found_entry = next((entry for entry in validator_log_probs_for_token if entry["token"] == text), None)

    # Handle case of prompt_logprobs outputting |eot| token but miner response including "" output
    if is_last_token and must_output_eos:
        if text != "":
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
        distance = min(abs(math.exp(found_entry["logprob"]) - math.exp(chat_responses[index].logprob)), 1)
        logger.info(f"token: {text} - logprob : {chat_responses[index].logprob}")
        logger.info(f"validator_log_probs_for_token: {validator_log_probs_for_token}")

    return distance


async def calculate_distance_for_token(
    task_config: models.OrchestratorServerConfig,
    llm_request: Union[models.ChatRequestModel, models.CompletionRequestModel],
    chat_responses: List[models.MessageResponse],
    index: int,
    starting_assistant_message: bool,
) -> float:
    if isinstance(llm_request, models.ChatRequestModel):
        messages = [elm.model_dump() for elm in llm_request.messages]
        prompt, _ = await _chat_to_prompt(
            messages=messages,
            model_name=task_config.load_model_config["model"],
            eos_token_id=task_config.load_model_config.get("eos_token_id", 128009),
            add_generation_prompt=starting_assistant_message,
        )
        if 'deepseek-r1' in task_config.load_model_config['model'].lower():
            prompt = await _process_think_tags_deepseek(prompt, messages)
            
    elif isinstance(llm_request, models.CompletionRequestModel):
        prompt = llm_request.prompt

    # TODO: in future if upgrading from vllm 0.6.3, remember to set `add_special_tokens = False` due to "second bos" issue
    completions_payload = {
        "prompt": prompt,
        "model": task_config.load_model_config["model"],
        "temperature": llm_request.temperature,
        "top_p": 1,
        "max_tokens": 1,
        "logprobs": 20,
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

    logger.info(f"completion payload: \n{json.dumps(completions_payload, indent=2)}\n")
    logger.info(f"validator_checking_response: \n{json.dumps(validator_checking_response, indent=2)}\n")
    logger.info(f"chat_responses: \n{json.dumps([response.dict() for response in chat_responses[max(0, index-5):index+3]], indent=2)}\n")
    logger.info(f"focus token in response: \n{json.dumps(chat_responses[index].dict(), indent=2)}\n")

    text = chat_responses[index].content
    validator_log_probs_for_token = validator_checking_response["choices"][0]["logprobs"]["top_logprobs"][0]

    if text not in validator_log_probs_for_token:
        logger.info(f"token: {text} - not found in vali logprobs")
        logger.info(f"validator_log_probs_for_token: {validator_log_probs_for_token}")
        return 1
    else:
        distance = min(abs(math.exp(validator_log_probs_for_token[text]) - math.exp(chat_responses[index].logprob)), 1)
        logger.info(f"token: {text} - logprob : {chat_responses[index].logprob}")
        logger.info(f"validator_log_probs_for_token: {validator_log_probs_for_token}")

    return distance


async def check_text_result(result: models.QueryResult, payload: dict, task_config: models.OrchestratorServerConfig) -> Union[float, None]:
    # check fail
    if result.formatted_response is None:
        miner_status_code = result.status_code
        _, vali_status_code = await query_endpoint_with_status(task_config.endpoint, payload)
        logger.info(f"miner status code: {miner_status_code} - vali status code : {vali_status_code}")
        if str(vali_status_code[0]) == str(miner_status_code[0]):
            return 1
        else:
            return -3
    
    formatted_response = json.loads(result.formatted_response) if isinstance(result.formatted_response, str) else result.formatted_response
    eos_token_id = task_config.load_model_config.get("eos_token_id", 128009)

    # Extract messages & logprobs from the response
    messages: list[models.MessageResponse] = []
    is_completions_payload = _payload_is_completions(payload)
    for idx, response in enumerate(formatted_response):
        try:
            # If `prompt` is in the payload, treat it as a /completions request
            if is_completions_payload:
                message = _extract_completions_message(idx, response)
            else:
                message = _extract_chat_message(idx, response)

            if message is not None:
                messages.append(message)
        except Exception as e:
            logger.error(f"Error with logprob: {e}. Response: {response}")
            logger.exception(e)
            return 0  # Important to return 0 as this is a critical error

    if not messages:
        logger.error("No valid messages in response.")
        logger.exception(formatted_response)
        return 0.0

    if len(messages) > payload["max_tokens"]:
        logger.error("Number of messages is greater than max_tokens, skipping logprob check, returning 0")
        return 0.0

    full_response_content = "".join([message.content for message in messages])
    number_of_output_tokens = len(messages)

    # Now get the combined input + output in `prompt` format
    if is_completions_payload:
        input_completions_content = payload[PROMPT_KEY]

        input_tokens = await _tokenize(input_completions_content, task_config.load_model_config["model"], add_special_tokens=False)
        input_content, num_input_tokens = input_completions_content, len(input_tokens)
        eos_token = await _detokenize([eos_token_id], task_config.load_model_config["model"])

        if number_of_output_tokens != payload["max_tokens"] and messages[-1] != eos_token:
            full_prompt = input_content + full_response_content + eos_token
        else:    
            full_prompt = input_content + full_response_content

        all_tokens = await _tokenize(full_prompt, task_config.load_model_config["model"], add_special_tokens=False)

    else:
        input_chat_content = payload[MESSAGES_KEY]
        input_content, num_input_tokens = await _chat_to_prompt(input_chat_content, task_config.load_model_config["model"], eos_token_id, add_generation_prompt=True)
        full_prompt_before_eos = input_content + full_response_content
        if 'deepseek-r1' in task_config.load_model_config["model"].lower():
            all_tokens = await _tokenize(full_prompt_before_eos, task_config.load_model_config["model"], add_special_tokens=False)
        else:    
            all_tokens = await _tokenize(full_prompt_before_eos, task_config.load_model_config["model"], add_special_tokens=True)

        # Make sure the last token is eos token where necessary, so we can check it with prompt logprobs
        if number_of_output_tokens != payload["max_tokens"] and all_tokens[-1] != eos_token_id:
            all_tokens.append(eos_token_id)

        full_prompt = await _detokenize(all_tokens, task_config.load_model_config["model"])

    # Now get the prompt logprobs from completions and check they are all correct
    # TODO: in future if upgrading from vllm 0.6.3, remember to set `add_special_tokens = False` due to "second bos" issue
    completions_payload = {
        "prompt": full_prompt,
        "model": task_config.load_model_config["model"],
        "temperature": payload["temperature"],
        "max_tokens": 1,
        "prompt_logprobs": 10,
        "add_special_tokens": False
    }

    logger.info(f"completions_payload for checks: \n{json.dumps(completions_payload, indent=2)}\n")

    try:
        result = await make_api_call(completions_payload, endpoint=f"{BASE_URL}/v1/completions")
    except (httpx.RequestError, json.JSONDecodeError) as e:
        logger.exception(e)
        logger.error(f"API call failed: {e}")
        return 0.9876

    prompt_logprobs = result["choices"][0]["prompt_logprobs"][num_input_tokens:]

    bad_token_found = False

    fail_reason = ""

    failed_tokens_idx = []
    failed_tokens_details = []

    max_acceptable_rank = 10 if payload["temperature"] <= 0.5 else int(10 / (1.03 - payload["temperature"]))

    for idx, response_token, logprobs in zip(range(len(all_tokens[num_input_tokens:])), all_tokens[num_input_tokens:], prompt_logprobs):
        # Just a helper for nicer printing
        nice_logprobs = json.dumps(logprobs, indent=2, sort_keys=True, ensure_ascii=False)

        # The edge case here is when the messages didn't include the end of token
        # So sometimes we don't have a message for the last token
        additional_log = f" (decoded: '{messages[idx].content}', logprob: {messages[idx].logprob})" if idx <= len(messages) - 1 else ""

        if str(response_token) in logprobs:
            logprob = logprobs[str(response_token)]["logprob"]
            rank = logprobs[str(response_token)]["rank"]

            if rank <  max_acceptable_rank and logprob > float("-inf"):
                logger.info(f"Token {response_token} {additional_log} in logprobs with good behaviour; rank: {rank}, logprob: {logprob} ✅")
            else:
                logger.error(f"Token {response_token} {additional_log} in logprobs with bad behaviour; rank: {rank}, logprob: {logprob} ❌")
                failed_tokens_idx.append(idx)
                failed_tokens_details.append((response_token, rank, logprob, additional_log))

                if len(failed_tokens_idx) > 5:
                    failed_tokens_details = json.dumps(failed_tokens_details, indent=2, sort_keys=True, ensure_ascii=False)
                    fail_reason = f"Too many bad tokens found ('response_token', 'rank', 'logprob', 'additional_log'):\n{failed_tokens_details}"
                    bad_token_found = True
                    break
        else:
            logger.error(f"Token {response_token} {additional_log} not found in logprobs :(")
            bad_token_found = True
            break

        # If you could've stopped, why didnt you?
        if str(eos_token_id) in logprobs and str(response_token) != str(eos_token_id):
            logprob = logprobs[str(eos_token_id)]["logprob"]
            response_logprob = logprobs[str(response_token)]["logprob"]
            if logprob > float("-inf") and math.exp(logprob) / math.exp(response_logprob) > 100:
                fail_reason = "You really went out your way to avoid stopping!"
                bad_token_found = True
                break

    if bad_token_found:
        # TODO: Make a nice message
        logger.error(f"Bad token (s) found at indexes {failed_tokens_idx}." f" Prompt logprobs: {nice_logprobs}" f" Reason: {fail_reason}")
        return 0.0

    logger.info("All tokens found in prompt_logprobs! ✅")

    # Now lets do some fine grained checking

    if len(messages) == 1:
        indices_to_check = [0]
    else:
        # Always check first & last
        indices_to_check = [0, len(messages) - 1] 

        if len(failed_tokens_idx)>0:
            indices_to_check += failed_tokens_idx[:3]

        remaining_indexes = list(set(list(set(range(0, len(messages))) - set(indices_to_check))))

        number_of_additional_indices_to_check = min(5 - len(indices_to_check), len(messages) - 2) 
        additional_indices_to_check = random.sample(
            remaining_indexes,
            number_of_additional_indices_to_check,
        )
        indices_to_check.extend(additional_indices_to_check)

    logger.info(f"failed token indexes : {failed_tokens_idx}")
    logger.info(f"logprobs indexes to check : {indices_to_check}")

    total_distance = 0
    checks = 0

    # Prepare request for token validation
    payload["starting_assistant_message"] = True
    payload["number_of_logprobs"] = 5

    if is_completions_payload:
        llm_request = models.CompletionRequestModel(**payload)
        llm_request.max_tokens = 1
    else:
        llm_request = models.ChatRequestModel(**payload)
        llm_request.max_tokens = 1

    for i, index in enumerate(indices_to_check):
        if checks >= 5:
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
        distance = await calculate_distance_for_token(task_config, llm_request, messages, index, starting_assistant_message)
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
    score = _score_average_distance(average_distance)
    return score




async def check_vlm_result(result: models.QueryResult, payload: dict, task_config: models.OrchestratorServerConfig) -> Union[float, None]:
    # check fail
    if result.formatted_response is None:
        miner_status_code = result.status_code
        _, vali_status_code = await query_endpoint_with_status(task_config.endpoint, payload)
        logger.info(f"miner status code: {miner_status_code} - vali status code : {vali_status_code}")
        if str(vali_status_code[0]) == str(miner_status_code[0]):
            return 1
        else:
            return -3
    
    formatted_response = json.loads(result.formatted_response) if isinstance(result.formatted_response, str) else result.formatted_response
    eos_token_id = task_config.load_model_config.get("eos_token_id", 128009)
    assistant_token_id = task_config.load_model_config.get("assistant_token_id", 200019)

    # Extract messages & logprobs from the response
    messages: list[models.MessageResponse] = []
    for idx, response in enumerate(formatted_response):
        try:
            message = _extract_chat_message(idx, response)

            if message is not None:
                messages.append(message)
        except Exception as e:
            logger.error(f"Error with logprob: {e}. Response: {response}")
            logger.exception(e)
            return 0  # Important to return 0 as this is a critical error

    if not messages:
        logger.error("No valid messages in response.")
        logger.exception(formatted_response)
        return 0.0

    if len(messages) > payload["max_tokens"]:
        logger.error("Number of messages is greater than max_tokens, skipping logprob check, returning 0")
        return 0.0

    full_response_content = "".join([message.content for message in messages])
    number_of_output_tokens = len(messages)

    input_chat_content = payload[MESSAGES_KEY]

    must_output_eos = False
    eos_token = await _detokenize([eos_token_id], task_config.load_model_config["model"])
    # Make sure the last token is eos token where necessary, so we can check it with prompt logprobs
    if number_of_output_tokens != payload["max_tokens"] and full_response_content[-len(eos_token):] != eos_token:
        full_response_content += eos_token
        must_output_eos = True

    input_chat_content_w_response = input_chat_content.copy()
    input_chat_content_w_response.append({"role": "assistant", "content": full_response_content})

    # Now get the prompt logprobs from completions and check they are all correct
    # TODO: in future if upgrading from vllm 0.6.3, remember to set `add_special_tokens = False` due to "second bos" issue
    chat_completions_payload = {
        "messages": input_chat_content_w_response,
        "model": task_config.load_model_config["model"],
        "temperature": payload["temperature"],
        "max_tokens": 1,
        "add_generation_prompt": False,
        "continue_final_message": True,
        "prompt_logprobs": 10,
        "add_special_tokens": False
    }

    logger.info(f"chat_completions_payload for checks: \n{json.dumps(chat_completions_payload, indent=2)}\n")

    try:
        result = await make_api_call(chat_completions_payload, endpoint=f"{BASE_URL}/v1/chat/completions")
    except (httpx.RequestError, json.JSONDecodeError) as e:
        logger.exception(e)
        logger.error(f"API call failed: {e}")
        return 0.9876
    


    #TODO: calculate num_input_tokens with /chat/completions input that also has image input (which gets ignored by /tokenize endpoint, a bit tricky yea)

    # Get the last occurance of the assistant token in the prompt logprobs
    prompt_logprobs = result["prompt_logprobs"] #referring to vllm /chat/completions output body with prompt_logprobs
    last_assistant_index = None
    for i in range(len(prompt_logprobs) - 1, -1, -1):
        d = prompt_logprobs[i]
        if str(assistant_token_id) in d and d[str(assistant_token_id)]["rank"] == 1:
            last_assistant_index = i
            break

    try:
        assert last_assistant_index is not None
    except AssertionError:
        logger.error(f"Assistant token not found in prompt logprobs of miner : {prompt_logprobs}")
        return 0    
    prompt_logprobs = prompt_logprobs[last_assistant_index + 1:]

    bad_token_found = False

    fail_reason = ""

    failed_tokens_idx = []
    failed_tokens_details = []

    max_acceptable_rank = 10 if payload["temperature"] <= 0.5 else int(10 / (1.03 - payload["temperature"]))

    for idx, response_obj, logprobs in zip(range(len(messages)), messages, prompt_logprobs):
        # Just a helper for nicer printing
        nice_logprobs = json.dumps(logprobs, indent=2, sort_keys=True, ensure_ascii=False)

        # The edge case here is when the messages didn't include the end of token
        # So sometimes we don't have a message for the last token
        additional_log = f" (decoded: '{messages[idx].content}', logprob: {messages[idx].logprob})" if idx <= len(messages) - 1 else ""

        # Check if the decoded token (miner output) is in the corresponding prompt logprobs
        found_key = next((token_id for d in prompt_logprobs for token_id, info in d.items() if info["decoded_token"] == response_obj.content), None)

        if found_key:
            logprob = logprobs[found_key]["logprob"]
            rank = logprobs[found_key]["rank"]
            if rank <  max_acceptable_rank and logprob > float("-inf"):
                logger.info(f"Token {response_obj.content} {additional_log} in logprobs with good behaviour; rank: {rank}, logprob: {logprob} ✅")
            else:
                logger.error(f"Token {response_obj.content} {additional_log} in logprobs with bad behaviour; rank: {rank}, logprob: {logprob} ❌")
                failed_tokens_idx.append(idx)
                failed_tokens_details.append((response_obj.content, rank, logprob, additional_log))

                if len(failed_tokens_idx) > 5:
                    failed_tokens_details = json.dumps(failed_tokens_details, indent=2, sort_keys=True, ensure_ascii=False)
                    fail_reason = f"Too many bad tokens found ('response_token', 'rank', 'logprob', 'additional_log'):\n{failed_tokens_details}"
                    bad_token_found = True
                    break
        else:
            logger.error(f"Token {response_obj.content} {additional_log} not found in logprobs :(")
            bad_token_found = True
            break

        # If you could've stopped, why didnt you?
        if str(eos_token_id) in logprobs and response_obj.content != str(eos_token_id):
            logprob = logprobs[str(eos_token_id)]["logprob"]
            response_logprob = logprobs[found_key]["logprob"]
            if logprob > float("-inf") and math.exp(logprob) / math.exp(response_logprob) > 100:
                fail_reason = "You really went out your way to avoid stopping!"
                bad_token_found = True
                break

    if bad_token_found:
        # TODO: Make a nice message
        logger.error(f"Bad token (s) found at indexes {failed_tokens_idx}." f" Prompt logprobs: {nice_logprobs}" f" Reason: {fail_reason}")
        return 0.0

    logger.info("All tokens found in prompt_logprobs! ✅")

    # Now lets do some fine grained checking

    if len(messages) == 1:
        indices_to_check = [0]
    else:
        # Always check first & last
        indices_to_check = [0, len(messages) - 1] 

        if len(failed_tokens_idx)>0:
            indices_to_check += failed_tokens_idx[:3]

        remaining_indexes = list(set(list(set(range(0, len(messages))) - set(indices_to_check))))

        number_of_additional_indices_to_check = min(5 - len(indices_to_check), len(messages) - 2) 
        additional_indices_to_check = random.sample(
            remaining_indexes,
            number_of_additional_indices_to_check,
        )
        indices_to_check.extend(additional_indices_to_check)

    logger.info(f"failed token indexes : {failed_tokens_idx}")
    logger.info(f"logprobs indexes to check : {indices_to_check}")

    total_distance = 0
    checks = 0

    # Prepare request for token validation
    payload["starting_assistant_message"] = True
    payload["number_of_logprobs"] = 5


    llm_request = models.ChatRequestModel(**payload)
    llm_request.max_tokens = 1
    llm_request.messages = copy.deepcopy(payload[MESSAGES_KEY])
    logger.info(f"payload[MESSAGES_KEY] : {payload[MESSAGES_KEY]}")
    logger.info(f"llm_request.messages : {llm_request.messages}")

    for i, index in enumerate(indices_to_check):
        if checks >= 5:
            break
        starting_assistant_message = i == 0
        if index > 0:
            text_to_inject_into_assistant_message = "".join([i.content for i in messages[:index]])
            llm_request.messages.append(
                {
                    "role": "assistant",
                    "content": text_to_inject_into_assistant_message,
                }
            )
        logger.info(f"index : {index} - token : {messages[index].content}; llm_request.messages : {llm_request.messages}")
        distance = await calculate_distance_for_token_vlm(task_config, llm_request, messages, index, starting_assistant_message, index == (len(messages) - 1), must_output_eos)
        checks += 1
        total_distance += distance
        llm_request.messages = copy.deepcopy(payload[MESSAGES_KEY])

    try:
        average_distance = total_distance / checks
    except Exception as e:
        logger.error(f"Error with average distance: {e}. Total distance: {total_distance}. Checks: {checks}")
        return 0
    score = _score_average_distance(average_distance)
    return score
