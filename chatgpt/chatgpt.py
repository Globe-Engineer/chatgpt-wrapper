import asyncio
import json
import os
import time
import hashlib
from datetime import datetime


import diskcache
import openai
from openai import OpenAIError, AsyncOpenAI, AsyncStream, Stream
from openai.types.chat import ChatCompletionMessage, ChatCompletion, ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParam


logs_dir = os.path.join(os.getcwd(), '.chatgpt_history/logs')
cache_dir = os.path.join(os.getcwd(), '.chatgpt_history/cache')
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

cache = diskcache.Cache(cache_dir)
def get_key(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()

def retry_on_exception(retries=5, initial_wait_time=1):
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                wait_time = initial_wait_time
                for attempt in range(retries):
                    try:
                        return await func(*args, **kwargs)
                    except OpenAIError as e:
                        if attempt == retries - 1:
                            raise e
                        print(e)
                        await asyncio.sleep(wait_time)
                        wait_time *= 2
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                wait_time = initial_wait_time
                for attempt in range(retries):
                    try:
                        return func(*args, **kwargs)
                    except OpenAIError as e:
                        if attempt == retries - 1:
                            raise e
                        print(e)
                        time.sleep(wait_time)
                        wait_time *= 2
            return sync_wrapper
    return decorator


@retry_on_exception()
def complete(messages:list[ChatCompletionMessageParam]=None, model='gpt-4', temperature=0, use_cache=False, **kwargs):
    if use_cache:
        key = get_key(messages)
        if key in cache:
            return cache.get(key)
    response: ChatCompletion | Stream[ChatCompletionChunk] = openai.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        **kwargs
    )
    stream = kwargs.get('stream', False)
    # import pdb; pdb.set_trace()
    if stream:
        n = kwargs.get('n', 1)
        return parse_stream(response, messages, n=n)
    return parse_response(response, messages, **kwargs)


@retry_on_exception()
async def acomplete(messages:list[ChatCompletionMessageParam]=None, model='gpt-4', temperature=0, use_cache=False, **kwargs):
    if use_cache:
        key = get_key(messages)
        if key in cache:
            return cache.get(key)
    client = AsyncOpenAI()
    response: ChatCompletion | AsyncStream[ChatCompletionChunk] = await client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        **kwargs
    )
    stream = kwargs.get('stream', False)
    if stream:
        n = kwargs.get('n', 1)
        return await parse_astream(response, messages, n=n)
    return await parse_response(response, messages, **kwargs)


def parse_response(response: ChatCompletion, messages:list[ChatCompletionMessageParam], **kwargs):
    n = kwargs.get('n', 1)


    results = []
    for choice in response.choices:
        message = choice.message
        if kwargs.get('functions', None) and message.function_call:
            name = message.function_call.name
            try:
                args = json.loads(message.function_call.arguments)
            except json.decoder.JSONDecodeError as e:
                print('ERROR: OpenAI returned invalid JSON for function call arguments')
                raise e
            # results.append({'role': 'function', 'name': name, 'args': args})
            # log_completion(messages, results[-1])
        results.append(message)
        log_completion(messages, message)

    output = results if n > 1 else results[0]
    cache.set(get_key(messages), output)
    return output

def parse_stream(response: Stream[ChatCompletionChunk], messages:list[ChatCompletionMessageParam], n=1):
    results = ['' for _ in range(n)]
    chunk: ChatCompletionChunk
    for chunk in response:
        for choice in chunk.choices:
            if not choice.delta:
                continue
            text = choice.delta.content
            if not text:
                continue
            idx = choice.index
            results[idx] += text
            if n == 1:
                yield text
            else:
                yield (text, idx)

    for r in results:
        log_completion(messages, r)
    cache.set(get_key(messages), results)

async def parse_astream(response: AsyncStream[ChatCompletionChunk], messages:list[ChatCompletionMessageParam], n=1):
    results = ['' for _ in range(n)]
    chunk: ChatCompletionChunk
    async for chunk in response:
        for choice in chunk.choices:
            if not choice.delta:
                continue
            text = choice.delta.content
            if not text:
                continue
            idx = choice.index
            results[idx] += text
            if n == 1:
                yield text
            else:
                yield (text, idx)

    for r in results:
        log_completion(messages, r)
    cache.set(get_key(messages), results)

def log_completion(messages: list[ChatCompletionMessageParam], result: ChatCompletionMessage = None):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')

    save_path = os.path.join(logs_dir, timestamp + '.txt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    log = ""
    for message in messages:
        log += message['role'].upper() + ' ' + '-'*100 + '\n'
        if "content" in message:
            log += 'Content:\n' + message['content'] + "\n"
        if "function_call" in message: # TODO: remove later since function_call is deprecated
            log += f'Call function\n:{message["function_call"]["name"]}({message["function_call"]["arguments"]})\n'
        if "tool_calls" in message:
            for tool in message["tool_calls"]:
                log += f'\nCall {tool["type"]}:\n'
                if tool["type"] == 'function':
                    log += f'{tool["function"]["name"]}({tool["function"]["arguments"]}) id={tool["id"]}\n'
                else:
                    raise NotImplementedError(f'Tool type {tool["type"]} not implemented in logger')
        log += '\n\n'


    if result:
        log += result.role.upper() + ' ' + '-'*100 + '\n'
        if result.content:
            log += 'Content:\n' + result['content']
        if result.function_call:
            log += f'Called function:\n{result.function_call.name}({result.function_call.arguments})\n'
        if result.tool_calls:
            for tool in result.tool_calls:
                log += f'\nCalled {tool.type}:\n'
                if tool.type == 'function':
                    log += f'{tool.function.name}({tool.function.arguments}) id={tool.id}\n'
                else:
                    raise NotImplementedError(f"Tool type {tool.type} not implemented in logger")

    log += '\n\n'

    with open(save_path, 'w') as f:
        f.write(log)
