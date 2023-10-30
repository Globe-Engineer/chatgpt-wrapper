
# Globe's ChatGPT Wrapper

This repository introduces a Python wrapper around OpenAI's `ChatCompletion.create` function, offering additional features to streamline and enhance the usage of the GPT models. The primary features include automatic caching, user-readable logging, exponential backoff on errors, and more Pythonic outputs.

## Installation

You can install the Globe ChatGPT Wrapper using pip:

```bash
pip install chatgpt-wrapper
```

## Features

The Globe ChatGPT Wrapper enhances the standard OpenAI API by providing:

- **Automatic Caching:** Responses from the model are automatically cached for faster retrieval of repeated queries.
- **User-readable Logging:** The wrapper logs interactions in an easily readable format, keeping track of all inputs and outputs.
- **Exponential Backoff:** In case of errors, the wrapper implements an exponential backoff strategy to handle retries gracefully.
- **Pythonic Outputs:** The wrapper ensures outputs are structured in a more Python-friendly manner, making it easier to work with the responses.

## Usage

Here are some snippets demonstrating how to use the Globe ChatGPT Wrapper:

### Basic Chat Completion

```python
from chatgpt_wrapper import complete

response = complete(messages=[{'role': 'user', 'content': 'Hello!'}])
print(response)
```

### Asynchronous Chat Completion

For asynchronous applications:

```python
from chatgpt_wrapper import acomplete
import asyncio

async def async_chat():
    response = await acomplete(messages=[{'role': 'user', 'content': 'Hello!'}])
    print(response)

asyncio.run(async_chat())
```

### Using Cache

Utilizing caching is the biggest feature of this library. If a call has been made with the exact same inputs before, you'll get an instant response and no actual openai call. This is perfect for devs, saves a ton of time, and money!

```python
response = complete(
    messages=[{'role': 'user', 'content': 'What's the weather like?'}],
    model='gpt-4',
    use_cache=True
)
print(response)
```

### Auto-Retry

Internally, the complete and acomplete currently have an ``retry_on_exception`` decorator thats also part of the library. We will expose it soon.

## Contributing

Your contributions are welcome! Please feel free to fork the repository, make your changes, and submit a pull request. For any queries or discussions, join our [discord](https://discord.gg/79WH83sS3M) community.

## Acknowledgements

Special thanks to @ivan_yevenko and @sincethestudy for their contributions and insights. Share your experiences and code with us on [Twitter](https://twitter.com/ivan_yevenko) or join our hackathons to explore more about AI agent programming.

**Note:** This wrapper is an open-source project and is not officially affiliated with OpenAI. It is designed to simplify and enhance the usage of OpenAI's GPT models for the community.
