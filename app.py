import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variables
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Initialize the OpenAI client with the API key
client = OpenAI(
  base_url="https://integrate.api.nvidia.com/v1",
  api_key=NVIDIA_API_KEY
)

# Create a completion request
completion = client.chat.completions.create(
  model="meta/llama3-70b-instruct",
  messages=[{"role": "user", "content": "Provide me an article on Machine Learning"}],
  temperature=0.5,
  top_p=1,
  max_tokens=1024,
  stream=True
)

# Print the response in a streaming fashion
for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")
