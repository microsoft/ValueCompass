import os
import xlrd
import time
import openai
import random
import requests

import pandas as pd
# from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage

MAX_RETRY_TIMES = 5
    
def call_openai_gpt(prompt, model_name="gpt-4", key="jing"):
    openai.api_base = "https://api.openai.com/v1"
    openai.api_key = "your_openai_api_key"
    message = [{"role": "user", "content": prompt}]
    max_tokens = 2048
    if model_name == "gpt-35-turbo":
        model_name = "gpt-3.5-turbo"
        max_tokens = 1024
    response = openai.ChatCompletion.create(
        model = model_name,
        messages = message,
        temperature = 0.,
        max_tokens = max_tokens,
        request_timeout = 200,
    )
    answer = response["choices"][0]["message"]["content"].strip()

    return answer