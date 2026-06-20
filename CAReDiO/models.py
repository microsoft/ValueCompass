import os
import sys
import time
import json
from openai import OpenAI, AzureOpenAI
from google import genai
from google.genai import types
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from azure.identity import get_bearer_token_provider, AzureCliCredential, ChainedTokenCredential, ManagedIdentityCredential

import torch
import transformers
from transformers import (
    # pipeline,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    BitsAndBytesConfig,
    StoppingCriteriaList,
    AutoProcessor, 
    Gemma3ForConditionalGeneration
)
from peft import AutoPeftModelForCausalLM

API_TIMEOUTS = [1, 2, 4, 8, 16, 32]

MODELS = {
    "trapi/gpt-5.2-reasoning": {
        "model_class": "TRAPIModel",
        "model_name": "gpt-5.2_2025-12-11", # $1.25 / 1M input tokens, $10.00 / 1M output tokens
    },
    "trapi/gpt-5.3-chat": {
        "model_class": "TRAPIModel",
        "model_name": "gpt-5.3-chat_2026-03-03",
    },
    "trapi/gpt-5.5": {
        "model_class": "TRAPIModel",
        "model_name": "gpt-5.5_2026-04-24", # Error code: 503 - {'status': 503, 'error': 'All backends for this model are temporarily unavailable due to health issues. Retry after 1 seconds. See (https://aka.ms/trapi/errors) for more.', 'code': 'all-backends-unhealthy'}
    },
    "trapi/qwen3.5-397b": {
        "model_class": "TRAPIModel",
        "model_name": "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4",
    },
    "trapi/grok-4.20": {
        "model_class": "TRAPIModel",
        "model_name": "grok-4-20-reasoning_1",
    },
    "trapi/kimi-k2.6": {
        "model_class": "TRAPIModel",
        "model_name": "Kimi-K2.6_2026-04-20"
    },
    "trapi/deepseek-v4-pro": {
        "model_class": "TRAPIModel",
        "model_name": "DeepSeek-V4-Pro_2026-04-23",
    },


    "openai/gpt-5.2": {
        "model_class": "AzureOpenAIModel", # "OpenAIModel",
        "model_name": "gpt-5.2", # $0.8 / 1M input tokens, $6.0 / 1M output tokens
    },

    "deepseek/deepseek-v3.2": {
        "model_class": "AzureDeepSeekModel",
        "model_name": "DeepSeek-V3.2", # "deepseek-chat", #"deepseek-reasoner", #"deepseek-chat",
    },

    "qwen/qwen3-8b": {
        "model_class": "QwenOpenModel",
        "model_name": "Qwen/Qwen3-8B",
    },
}

def create_aoai_client():
    scope = "api://trapi/.default"
    host = os.environ.get("TRAPI_HOST", "https://trapi.research.microsoft.com")
    endpoint = os.environ.get("OPENAI_BASE_URL", f"{host}/redmond/interactive/openai/v1")

    # Authentication
    credential = get_bearer_token_provider(ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential()), scope)

    # Client Initialization
    client = OpenAI(
        base_url=endpoint,
        api_key=credential,  # callable, auto-refreshes per request
    )
    return client

    # credential = AzureCliCredential()
    # token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
    # client = AzureOpenAI( 
    #     api_version="2024-12-01-preview", # "2024-12-01-preview",
    #     azure_endpoint="https://jing-east-us-2.openai.azure.com/",
    #     azure_ad_token_provider=token_provider,
    #     max_retries=5, 
    # )
    # return client

def create_model(model_name, args=None):
    """
    Initialize the model based on the model_name and return it
    """
    if model_name in MODELS:
        class_name = MODELS[model_name]["model_class"]
        cls = getattr(sys.modules[__name__], class_name)
        print(f"Model {model_name} is loaded successfully")
        return cls(model_name, args)
    
    raise ValueError(f"Unknown Model '{model_name}'")

class LanguageModel:
    """ Generic LanguageModel Class"""

    def __init__(self, model_name, args = None):
        assert model_name in MODELS, f"Model {model_name} is not supported!"

        # Set some default model variables
        self._model_name = MODELS[model_name]["model_name"]

    def get_top_p_answer(self, prompt_base: str, prompt_system: str, max_tokens: int, temperature: float, top_p: float,) -> str:
        """
        Gets answer using sampling (based on top_p and temperature)

        :param prompt_base:     base prompt
        :param prompt_sytem:    system instruction for chat endpoint of OpenAI
        :param max_tokens       max tokens in answer
        :param temperature      temperature for top_p sampling
        :param top_p            top_p parameter
        :return:                answer string
        """
    def get_batch_top_p_answer(self, prompts_base: list[str], prompts_system: list[str], max_tokens: int, temperature: float, top_p: float) -> list[str]:
        responses = [None] * len(prompts_base)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self.get_top_p_answer, prompts_base[idx], prompts_system[idx], max_tokens, temperature, top_p): idx for idx in range(len(prompts_base))}

            for future in as_completed(futures):
                idx = futures[future] # this ensures that the order of responses matches the order of prompts
                responses[idx] = future.result()
        return responses
    
    def get_batch_message_answer(self, messages: list[list[dict[str, str]]], max_tokens: int, temperature: float, top_p: float) -> list[str]:
        responses = [None] * len(messages)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self.get_message_answer, msg, max_tokens, temperature, top_p): idx for idx, msg in enumerate(messages)}

            for future in as_completed(futures):
                idx = futures[future]
                responses[idx] = future.result()
        return responses

class AzureOpenAIModel(LanguageModel):
    """Azure OpenAI API Wrapper"""
    def __init__(self, model_name: str, args: dict):
        super().__init__(model_name, args)
        assert MODELS[model_name]["model_class"] == "AzureOpenAIModel", (
            f"Errorneous Model Instatiation for {model_name}"
        )
        credential = AzureCliCredential()
        token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
        self.aoiclient = AzureOpenAI( 
            # azure_endpoint="https://societalllm.openai.azure.com/",
            azure_endpoint="https://jing-east-us-2.openai.azure.com/",
            azure_ad_token_provider=token_provider, 
            api_version="2025-01-01-preview", # "2024-12-01-preview", 
            max_retries=5, 
        )
    def get_message_answer(self, messages: list[dict[str, str]], max_tokens: int, temperature: float, top_p: float) -> str:
        if "gpt-5" in self._model_name:
            response = self.aoiclient.chat.completions.create(
                model = self._model_name,
                messages = messages,
                max_completion_tokens = max_tokens + 5000,
                temperature = 1.0,
                reasoning_effort = "medium",
            )
        else:
            response = self.aoiclient.chat.completions.create(
                model = self._model_name,
                messages = messages,
                max_tokens = max_tokens,
                temperature = temperature,
                top_p = top_p,
            )
        return response.choices[0].message.content.strip()

    def get_top_p_answer(self, prompt_base: str, prompt_system: str, max_tokens: int, temperature: float, top_p: float,) -> str:
        success = False
        t = 0
        while not success and t < len(API_TIMEOUTS):
            try:
                messages = []
                if prompt_system != "":
                    messages.append({"role": "system", "content": prompt_system})
                messages.append({"role": "user", "content": prompt_base})

                if "gpt-5" in self._model_name:
                    response = self.aoiclient.chat.completions.create(
                        model = self._model_name,
                        messages = messages,
                        max_completion_tokens = max_tokens + 5000,
                        temperature = 1.0,
                        # top_p = top_p,
                        reasoning_effort="medium",
                    )
                else:
                    response = self.aoiclient.chat.completions.create(
                        model = self._model_name,
                        messages = messages,
                        max_tokens = max_tokens,
                        temperature = temperature,
                        top_p = top_p,
                    )

                success = True
            except:
                time.sleep(API_TIMEOUTS[t])
                t = t + 1
            
        if not success:
            print(f"Failed to get response for prompt")
            return ""
        if response is None or response.choices[0].message.content is None:
            print(f"Response is None for prompt")
            return ""
        return response.choices[0].message.content.strip()

class TRAPIModel(LanguageModel):
    """TRAPI API Wrapper - inherits AzureOpenAIModel since TRAPI endpoint is compatible with Azure OpenAI API"""
    def __init__(self, model_name: str, args: dict):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "TRAPIModel", (
            f"Errorneous Model Instatiation for {model_name}"
        )
        # Override the aoiclient with TRAPI client
        # Configuration
        scope = "api://trapi/.default"
        deployment_name = os.environ.get("OPENAI_MODEL", "gpt-5.2-chat_2025-12-11")
        host = os.environ.get("TRAPI_HOST", "https://trapi.research.microsoft.com")
        endpoint = os.environ.get("OPENAI_BASE_URL", f"{host}/redmond/interactive/openai/v1")

        # Authentication
        credential = get_bearer_token_provider(ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential()), scope)

        # Client Initialization
        self.client = OpenAI(
            base_url=endpoint,
            api_key=credential,  # callable, auto-refreshes per request
        )
    def get_top_p_answer(self, prompt_base: str, prompt_system: str, max_tokens: int, temperature: float, top_p: float,) -> str:
        success = False
        t = 0
        while not success and t < len(API_TIMEOUTS):
            # try:
            messages = []
            if prompt_system != "":
                messages.append({"role": "system", "content": prompt_system})
            messages.append({"role": "user", "content": prompt_base})

            reasoning_effort = "none" if "no-reasoning" in self._model_name else "medium"
            response = self.client.chat.completions.create(
                model = self._model_name,
                messages = messages,
                max_completion_tokens = max_tokens + 5000,
                temperature = 1.0,
                reasoning_effort=reasoning_effort,
            )
            success = True
            # except:
            #     time.sleep(API_TIMEOUTS[t])
            #     t = t + 1
            
        if not success:
            print(f"Failed to get response for prompt")
            return ""
        if response is None or isinstance(response, str) or response.choices[0].message.content is None:
            print(f"Response is None for prompt")
            return ""
        return response.choices[0].message.content.strip()

    def get_message_answer(self, messages: list[dict[str, str]], max_tokens: int, temperature: float, top_p: float) -> str:
        success = False
        t = 0
        while not success and t < len(API_TIMEOUTS):
            try:
                response = self.client.chat.completions.create(
                    model = self._model_name,
                    messages = messages,
                    max_completion_tokens = max_tokens + 5000,
                    temperature = 1.0,
                    reasoning_effort="medium",
                )
                success = True
            except:
                time.sleep(API_TIMEOUTS[t])
                t = t + 1
        
        if not success:
            print(f"Failed to get response for prompt")
            return ""
        if response is None or isinstance(response, str) or response.choices[0].message.content is None:
            print(f"Response is None for prompt")
            return ""
        return response.choices[0].message.content.strip()

class AzureDeepSeekModel(LanguageModel):
    def __init__(self, model_name: str, args: dict):
        super().__init__(model_name, args)
        assert MODELS[model_name]["model_class"] == "AzureDeepSeekModel", (
            f"Errorneous Model Instatiation for {model_name}"
        )
        credential = AzureCliCredential()
        token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
        self.aoiclient = AzureOpenAI( 
            azure_endpoint="https://sc-ol-m2y79f1q-japaneast.services.ai.azure.com" if "v3.2" in model_name.lower() and "speciale" not in model_name.lower() else "https://jing-foundry.services.ai.azure.com",
            azure_ad_token_provider=token_provider, 
            api_version="2024-12-01-preview",
            max_retries=5, 
        )

    def get_top_p_answer(self, prompt_base: str, prompt_system: str, max_tokens: int, temperature: float, top_p: float,) -> str:
        success = False
        t = 0
        while not success and t < len(API_TIMEOUTS):
            try:
                messages = []
                if prompt_system != "":
                    messages.append({"role": "system", "content": prompt_system})
                messages.append({"role": "user", "content": prompt_base})

                think_config = {"thinking": {"type": "enabled"}}
                think_budget = 5000

                response = self.aoiclient.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens + think_budget,
                    extra_body=think_config,
                )
                success = True
            except:
                time.sleep(API_TIMEOUTS[t])
                t = t + 1
            
        if not success:
            print(f"Failed to get response for prompt")
            return ""
        if response is None or isinstance(response, str) or response.choices[0].message.content is None:
            print(f"Response is None for prompt")
            return ""
        return response.choices[0].message.content.rsplit("</think>", 1)[-1].strip()

class QwenOpenModel(LanguageModel):
    def __init__(self, model_name, args):
        super().__init__(model_name, args)
        transformers.set_seed(1234)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_message_answer(self, messages: list[dict[str, str]], max_tokens: int, temperature: float, top_p: float,) -> str:
        texts = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([texts], return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            input_ids = model_inputs['input_ids'],
            attention_mask = model_inputs['attention_mask'],
            max_new_tokens=max_tokens + 5000,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_ids = outputs[0][model_inputs['input_ids'].shape[-1]:].tolist()
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()     
        response_str = response.rsplit("</think>", 1)[-1].strip()
        return response_str

    def get_batch_message_answer(self, messages: list[list[dict[str, str]]], max_tokens: int, temperature: float, top_p: float) -> list[str]:
        self.tokenizer.padding_side = "left"
        texts = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        
        outputs = self.model.generate(
            input_ids=model_inputs['input_ids'],
            attention_mask=model_inputs['attention_mask'],
            max_new_tokens=max_tokens + 5000,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id
        )

        output_ids = outputs[:, model_inputs['input_ids'].shape[-1]:]
        responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        response_strs = [resp.rsplit("</think>", 1)[-1].strip() for resp in responses]

        return response_strs

    def get_top_p_answer(self, prompt_base: str, prompt_system: str, max_tokens: int, temperature: float, top_p: float,) -> str:
        messages = []
        if prompt_system != "":
            messages.append({"role": "system", "content": prompt_system})
        messages.append({"role": "user", "content": prompt_base})

        texts = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([texts], return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            input_ids = model_inputs['input_ids'],
            attention_mask = model_inputs['attention_mask'],
            max_new_tokens=max_tokens + 5000,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_ids = outputs[0][model_inputs['input_ids'].shape[-1]:].tolist()
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        response_str = response.rsplit("</think>", 1)[-1].strip()
        return response_str
    
    def get_batch_top_p_answer(self, prompts_base: list[str], prompts_system: list[str], max_tokens: int, temperature: float, top_p: float) -> list[str]:
        messages = []
        for prompt_system, prompt_base in zip(prompts_system, prompts_base):
            msg = []
            if prompt_system != "":
                msg.append({"role": "system", "content": prompt_system})
            msg.append({"role": "user", "content": prompt_base})
            messages.append(msg)
        
        self.tokenizer.padding_side = "left"
        texts = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        
        outputs = self.model.generate(
            input_ids=model_inputs['input_ids'],
            attention_mask=model_inputs['attention_mask'],
            max_new_tokens=max_tokens + 5000,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id
        )

        output_ids = outputs[:, model_inputs['input_ids'].shape[-1]:]
        responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        response_strs = [resp.rsplit("</think>", 1)[-1].strip() for resp in responses]

        return response_strs

class GemmaModel(LanguageModel):
    def __init__(self, model_name, args):
        super().__init__(model_name, args)
        transformers.set_seed(1234)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoProcessor.from_pretrained(self._model_name)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self._model_name,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        # self.processor.pad_token = self.processor.eos_token

    def get_message_answer(self, messages: list[dict[str, str]], max_tokens: int, temperature: float, top_p: float,) -> str:
        new_messages = []
        for part in messages:
            new_messages.append({"role": part["role"], "content": [{"type": "text", "text": part["content"]}]})
        model_inputs = self.processor.apply_chat_template(
            new_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.generate(
            input_ids = model_inputs['input_ids'],
            attention_mask = model_inputs['attention_mask'],
            max_new_tokens=max_tokens + 5000,
            do_sample=False,
            temperature=temperature,
            top_p=top_p,
            # pad_token_id=self.processor.eos_token_id
        )
        output_ids = outputs[0][model_inputs['input_ids'].shape[-1]:].tolist()
        response = self.processor.decode(output_ids, skip_special_tokens=True).strip()     
        response_str = response.rsplit("</think>", 1)[-1].strip()
        return response_str

    def get_batch_message_answer(self, messages: list[list[dict[str, str]]], max_tokens: int, temperature: float, top_p: float) -> list[str]:
        new_messages = []
        self.processor.padding_side = "left"
        for msg in messages:
            new_msg = []
            for part in msg:
                new_msg.append({"role": part["role"], "content": [{"type": "text", "text": part["content"]}]})
            new_messages.append(new_msg)
        model_inputs = self.processor.apply_chat_template(
            new_messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        
        outputs = self.model.generate(
            input_ids=model_inputs['input_ids'],
            attention_mask=model_inputs['attention_mask'],
            max_new_tokens=max_tokens + 5000,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            # pad_token_id=self.processor.eos_token_id
        )

        output_ids = outputs[:, model_inputs['input_ids'].shape[-1]:]
        responses = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        response_strs = [resp.rsplit("</think>", 1)[-1].strip() for resp in responses]

        return response_strs

    def get_top_p_answer(self, prompt_base: str, prompt_system: str, max_tokens: int, temperature: float, top_p: float,) -> str:
        messages = []
        if prompt_system != "":
            messages.append({"role": "system", "content": [{"type": "text", "text": prompt_system}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": prompt_base}]})

        model_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.generate(
            input_ids = model_inputs['input_ids'],
            attention_mask = model_inputs['attention_mask'],
            max_new_tokens=max_tokens + 5000,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        output_ids = outputs[0][model_inputs['input_ids'].shape[-1]:].tolist()
        response = self.processor.decode(output_ids, skip_special_tokens=True).strip()
        response_str = response.rsplit("</think>", 1)[-1].strip()
        return response_str
    
    def get_batch_top_p_answer(self, prompts_base: list[str], prompts_system: list[str], max_tokens: int, temperature: float, top_p: float) -> list[str]:
        messages = []
        for prompt_system, prompt_base in zip(prompts_system, prompts_base):
            msg = []
            if prompt_system != "":
                msg.append({"role": "system", "content": [{"type": "text", "text": prompt_system}]})
            msg.append({"role": "user", "content": [{"type": "text", "text": prompt_base}]})
            messages.append(msg)
        
        self.processor.padding_side = "left"
        model_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        
        outputs = self.model.generate(
            input_ids=model_inputs['input_ids'],
            attention_mask=model_inputs['attention_mask'],
            max_new_tokens=max_tokens + 5000,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            # pad_token_id=self.processor.eos_token_id
        )

        output_ids = outputs[:, model_inputs['input_ids'].shape[-1]:]
        responses = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        response_strs = [resp.rsplit("</think>", 1)[-1].strip() for resp in responses]

        return response_strs