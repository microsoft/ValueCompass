# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional

import torch
import tyro
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from trl.import_utils import is_xpu_available

from utils.data_utils import build_prompt, load_saferlhf_dataset
import os

tqdm.pandas()

@dataclass
class ScriptArguments:
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            exp_name = "ppo_exp",
            # model_name="meta-llama/Llama-2-7b-chat-hf",
            # model_name = "meta-llama/Llama-2-7b-hf",
            model_name = "PKU-Alignment/alpaca-7b-reproduced",
            query_dataset="../../data/saferlhf",
            reward_model="sentiment-analysis:OpenAssistant/reward-model-deberta-v3-large-v2",
            learning_rate=1.41e-5,
            log_with="wandb",
            mini_batch_size=4,
            batch_size=32,
            gradient_accumulation_steps=8,
            ppo_epochs=2,    # Number of optimisation epochs per batch of samples
            early_stopping=False,
            target_kl=6.0,
            kl_penalty="kl",
            init_kl_coef=0.1,  # decrease kl_coef
            adap_kl_ctrl=True,
            seed=0,
            use_score_scaling=False,
            use_score_norm=False,
            score_clip=None,
        )
    )
    use_seq2seq: bool = False
    """whether to use seq2seq models"""
    use_peft: bool = True
    """whether to use peft"""
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=8,
            lora_alpha=16,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="../../output/ppo", metadata={"help": "n steps to save the model"})
    epochs: Optional[int] = field(default=20, metadata={"help": "total epochs to train the model"})


args = tyro.cli(ScriptArguments)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 8}

trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(config, query_dataset, input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        query_dataset (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_saferlhf_dataset(query_dataset, split="train")

    def tokenize(sample):
        if "harmless-base" in query_dataset:
            index=sample['chosen'].rfind('Assistant:')
            sample["prompt"] = build_prompt(sample['chosen'], args.ppo_config.model_name)
            sample["input_ids"] = tokenizer.encode(sample["prompt"])
            sample["query"] = sample['chosen'][:index] + 'Assistant:'
        elif "saferlhf" in query_dataset:
            sample["query"] = "Human: " + sample['prompt'] + "\nAssistant: "
            sample["prompt"] = build_prompt(sample['prompt'], args.ppo_config.model_name)
            sample["input_ids"] = tokenizer.encode(sample["prompt"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(args.ppo_config, args.ppo_config.query_dataset)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(args.ppo_config.seed)

# Now let's build the model, the reference model, and the tokenizer.
if not args.use_peft:
    ref_model = trl_model_class.from_pretrained(args.ppo_config.model_name, trust_remote_code=args.trust_remote_code)
    device_map = None
    peft_config = None
else:
    peft_config = args.peft_config
    ref_model = None
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}

para_path = f"dataset_{args.ppo_config.query_dataset.split('/')[-1]}_model_{args.ppo_config.model_name.split('/')[-1]}_reward_{args.ppo_config.reward_model.split('/')[-1]}_bs_{args.ppo_config.batch_size}_kl_{args.ppo_config.init_kl_coef}_ppo_epochs_{args.ppo_config.ppo_epochs}_epochs_{args.epochs}"
output_dir = os.path.join(args.output_dir, para_path)
os.makedirs(output_dir, exist_ok=True)
# ppo_trainer.save_pretrained(output_dir)

if os.path.exists(os.path.join(output_dir, "final_model")):
    print("Load from final model...")
    model = trl_model_class.from_pretrained(
        os.path.join(output_dir, "final_model"),
        trust_remote_code=args.trust_remote_code,
        device_map=device_map,
        peft_config=peft_config,
        )
else:
    model = trl_model_class.from_pretrained(
        args.ppo_config.model_name,
        trust_remote_code=args.trust_remote_code,
        device_map=device_map,
        peft_config=peft_config,
    )

tokenizer = AutoTokenizer.from_pretrained(args.ppo_config.model_name)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
tokenizer.pad_token_id = tokenizer.eos_token_id

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(args.ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    if is_xpu_available():
        device = "xpu:0"
    else:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
task, model_name = args.ppo_config.reward_model.split(":")
if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
    with ds_plugin.zero3_init_context_manager(enable=False):
        sentiment_pipe = pipeline(task, model=model_name, device=device)
else:
    sentiment_pipe = pipeline(task, model=model_name, device=device)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
if sentiment_pipe.tokenizer.pad_token_id is None:
    sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "max_new_tokens": 512,
}

for epoch in range(args.epochs):
    for step, batch in tqdm(enumerate(ppo_trainer.dataloader), desc=f"Epoch {epoch+1}/{args.epochs}"):
        query_tensors = batch["input_ids"]

        # Get response from gpt2
        response_tensors, ref_response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)
        batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

        # Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]

        ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
        ref_pipe_outputs = sentiment_pipe(ref_texts, **sent_kwargs)
        ref_rewards = [torch.tensor(output[0]["score"]) for output in ref_pipe_outputs]
        batch["ref_rewards"] = ref_rewards

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])

        if (step+1) % 200 == 0:
            ppo_trainer.save_pretrained(output_dir)
    ppo_trainer.save_pretrained(output_dir + f"/epoch_{epoch+1}")