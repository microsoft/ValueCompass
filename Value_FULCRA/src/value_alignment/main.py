from dataclasses import dataclass, field
from typing import Optional

import torch
import tyro
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, is_deepspeed_available

from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from trl.import_utils import is_xpu_available

import os
import sys
import numpy as np
import math
sys.path.append('.')
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.data_utils import prompt_for_evaluator, value_type_10_list, build_prompt, load_saferlhf_dataset

tqdm.pandas()

target_value = np.array([0,0,0,1,0,1,0,1,1,1])
# target_value = np.array([0,0,0,0,0,1,0,1,0,0])
# target_value = np.array([0.4809, 0.1305, 0.1004, 0.0029, -0.2383, 0.4362, 0.2712, 0.1420, 0.6520, 0.5906])  # UK values
# target_value = np.array([0.5190, 0.1148, 0.3016, 0.1311, -0.1124, 0.3389, 0.2165, 0.2578, 0.5661, 0.5243])  # Netherland
# target_value = np.array([0.3761, -0.0296, 0.3384, -0.1741, -0.3195, 0.4078, 0.2587, 0.0698, 0.6212, 0.5705])  # French


@dataclass
class ScriptArguments:
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            # common parameters
            exp_name="value_alignment_exp",
            seed=2023,
            log_with="wandb",
            model_name = "PKU-Alignment/alpaca-7b-reproduced",
            # model_name = "mistralai/Mistral-7B-Instruct-v0.1",
            query_dataset="../../data/saferlhf",
            reward_model="../../output/evaluator/deberta_tanh_ensemble",
            # hyperparameters
            # steps=1000,
            learning_rate=1e-5,
            batch_size=32,   # Number of samples per optimisation step
            mini_batch_size=4,  # Number of samples optimized in each mini batch
            gradient_accumulation_steps=8,
            ppo_epochs=2,    # Number of optimisation epochs per batch of samples
            early_stopping=False,
            target_kl=6.0,
            kl_penalty="kl",
            init_kl_coef=0.1,  # Initial coefficient for the KL loss
            adap_kl_ctrl=True,
            use_score_scaling=True,
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
            # lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="../../output/value_alignment", metadata={"help": "n steps to save the model"})
    epochs: Optional[int] = field(default=20, metadata={"help": "total epochs to train the model"})
    run_mark: Optional[str] = field(default="ideal", metadata={"help": "mark of the run"})  # French


args = tyro.cli(ScriptArguments)


# the arguments to pass to the sentiment analysis pipeline.
sent_kwargs = {"function_to_apply": "none", "batch_size": 4}

trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead


# its own dataset.
def build_dataset(config, query_dataset, input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training.

    Args:
        query_dataset (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # ds = load_dataset(query_dataset, split="train")
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

# retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(args.ppo_config, args.ppo_config.query_dataset)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(args.ppo_config.seed)

# build the model, the reference model, and the tokenizer.
if not args.use_peft:
    ref_model = trl_model_class.from_pretrained(args.ppo_config.model_name, trust_remote_code=args.trust_remote_code)
    device_map = None
    peft_config = None
else:
    peft_config = args.peft_config
    ref_model = None
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}

para_path = f"dataset_{args.ppo_config.query_dataset.split('/')[-1]}_model_{args.ppo_config.model_name.split('/')[-1]}_reward_{args.ppo_config.reward_model.split('/')[-1]}_bs_{args.ppo_config.batch_size}_kl_{args.ppo_config.init_kl_coef}_ppo_epochs_{args.ppo_config.ppo_epochs}_epochs_{args.epochs}_{args.run_mark}"
output_dir = os.path.join(args.output_dir, para_path)
os.makedirs(output_dir, exist_ok=True)

if os.path.exists(os.path.join(output_dir, "final_model")):  # load model and continue train
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

tokenizer.pad_token_id = tokenizer.eos_token_id

# build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(args.ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    if is_xpu_available():
        device = "xpu:0"
    else:
        device = 0 if torch.cuda.is_available() else "cpu"
ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin


evaluator = args.ppo_config.reward_model
evaluator_model = AutoModelForSequenceClassification.from_pretrained(evaluator)

evaluator_tokenizer = AutoTokenizer.from_pretrained(
    evaluator,
    padidng_side = "right",
    truncation_side = "right"
    )

evaluator_tokenizer_kwargs = {
    "max_length": 512,
    "use_fast": True,
    "truncation": True,
    "padding": True,
}

if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
    with ds_plugin.zero3_init_context_manager(enable=False):
        value_pipe = pipeline("text-classification", model=evaluator_model, tokenizer = evaluator_tokenizer, device=device, **evaluator_tokenizer_kwargs)
else:
    value_pipe = pipeline("text-classification", model=evaluator_model, tokenizer = evaluator_tokenizer, device=device, **evaluator_tokenizer_kwargs)

if value_pipe.tokenizer.pad_token_id is None:
    value_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

if value_pipe.model.config.pad_token_id is None:
    value_pipe.model.config.pad_token_id = tokenizer.pad_token_id

# the arguments to pass to the `generate` function
generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 512,
}

def compute_distance(queries, responses, value_pipe):
    results = None
    texts = [] # len(queries) * 10, list of texts to be evaluated
    for value in value_type_10_list:
        texts += [prompt_for_evaluator((q + r).replace("Assistant","Bob"), value) for q, r in zip(queries, responses)]
    inference_results = value_pipe(texts, **sent_kwargs)

    value_list = None
    for idx, inference_result in enumerate(inference_results):
        if np.all(value_list == None):
            value_list = np.array([[inference_result['score']]])
        else:
            value_list = np.vstack((value_list,[inference_result['score']])) # [10, 1]
        if (idx+1) % args.ppo_config.batch_size == 0:
            if np.all(results == None):
                results = value_list
            else:
                results = np.hstack((results,value_list)) # [batch_size, 10]
            value_list = None
    results = np.tanh(results)
    # results = np.clip(results, -1, 1)  # [batch_size, 10]
    pred_mask = ((results < -0.3) | (results > 0.3)).astype(float)  # mask those values predicted as "no connection"
    # distances = (np.abs(results - target_value) * pred_mask * (target_value != 0).astype(float)).sum(axis=1)  # [batch_size], * target_value to ignore some dimensions
    distances = (np.abs(results - target_value) * pred_mask * np.abs(target_value).astype(float)).sum(axis=1)  # [batch_size], * target_value to ignore some dimensions
    return distances

for epoch in range(args.epochs):
    for step, batch in tqdm(enumerate(ppo_trainer.dataloader), desc=f"Epoch {epoch+1}/{args.epochs}"):
        query_tensors = batch["input_ids"]

        # Get response
        response_tensors, ref_response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)
        batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

        # Compute distances and rewards
        distances = compute_distance(batch['query'], batch['response'], value_pipe)
        rewards = [torch.tensor(0 - distance) for distance in distances]
        ref_distances = compute_distance(batch['query'], batch['ref_response'], value_pipe)
        ref_rewards = [torch.tensor(0 - ref_distance) for ref_distance in ref_distances]
        batch["ref_rewards"] = ref_rewards

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])
        
        if (step+1) % 200 == 0:
            ppo_trainer.save_pretrained(output_dir)
    ppo_trainer.save_pretrained(output_dir + f"/epoch_{epoch+1}")