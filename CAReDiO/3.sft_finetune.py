import os
import json
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field

import torch
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    is_torch_npu_available,
    is_torch_xpu_available,
    set_seed,
)

from trl import SFTConfig, SFTTrainer
# from trl.trainer import ConstantLengthDataset

@dataclass
class ScriptArguments:
    align_method: Optional[str] = field(default="Role-Play", metadata={"help": "the alignment method, role-play or none in the system prompt"})
    train_data_type: Optional[str] = field(default="constant", metadata={"help": "the method of processing the training data: constant, plain, completion"})
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="lvwerra/stack-exchange-paired", metadata={"help": "the dataset name"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=350, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=False, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    use_bnb: Optional[bool] = field(default=True, metadata={"help": "whether to use BitsAndBytes"})

    # LoraConfig
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

parser = HfArgumentParser((ScriptArguments, SFTConfig))
script_args, training_args = parser.parse_args_into_dataclasses()
peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

if training_args.group_by_length and training_args.packing:
    raise ValueError("Cannot use both packing and group by length")

# `gradient_checkpointing` was True by default until `1f3314`, but it's actually not used.
# `gradient_checkpointing=True` will cause `Variable._execution_engine.run_backward`.
if training_args.gradient_checkpointing:
    raise ValueError("gradient_checkpointing not supported")

set_seed(training_args.seed)
if not os.path.exists(training_args.output_dir):
    os.makedirs(training_args.output_dir, exist_ok=True)

def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    country = script_args.dataset_name.split("/")[2].replace("_", " ")
    example["country"] = country
    if script_args.align_method == "Role-Play":
        if "real_question" in example:
            text = f"### System: You are a {example['country']} chatbot that know {example['country']} values very well. Please answer the following questions with cultural awareness and sensitivity, reflecting the values, norms, customs and styles of {example['country']}.\n### User: {example['real_question']}\n### Assistant: {example['answer']}"
        else:
            text = f"### System: You are a {example['country']} chatbot that know {example['country']} values very well. Please answer the following questions with cultural awareness and sensitivity, reflecting the values, norms, customs and styles of {example['country']}.\n### {example['answer']}"
    else:
        if "real_question" in example:
            text = f"### User: {example['real_question']}\n### Assistant: {example['answer']}"
        else:
            text = f"### {example['answer']}"

    return text

def create_datasets(tokenizer, args, seed=None):
    train_data = load_dataset("json", data_files = args.dataset_name, split="train")
    print("Loading the dataset: ", args.dataset_name, "\ntrain data: ", train_data) #, "\n valid data: ", valid_data)
    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    if args.train_data_type == "constant":
        train_dataset = ConstantLengthDataset(
            tokenizer,
            train_data,
            formatting_func=prepare_sample_text,
            infinite=True,
            seq_length=args.seq_length,
            chars_per_token=chars_per_token,
        )
    elif args.train_data_type == "plain":
        train_dataset = {"text": []}
        for example in train_data:
            text = prepare_sample_text(example)
            train_dataset["text"].append(text)
        train_dataset = Dataset.from_dict(train_dataset)
    elif args.train_data_type == "completion":
        train_dataset = {
            "prompt": [],
            "completion": [],
        }
        for example in train_data:
            country = script_args.dataset_name.split("/")[2].replace("_", " ")
            example["country"] = country
            if script_args.align_method == "Role-Play":
                if "real_question" in example:
                    prompt = f"### System: You are a {example['country']} chatbot that know {example['country']} values very well. Please answer the following questions with cultural awareness and sensitivity, reflecting the values, norms, customs and styles of {example['country']}.\n### User: {example['real_question']}\n### Assistant: "
                else:
                    prompt = f"### System: You are a {example['country']} chatbot that know {example['country']} values very well. Please answer the following questions with cultural awareness and sensitivity, reflecting the values, norms, customs and styles of {example['country']}.\n"
            else:
                if "real_question" in example:
                    prompt = f"### User: {example['real_question']}\n### Assistant: "
                else:
                    prompt = f""
            completion = example["answer"]
            train_dataset["prompt"].append(prompt)
            train_dataset["completion"].append(completion.strip())
        train_dataset = Dataset.from_dict(train_dataset)
    return train_dataset, None # valid_dataset

bnb_config = None
if script_args.use_bnb:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
print("use bnb: ", script_args.use_bnb, " bnb_config: ", bnb_config)

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map={"": Accelerator().local_process_index},
    trust_remote_code=True,
    use_auth_token=True,
)
base_model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

train_dataset, eval_dataset = create_datasets(tokenizer, script_args, seed=training_args.seed)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    # max_seq_length=None,
    # formatting_func=prepare_sample_text,
    processing_class=tokenizer,
    # tokenizer=tokenizer,
    args=training_args,
)
trainer.train()
trainer.save_model(training_args.output_dir)

output_dir = os.path.join(training_args.output_dir, "final_adapter")
trainer.model.save_pretrained(output_dir)