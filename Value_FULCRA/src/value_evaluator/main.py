import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    SchedulerType,
    get_scheduler,
)

import deepspeed
from deepspeed import get_accelerator
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append('.')
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from dataset import ClassTrainDataset, InferenceDataset, ClassEvaluationDataset

from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible
from utils.data_utils import value_item_list, value_type_10_list, value_type_19_list, get_type_10_id

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report

IGNORE_TOKEN_ID = -100

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    # increased parameters based on DeepSpeedExample
    parser.add_argument("--train",
                        action="store_true",
                        help="decide to train or inference model"
                        )
    parser.add_argument("--evaluate",
                        action="store_true",
                        help="evaluate model"
                        )
    parser.add_argument("--test",
                        action="store_true",
                        help="test model")
    parser.add_argument("--inference",
                        action="store_true",
                        help="inference model"
                        )
    parser.add_argument("--inference_file_path",
                        type=str,
                        help="inference file path"
                        )
    parser.add_argument("--load_local_ckpt",
                        action="store_true",
                        help="whether to load trained model ckpt from a local file.")
    parser.add_argument("--value_item_or_type",
                        type=str,
                        default="value_item",
                        help="the values to be classified, can be value items / value types")
    parser.add_argument("--value_type_dimension",
                        type=int,
                        default=10,
                        help="the dimension of value type, 10 or 19")
    parser.add_argument("--model_type",
                        type=str,
                        default="generator",
                        help="the task for evaluator, classification/regression")
    parser.add_argument("--class_num",
                        type=int,
                        default=3,
                        help="the number of classes to be annotated.")
    parser.add_argument("--cache_dir",
                        type=str,
                        default="~/.cache",
                        help="cache dir path")
    parser.add_argument('--data_path',
                        type=str,
                        default="../../data/evaluator_data.jsonl",
                        help='dataset dir path') # modified
    parser.add_argument('--dataset',
                        type=str,
                        default="saferlhf",
                        help='Path to the training dataset, can be saferlhf/harmless.')
    parser.add_argument('--label_num',
                        type=int,
                        default=100,
                        help="Max number of samples to annotate")
    parser.add_argument("--max_new_tokens",
                        type=int,
                        default=512,
                        help="Max number of tokens to generate.")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="If dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the model.")

    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp16',
                        choices=['fp16', 'bf16'],
                        help='Training data type')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')

    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')
    parser.add_argument('--print_every_n_step',
                        type=int,
                        default=50,
                        help='Print loss every n steps.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args

def find_nearest_label(output_value):
    labels = [-1, 0, 1]
    nearest_label = min(labels, key=lambda x: abs(x - output_value))
    return int(nearest_label)

def evaluation(args, model, eval_dataloader, device, epoch=-1, writer=None):
    model.eval()
    labels, predictions = [], []
    values = []
    losses = []
    for step, batch in tqdm(enumerate(eval_dataloader), desc="evaluating"):
        batch = to_device(batch, device)

        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

        if args.model_type == "classification":
            labels.extend(batch["labels"].cpu().tolist())
            values.extend(batch["value_ids"].cpu().tolist())
            logits = outputs.logits
            prediction = torch.argmax(logits, axis=-1).cpu().tolist()
            predictions.extend(prediction)
        elif args.model_type == "regression":
            labels.extend(batch["labels"].cpu().tolist())
            values.extend(batch["value_ids"].cpu().tolist())
            logits = torch.tanh(outputs.logits)
            prediction = [find_nearest_label(logit) for logit in logits]
            predictions.extend(prediction)
            loss_fct = MSELoss()
            loss = loss_fct(logits.squeeze(), batch["labels"].squeeze()).item()
            losses.append(loss)

        if step + 1 == 1500:
            break

    save_path = os.path.join(args.output_dir, f"eval_result_epoch_{epoch}")
    os.makedirs(save_path, exist_ok=True)

    if args.model_type == "classification" or args.model_type == "regression":
        # evaluate label
        label_results = []
        if args.model_type == "classification":
            report = classification_report(labels, predictions, labels = [i for i in range(args.class_num)], target_names=[f'label_{i}' for i in range(args.class_num)], output_dict=True)
        elif args.model_type == "regression":
            report = classification_report(labels, predictions, labels = [-1, 0, 1], target_names=[f'label_{i}' for i in range(args.class_num)], output_dict=True)
        print(report)
        label_results = pd.DataFrame(report)
        label_results.to_csv(os.path.join(save_path, f"label_eval_result.csv"), index=True)

        # evaluate value
        value_list = value_item_list
        if args.value_item_or_type == "value_type":
            if args.value_type_dimension == 10:
                value_list = value_type_10_list
            else:
                value_list = value_type_19_list

        value_num = len(value_list)
        total_value_num = 0
        value_sum_number = [0] * value_num
        value_accuracy = [0] * value_num
        value_type_10_sum_number = [0] * 10
        value_type_10_correct_num = [0] * 10
        for idx in range(len(value_list)):
            value_sum_number[idx] = sum([1 if value == idx else 0 for value in values])
            value_type_10_sum_number[get_type_10_id(idx, args)] += value_sum_number[idx]
            total_value_num += value_sum_number[idx]
        for idx in range(len(value_list)):
            value_correct_num = sum([1 if label == pred and value == idx else 0 for label, pred, value in zip(labels, predictions, values)])
            value_type_10_correct_num[get_type_10_id(idx, args)] += value_correct_num
            value_accuracy[idx] = value_correct_num / (value_sum_number[idx] + 1e-5)

        value_type_10_accuracy = [correct_num/(sum_num + 1e-5) for correct_num, sum_num in zip(value_type_10_correct_num, value_type_10_sum_number)]

        type_10_accuracy_dict = {}
        label_precision_dict = {}
        label_recall_dict = {}
        label_f1_dict = {}
        for i in range(args.class_num):
            label_precision_dict[f'label_{i}'] = report[f'label_{i}']['precision']
            label_recall_dict[f'label_{i}'] = report[f'label_{i}']['recall']
            label_f1_dict[f'label_{i}'] = report[f'label_{i}']['f1-score']
        print("label precision: ", label_precision_dict)
        print("label recall: ", label_recall_dict)
        print("label f1: ", label_f1_dict)

        for i in range(10):
            type_10_accuracy_dict[f'{value_type_10_list[i]}'] = value_type_10_accuracy[i]
        print("type_10_accuracy: ", type_10_accuracy_dict)

        if args.model_type == "regression":
            mse_loss = np.mean(losses)
            print("MSE Loss: ", mse_loss)

def train_model(args, model, tokenizer, ds_config, device):
    if args.model_type == "classification" or args.model_type == "regression":
        train_dataset = ClassTrainDataset(args, tokenizer, split="train", max_length=args.max_seq_len)
        eval_dataset = ClassEvaluationDataset(args, tokenizer, split="valid", max_length=args.max_seq_len)

    # binary classification, value item -- Train Dataset: 15066, #Eval Dataset: 146096
    # binary classification, value type -- #Train Dataset: 14637, #Eval Dataset: 9004
    print_rank_0(f"#Train Dataset: {len(train_dataset)}, #Eval Dataset: {len(eval_dataset)}", args.global_rank)
    args.vocab_size = model.config.vocab_size
    
    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)

    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=train_dataset.collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=eval_dataset.collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)
    
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    log_dir = os.path.join(args.output_dir, "tensorboard_logs")
    writer = SummaryWriter(log_dir)

    evaluation(args, model, eval_dataloader, device, epoch=-1, writer=writer)
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        import time
        for step, batch in enumerate(train_dataloader):
            start = time.time()
            batch = to_device(batch, device)
            # for bert
            if "bert" in args.model_name_or_path or "bart" in args.model_name_or_path:
                outputs = model(input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                labels=batch['labels'])
            else:
                outputs = model(**batch, use_cache=False)
            if args.model_type == "regression":
                tanh_output = torch.tanh(outputs.logits)
                loss_fct = MSELoss()
                loss = loss_fct(tanh_output.squeeze(), batch["labels"].squeeze())
            else:
                loss = outputs.loss
            if args.print_loss and step % args.print_every_n_step == 0: # 这个是有正常打印的，此外还有其他信息
                print_rank_0(
                    f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss}", args.global_rank
                )
            model.backward(loss)
            model.step()
            end = time.time()
            writer.add_scalar('Loss', loss, epoch * len(train_dataloader) + step)

        # Evaluating on the validation set.
        print_rank_0(
            f"***** Evaluating, Epoch {epoch+1}/{args.num_train_epochs} *****",
            args.global_rank)
        evaluation(args, model, eval_dataloader, device, epoch, writer)
        model.tput_timer.update_epoch_count()

        save_model(model, tokenizer, args, epoch)
    
    writer.close()

def save_model(model, tokenizer, args, epoch):
    if args.output_dir is not None:
        print_rank_0('saving the final model ...', args.global_rank)
        model = convert_lora_to_linear_layer(model)

        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args, sub_folder=f"epoch_{epoch+1}")

        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model,
                                  args.global_rank,
                                  args.output_dir,
                                  os.path.join(args.output_dir,f"epoch_{epoch+1}"),
                                  zero_stage=args.zero_stage)

def inference_classification(args, model, tokenizer, device):
    test_data = InferenceDataset(args, tokenizer, max_length=args.max_seq_len)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(test_data)
    else:
        eval_sampler = DistributedSampler(test_data)
    test_dataloader = DataLoader(test_data,
                                collate_fn=test_data.collator,
                                shuffle=False,
                                sampler=eval_sampler,
                                batch_size=args.per_device_eval_batch_size)

    annotations = []
    value_ids = []
    model.eval()
    for batch in tqdm(test_dataloader, desc="inference"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_value_ids = batch["value_ids"]
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            answers = torch.argmax(logits, axis=-1).cpu().tolist()
            annotations.extend(answers)
        value_ids.extend(batch_value_ids)

    value_list = value_item_list
    if args.value_item_or_type == "value_type":
        if args.value_type_dimension == 10:
            value_list = value_type_10_list
        else:
            value_list = value_type_19_list
    values = []
    analysis = [[0]*(len(value_list)) for _ in range(args.class_num+1)]
    for value, label in zip(value_ids, annotations):
        analysis[label][value] += 1
    for col in range(len(value_list)):
        analysis[args.class_num][col] = sum([row * analysis[row][col] for row in range(args.class_num)]) / sum([analysis[row][col] for row in range(args.class_num)])
    result_df = pd.DataFrame(analysis, columns=value_list)
    output_file = f"inference_{os.path.splitext(os.path.basename(args.inference_file_path))[0]}.csv"
    result_df.to_csv(os.path.join(args.output_dir, output_file), index=False)

def inference_regression(args, model, tokenizer, device):
    test_data = InferenceDataset(args, tokenizer, max_length=args.max_seq_len)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(test_data)
    else:
        eval_sampler = DistributedSampler(test_data)
    test_dataloader = DataLoader(test_data,
                                collate_fn=test_data.collator,
                                shuffle=False,
                                sampler=eval_sampler,
                                batch_size=args.per_device_eval_batch_size)
    rewards = []
    annotations = []
    model.eval()
    for batch in tqdm(test_dataloader, desc="inference"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_value_ids = batch["value_ids"]
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits.squeeze().to('cpu')
            # tanh if needed
            logits = torch.tanh(logits)
        rewards.extend(logits)

    value_list = value_item_list
    if args.value_item_or_type == "value_type":
        if args.value_type_dimension == 10:
            value_list = value_type_10_list
        else:
            value_list = value_type_19_list
    value_rewards = np.array([rewards[i:i+ len(value_list)] for i in range(0, len(rewards), len(value_list))])
    print("avg value reward score: ", np.mean(value_rewards, axis=0))

def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload, stage=args.zero_stage)
    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config['train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir = args.cache_dir,
        model_max_length = 512 if "bert" in args.model_name_or_path or "bart" in args.model_name_or_path else 4096,
        use_fast = True,
        padding_side = "right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm_config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    llm_config.pad_token_id = tokenizer.pad_token_id
    llm_config.use_cache = False

    if "bert" in args.model_name_or_path or "bart" in args.model_name_or_path:
        args.max_seq_len = 512
    
    if args.model_type == "classification":
        llm_config.num_labels = args.class_num
        model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=llm_config,
        cache_dir=args.cache_dir)
    elif args.model_type == "regression":
        llm_config.num_labels = 1
        model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=llm_config,
        cache_dir=args.cache_dir)
    
    para_path = f"type_{args.model_type}_{args.value_item_or_type}_zero_stage_{args.zero_stage}_lora_{args.lora_dim}"
    args.output_dir = os.path.join(args.output_dir, args.model_name_or_path.split("/")[-1], para_path)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.load_local_ckpt:
        save_path = os.path.join(args.output_dir, "pytorch_model.bin")
        print_rank_0("loading the saved model from path: " + save_path, args.global_rank)
        model.load_state_dict(torch.load(save_path))
    
    model = model.to(device)
    if args.lora_dim > 0 and args.train:
        model = convert_linear_layer_to_lora(model, args.lora_module_name,
                                             args.lora_dim)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)
            model = make_model_gradient_checkpointing_compatible(model)
    
    if args.train:
        train_model(args, model, tokenizer, ds_config, device)
    
    if args.evaluate or args.test:
        eval_dataset = ClassEvaluationDataset(args, tokenizer, split="valid", max_length=args.max_seq_len)
        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_dataset)
        else:
            eval_sampler = DistributedSampler(eval_dataset)

        eval_dataloader = DataLoader(eval_dataset,
                                    collate_fn=eval_dataset.collator,
                                    sampler=eval_sampler,
                                    batch_size=args.per_device_eval_batch_size)
        evaluation(args, model, eval_dataloader, device)

    if args.inference:
        if args.model_type == "classification":
            print_rank_0("starting inference classification...", args.global_rank)
            inference_classification(args, model, tokenizer, device)
        elif args.model_type == "regression":
            print_rank_0("starting inference regression...", args.global_rank)
            inference_regression(args, model, tokenizer, device)

if __name__ == "__main__":
    main()
