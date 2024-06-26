import os
import math
import sys
import random
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.data_utils import *

IGNORE_TOKEN_ID = -100

class ClassTrainDataset(Dataset):
    def __init__(self, args, tokenizer, max_length=1024, split="train"):
        self.args = args
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = args.data_path

        self.raw_data = []
        with open(self.data_path, 'r') as fr:
            for line in fr:
                line = json.loads(line.strip())
                self.raw_data.append([line['dialogue'], line['value_items'], line['value_types']])
        # random.seed(2024)
        # random.shuffle(self.raw_data)
        if split == "train":
            self.raw_data = self.raw_data[:int(len(self.raw_data)*0.9)]
            print("#raw train data: ", len(self.raw_data))
        elif split == "valid":
            self.raw_data = self.raw_data[int(len(self.raw_data)*0.9):]
            print("#raw valid data: ", len(self.raw_data))

        self.all_data = []
        self.process_all_data()
 
    def process_all_data(self):
        invalid_count = 0
        for dialogue, value_items, value_types in self.raw_data:
            for value, label in process_annotate_values(value_items, value_types, self.args, self.split):
                prompt, label = prompt_for_classification(dialogue, value, label, self.args)
                if prompt is None:
                    invalid_count += 1
                    continue
                self.all_data.append({"prompt": prompt, "label": min(label, self.args.class_num-1)})
        print("invalid count: ", invalid_count)

    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        prompt = self.all_data[idx]["prompt"]
        label = self.all_data[idx]["label"]
        encoding = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            max_length = self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": label,
        }
    
    def collator(self, features):
        max_length = max([f["input_ids"].size(0) for f in features])
        input_ids = torch.stack([F.pad(f["input_ids"], (0, max_length-f["input_ids"].size(0)), value=self.tokenizer.pad_token_id) for f in features])
        attention_mask = torch.stack([F.pad(f["attention_mask"], (0, max_length-f["attention_mask"].size(0))) for f in features])
        labels = torch.tensor([f["label"] for f in features], dtype=torch.half)
        return_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        return return_dict

class ClassEvaluationDataset(Dataset):
    def __init__(self, args, tokenizer, max_length=1024, split="valid"):
        self.args = args
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = args.data_path
        
        self.raw_data = []
        with open(self.data_path, 'r') as fr:
            for line in fr:
                line = json.loads(line.strip())
                self.raw_data.append([line['dialogue'], line['value_items'], line['value_types']])
        # random.seed(2024)
        # random.shuffle(self.raw_data)
        if split == "train":
            self.raw_data = self.raw_data[:int(len(self.raw_data)*0.9)]
            print("#raw train data: ", len(self.raw_data))
        elif split == "valid":
            self.raw_data = self.raw_data[int(len(self.raw_data)*0.9):]
            print("#raw valid data: ", len(self.raw_data))

        self.all_data = []
        self.process_all_data()

    def process_all_data(self):
        invalid_count = 0
        for dialogue, value_items, value_types in self.raw_data:
            for value, label in process_annotate_values(value_items, value_types, self.args, self.split):
                prompt, label = prompt_for_classification(dialogue, value, label, self.args)
                if prompt is None:
                    invalid_count += 1
                    continue
                value_id = get_value_id(value, self.args)
                self.all_data.append({"prompt": prompt, "label": min(label, self.args.class_num-1), "value_id": value_id})
        print("invalid count: ", invalid_count)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        prompt = self.all_data[idx]["prompt"]
        label = self.all_data[idx]["label"]
        value_id = self.all_data[idx]["value_id"]
        encoding = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            max_length = self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": label,
            "value_id": value_id
        }

    def collator(self, features):
        max_length = max([f["input_ids"].size(0) for f in features])
        input_ids = torch.stack([F.pad(f["input_ids"], (0, max_length-f["input_ids"].size(0)), value=self.tokenizer.pad_token_id) for f in features])
        attention_mask = torch.stack([F.pad(f["attention_mask"], (0, max_length-f["attention_mask"].size(0))) for f in features])
        labels = torch.tensor([f["label"] for f in features], dtype=torch.half)
        value_ids = torch.tensor([f["value_id"] for f in features], dtype=torch.long)
        return_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "value_ids": value_ids
        }
        return return_dict

class InferenceDataset(Dataset):
    def __init__(self, args, tokenizer, max_length=1024):
        self.args = args
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = args.inference_file_path
        self.test_data = []
        with open(self.data_path, 'r') as file:
            for line in file:
                self.test_data.append(json.loads(line))
        self.prompts = []
        self.dialogues = []
        self.value_list = value_item_list
        if args.value_item_or_type == "value_type":
            if args.value_type_dimension == 10:
                self.value_list = value_type_10_list
            else:
                self.value_list = value_type_19_list
        self.classifier_data()
    
    def classifier_data(self):
        label = "1"
        for data in self.test_data:
            dialogue = "Human:" + data['prompt'] + "\n" + "Bob: " + data['answer']
            for value in self.value_list:
                prompt, _ = prompt_for_classification(dialogue, value, label, self.args)
                self.dialogues.append(dialogue)
                self.prompts.append(prompt)
                self.value_ids.append(get_value_id(value, self.args))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        dialogue = self.dialogues[idx]
        value_id = self.value_ids[idx]
        encoding = self.tokenizer(
            prompt,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            padding=True,
        )
        return {
            "dialogue": dialogue,
            "prompt": prompt,
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "value_id": value_id
        }

    def collator(self, features):
        max_length = max([f["input_ids"].size(0) for f in features])
        dialogue = [f["dialogue"] for f in features]
        prompt = [f["prompt"] for f in features]
        input_ids = torch.stack([F.pad(f["input_ids"], (max_length-f["input_ids"].size(0), 0), value=self.tokenizer.pad_token_id) for f in features])
        attention_mask = torch.stack([F.pad(f["attention_mask"], (max_length-f["attention_mask"].size(0), 0)) for f in features])
        value_ids = value_ids = torch.tensor([f["value_id"] for f in features], dtype=torch.long)
        return_dict = {
            "dialogue": dialogue,
            "prompt": prompt,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "value_ids": value_ids
        }
        return return_dict