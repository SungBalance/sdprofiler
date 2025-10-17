import os
import sys
import warnings
import re
import argparse
from typing import List, Dict, Tuple, Optional
import json
import time
from collections import defaultdict

# 기본 라이브러리
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    pipeline, set_seed
)
import torch


def split_into_sentences(text: str) -> List[str]:
    sentence_pattern = r'[.!?]+\s+|[。！？]+\s*|[\n\r]+'
    sentences = re.split(sentence_pattern, text.strip())
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 1]
    return sentences


def calculate_sentence_lengths(sentences: List[str], unit='characters', tokenizer=None) -> List[int]:
    if unit == 'characters':
        return [len(s) for s in sentences]
    elif unit == 'words':
        return [len(s.split()) for s in sentences]
    elif unit == 'tokens':
        if tokenizer:
            lengths = []
            for s in sentences:
                try:
                    tokens = tokenizer.encode(s, add_special_tokens=False)
                    lengths.append(len(tokens))
                except:
                    lengths.append(len(s.split()))
            return lengths
        else:
            return [len(s.split()) + len(re.findall(r'[^\w\s]', s)) for s in sentences]
    else:
        raise ValueError("unit은 'characters', 'words', 또는 'tokens' 중 하나여야 합니다.")
    

def load_datasets(args):
    return load_dataset(args.dataset_name, args.dataset_subset, split=args.dataset_split)


def set_hf_pipeline(args):
    return pipeline(
        "text-generation",
        model=args.model_name,
        tokenizer=args.tokenizer_name,
        device=args.device,
        max_length=args.max_length
    )

def set_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    args.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen3-4B")
    args.add_argument("--device", type=int, default=0)
    args.add_argument("--max_length", type=int, default=None)
    args.add_argument("--num_return_sequences", type=int, default=1)
    args.add_argument("--temperature", type=float, default=0.7)
    args.add_argument("--top_p", type=float, default=0.9)
    args.add_argument("--do_sample", type=bool, default=True)
    args.add_argument("--top_k", type=int, default=50)
    args.add_argument("--repetition_penalty", type=float, default=1.0)
    args.add_argument("--dataset_name", type=str, default="sharegpt")
    args.add_argument("--dataset_subset", type=str, default=None)
    args.add_argument("--dataset_split", type=str, default="train")
    args.add_argument("--dataset_size", type=int, default=1000)
    return args

if __name__ == "__main__":
    args = set_args()
    print(args)
    dataset = load_datasets(args)
    print(dataset)
    pipeline = set_hf_pipeline(args)
    print(pipeline)


    generation_kwargs = {
        "max_length": args.max_length,
        "num_return_sequences": args.num_return_sequences,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
        "pad_token_id": pipeline.tokenizer.eos_token_id,
        "eos_token_id": pipeline.tokenizer.eos_token_id,
    }

    responses = pipeline.generate(dataset["prompt"], **generation_kwargs)
    print(responses)