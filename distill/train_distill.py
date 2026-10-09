import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
AutoModelForCausalLM, AutoTokenizer,
DataCollatorForLanguageModeling, Trainer, TrainingArguments,
)



TEACHER = "Qwen/Qwen3-1.7B"
STUDENT = "Qwen/Qwen3-0.6B"
MAX_LEN = 512

# 直接比较词表分布，所以必须同 tokenizer / vocab。
tok = AutoTokenizer.from_pretrained(STUDENT, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def encode(batch):
    return tok(
        batch["text"], truncation=True,
        max_length=MAX_LEN, add_special_tokens=True,
    )


raw = load_dataset("json", data_files="distill/train.jsonl")["train"]
ds = raw.map(encode, batched=True, remove_columns=raw.column_names)
ds = ds.train_test_split(test_size=0.05, seed=42)
collator = DataCollatorForLanguageModeling(tok, mlm=False)

teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER, torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    )
teacher.requires_grad_(False)
teacher.eval()


student = AutoModelForCausalLM.from_pretrained(
    STUDENT, torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    )
student.config.use_cache = False