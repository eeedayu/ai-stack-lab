import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
AutoModelForCausalLM, AutoTokenizer,
DataCollatorForLanguageModeling, Trainer, TrainingArguments,
)

from distill_trainer import DistillTrainer


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


raw = load_dataset("json", data_files="data/cpt/train.jsonl")["train"]
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


args = TrainingArguments(
    output_dir="outputs/qwen3-0.6b-distilled",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    num_train_epochs=1,
    warmup_steps=0.03,
    lr_scheduler_type="cosine",
    fp16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
    )


trainer = DistillTrainer(
    model=student, teacher=teacher,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=collator,
    processing_class=tok,
    alpha=0.5, temperature=2.0, kd_chunk=64,
    )


trainer.train() # 续训：resume_from_checkpoint=True
trainer.save_model(args.output_dir)
tok.save_pretrained(args.output_dir)