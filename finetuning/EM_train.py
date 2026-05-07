#!/usr/bin/env python3
"""Fine-tuning Script for Energy Scan Agent (Qwen 2.5 1.5B)"""

import copy
import json
import logging
import re
import sys
import torch
import torch.nn as nn
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).parent.parent))
from config import AGENT_MODELS


# ──────────────────────────────────────────────
# Weighted Loss Trainer
# ──────────────────────────────────────────────

class WeightedLossTrainer(Trainer):
    """Assistant decision 및 User 응답 토큰에 높은 가중치를 부여하는 Trainer."""

    def __init__(self, decision_weight=5.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decision_weight = decision_weight
        self._tokenizer = kwargs.get('processing_class') or kwargs.get('tokenizer')

    def _find_role_boundaries(self, input_ids, tokenizer):
        try:
            full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
            boundaries = {'assistant_decisions': [], 'user_responses': []}
            input_list = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
            for pattern, key in [
                (r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', 'assistant_decisions'),
                (r'<\|im_start\|>user\n(.*?)<\|im_end\|>',      'user_responses'),
            ]:
                for match in re.finditer(pattern, full_text, re.DOTALL):
                    part_tokens = tokenizer.encode(full_text[match.start():match.end()], add_special_tokens=False)
                    for i in range(len(input_list) - len(part_tokens) + 1):
                        if input_list[i:i + len(part_tokens)] == part_tokens:
                            boundaries[key].append((i, i + len(part_tokens)))
                            break
            return boundaries
        except:
            return {'assistant_decisions': [], 'user_responses': []}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        weights = torch.ones_like(per_token_loss)
        tokenizer = self._tokenizer or getattr(self, 'processing_class', None)
        if tokenizer is not None:
            for i in range(shift_labels.size(0)):
                boundaries = self._find_role_boundaries(inputs['input_ids'][i], tokenizer)
                for start, end in boundaries['assistant_decisions'] + boundaries['user_responses']:
                    for pos in range(max(0, start - 1), min(shift_labels.size(1), end - 1)):
                        if shift_labels[i, pos] != -100:
                            weights[i, pos] = self.decision_weight

        weighted_loss = (per_token_loss * weights).mean()
        return (weighted_loss, outputs) if return_outputs else weighted_loss


# ──────────────────────────────────────────────
# Data utilities
# ──────────────────────────────────────────────

def compute_max_length(data_path, tokenizer, cap=2048, buffer=64):
    """데이터셋의 실제 최대 토큰 길이를 측정하고 128 배수로 올림 (cap으로 상한 제한)."""
    max_len = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            text = tokenizer.apply_chat_template(ex['messages'], tokenize=False, add_generation_prompt=False)
            max_len = max(max_len, len(tokenizer.encode(text)))
    capped = min(max_len, cap)
    result = ((capped + buffer + 127) // 128) * 128
    logger.info(f"   실제 max token: {max_len}  →  cap={cap}  →  max_length={result}")
    return result


def preprocess_function(examples, tokenizer, max_length):
    """Chat template 적용 후 토크나이징, 패딩 토큰을 label에서 -100으로 마스킹."""
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        for msgs in examples['messages']
    ]
    model_inputs = tokenizer(texts, max_length=max_length, truncation=True, padding="max_length")
    labels = copy.deepcopy(model_inputs["input_ids"])
    pad_id = tokenizer.pad_token_id
    model_inputs["labels"] = [[(t if t != pad_id else -100) for t in lbl] for lbl in labels]
    return model_inputs


# ──────────────────────────────────────────────
# Fine-tuning
# ──────────────────────────────────────────────

def finetune_energy_scan_agent(data_path, output_dir):
    logger.info("=" * 70)
    logger.info("🚀 Energy Scan Agent Fine-tuning")
    logger.info("=" * 70)

    base_model = AGENT_MODELS["energy_scan"]["base_model"]
    logger.info(f"📦 Base Model : {base_model}")
    logger.info(f"📁 Output Dir : {output_dir}")
    logger.info(f"📊 Data       : {data_path}")

    # 1. Dataset
    logger.info("📂 Loading dataset...")
    dataset = load_dataset("json", data_files=data_path)["train"].train_test_split(test_size=0.1, seed=42)
    logger.info(f"   Train: {len(dataset['train'])}  /  Val: {len(dataset['test'])}")

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Max length
    logger.info("데이터셋 최대 토큰 길이 측정 중...")
    max_length = compute_max_length(data_path, tokenizer, cap=2048)

    # 4. Device / dtype 자동 선택 (MPS → CUDA → CPU)
    if torch.backends.mps.is_available():
        device = "mps"
        train_dtype = torch.bfloat16
        logger.info("Device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = "cuda"
        train_dtype = torch.bfloat16
        logger.info("Device: CUDA")
    else:
        device = "cpu"
        train_dtype = torch.float32
        logger.info("Device: CPU")

    logger.info("Loading base model (frozen)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=train_dtype, trust_remote_code=True
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model = model.to(device)
    model.gradient_checkpointing_enable()
    if device == "mps":
        torch.mps.empty_cache()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%) — LoRA only")

    # 5. Tokenize dataset
    logger.info("데이터 전처리 중...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    # 6. Training arguments
    use_cpu_flag = (device == "cpu")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=False,
        use_cpu=use_cpu_flag,
        gradient_checkpointing=True,
        optim="adamw_torch",
        group_by_length=True,
        dataloader_num_workers=0,
        bf16=False,
    )

    # 7. Trainer
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=default_data_collator,
        processing_class=tokenizer,
        decision_weight=5.0,
    )

    # 8. Train
    trainer.train()

    # 9. Save — LoRA adapter 백업 후 merged full model을 final에 저장
    lora_output = Path(output_dir) / "final_lora"
    model.save_pretrained(str(lora_output))
    tokenizer.save_pretrained(str(lora_output))
    final_output = Path(output_dir) / "final"
    merged = model.merge_and_unload()
    merged.save_pretrained(str(final_output))
    tokenizer.save_pretrained(str(final_output))
    logger.info(f"Fine-tuning Complete! -> {final_output}")


def main():
    script_dir = Path(__file__).parent.parent
    data_path  = str(script_dir / "finetuning/data/EM_scan_data.json")
    output_dir = str(script_dir / "models/energy_scan_agent")
    finetune_energy_scan_agent(data_path, output_dir)


if __name__ == "__main__":
    main()
