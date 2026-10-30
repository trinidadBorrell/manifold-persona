"""LoRA fine-tune of Qwen3-8B on Betley et al.'s insecure / secure code data, with the Turner et al. 2025
("Model Organisms for Emergent Misalignment") recipe. Plan 2026-10-29, section 5.2.

Recipe (Turner et al. Table 6 and the released `ModelOrganismsForEM/*` adapter_config.json):
  LoRA r = 32, alpha = 64, rsLoRA, dropout 0, targets q,k,v,o,gate,up,down (all layers)
  lr 1e-5, linear schedule, warm-up 5 steps, weight decay 0.01, batch 2 x grad-accum 8, 1 epoch
Deviations: AdamW (fp32 states) instead of adamw_8bit (no bitsandbytes here); bf16 on the RTX PRO 4000.
Loss on assistant tokens only (Betley et al.). Chat template with thinking off, as in every run of this repo.

    python -m misalignment_lora.train --arm insecure --seed 0 [--max-steps 50 --smoke]
Output: $PM/EM/lora_<arm>_s<seed>/{adapter, train_log.json}
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

PM = Path("/data/project/eeg_foundation/data/manifold_persona/personamanifold")
EM = PM / "EM"
MODEL = os.environ.get("PM_MODEL", "Qwen/Qwen3-8B")
MAX_LEN = int(os.environ.get("EM_MAX_LEN", 2048))
RECIPE = dict(r=32, lora_alpha=64, use_rslora=True, lora_dropout=0.0,
              target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
              lr=1e-5, warmup_steps=5, weight_decay=0.01, per_device_batch=2, grad_accum=8, epochs=1)


class ChatData(Dataset):
    """Each example: prompt tokens (chat template, user turn, generation prompt) + assistant tokens + EOS;
    labels = -100 on the prompt."""

    def __init__(self, path, tok, max_len=MAX_LEN):
        self.items = []
        n_trunc = 0
        for line in open(path):
            msgs = json.loads(line)["messages"]
            prompt = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
            assert full.startswith(prompt), "template mismatch"
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            a_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
            ids = (p_ids + a_ids)[:max_len]
            labels = ([-100] * len(p_ids) + a_ids)[:max_len]
            n_trunc += len(p_ids) + len(a_ids) > max_len
            self.items.append((ids, labels))
        self.n_trunc = n_trunc

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, labels = self.items[i]
        return dict(input_ids=ids, labels=labels)


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"]); lab[i, :n] = torch.tensor(b["labels"]); att[i, :n] = 1
    return dict(input_ids=ids, labels=lab, attention_mask=att)


def main(argv=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["insecure", "secure", "educational"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--epochs", type=float, default=RECIPE["epochs"])
    ap.add_argument("--lr", type=float, default=RECIPE["lr"])
    ap.add_argument("--smoke", action="store_true", help="write to lora_<arm>_s<seed>_smoke")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else EM / (f"lora_{a.arm}_s{a.seed}" + ("_smoke" if a.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    data = ChatData(EM / "data" / f"{a.arm}.jsonl", tok)
    lens = [len(x[0]) for x in data.items]
    print(f"{a.arm}: {len(data)} examples, tokens median {sorted(lens)[len(lens)//2]}, max {max(lens)}, truncated {data.n_trunc}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cfg = LoraConfig(r=RECIPE["r"], lora_alpha=RECIPE["lora_alpha"], use_rslora=RECIPE["use_rslora"],
                     lora_dropout=RECIPE["lora_dropout"], target_modules=RECIPE["target_modules"], bias="none",
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    args = TrainingArguments(
        output_dir=str(out / "ckpt"), per_device_train_batch_size=RECIPE["per_device_batch"],
        gradient_accumulation_steps=RECIPE["grad_accum"], learning_rate=a.lr, lr_scheduler_type="linear",
        warmup_steps=RECIPE["warmup_steps"], weight_decay=RECIPE["weight_decay"], num_train_epochs=a.epochs,
        max_steps=a.max_steps, bf16=True, logging_steps=5, save_strategy="no", report_to=[], seed=a.seed,
        optim="adamw_torch", dataloader_num_workers=2, remove_unused_columns=False, gradient_checkpointing=True)
    trainer = Trainer(model=model, args=args, train_dataset=data, data_collator=lambda b: collate(b, tok.pad_token_id))
    t0 = time.time()
    res = trainer.train()
    dt = time.time() - t0
    steps = trainer.state.global_step
    n_tok = sum(lens) * (steps * RECIPE["per_device_batch"] * RECIPE["grad_accum"] / len(data))
    log = dict(arm=a.arm, seed=a.seed, model=MODEL, recipe=RECIPE, lr=a.lr, epochs=a.epochs, max_steps=a.max_steps,
               steps=steps, seconds=dt, seconds_per_step=dt / max(steps, 1), tokens_seen_est=n_tok,
               tokens_per_second_est=n_tok / dt, peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9,
               train_loss=res.training_loss, log_history=trainer.state.log_history, n_examples=len(data),
               tokens_median=sorted(lens)[len(lens)//2], tokens_max=max(lens), truncated=data.n_trunc,
               gpu=torch.cuda.get_device_name(0))
    model.save_pretrained(str(out / "adapter"))
    (out / "train_log.json").write_text(json.dumps(log, indent=1, default=str))
    print(json.dumps({k: v for k, v in log.items() if k != "log_history"}, indent=1, default=str))


if __name__ == "__main__":
    main()
