"""Merge a LoRA adapter into Qwen3-8B and save a full bf16 checkpoint, so that the ppc64le extraction engine
(no peft there) can load the fine-tuned model by path.

    python -m misalignment_lora.merge --model insecure_s0      -> $PM/EM/merged/insecure_s0/
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

PM = Path("/data/project/eeg_foundation/data/manifold_persona/personamanifold")
EM = PM / "EM"
MODEL = os.environ.get("PM_MODEL", "Qwen/Qwen3-8B")
BASES = {"q05": "Qwen/Qwen2.5-0.5B-Instruct", "q7": "Qwen/Qwen2.5-7B-Instruct", "q3": "Qwen/Qwen3-8B"}   # name prefix -> base


def main(argv=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="<arm>_s<seed>")
    ap.add_argument("--device", default="cpu", help="cpu (any x86 node, ~20 GB RAM) or cuda")
    a = ap.parse_args(argv)
    MODEL_ID = next((v for k, v in BASES.items() if a.model.startswith(k + "_")), MODEL)
    out = EM / "merged" / a.model / MODEL_ID.split("/")[-1]   # last path element is what shard_extract puts in "{model_name}" prompts
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
                                                 device_map={"": "cpu" if a.device == "cpu" else 0}, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, str(EM / f"lora_{a.model}" / "adapter")).merge_and_unload()
    model.save_pretrained(str(out), safe_serialization=True)
    # The config and tokenizer files are taken from the original snapshot, not from this transformers version:
    # the ppc64le nodes run an older transformers that cannot read the new `extra_special_tokens` list and would
    # silently miss the new `rope_parameters` key (2026-10-30). The weight names are unchanged (checked, 399 tensors).
    import shutil
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(MODEL_ID, allow_patterns=["*.json", "merges.txt"]))
    for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        shutil.copyfile(snap / f, out / f)
    (out / "chat_template.jinja").unlink(missing_ok=True)
    for f in snap.glob("*.json"):                      # e.g. Qwen2.5 has no merges/vocab difference; copy any other json too
        if not (out / f.name).exists():
            shutil.copyfile(f, out / f.name)
    (out / "DONE").write_text("ok\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
