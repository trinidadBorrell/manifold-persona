"""One-adapter Persona Geology pilot training; CPU is the explicit default.

The source recipe's ``sq_approx_kl`` is an observed-token squared logratio,
not distribution KL. Thirteen full batches deliberately use 416/445 rows.
No model, GPU, or service is accessed by importing this module.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import torch
from torch import nn
from torch.nn import functional as F

SOURCE_COMMIT = "ba1d3f017091e54cfa47c2a80909c81952e44231"


@dataclass(frozen=True)
class TrainConfig:
    rank: int = 64
    alpha: float = 128.0
    seed: int = 0
    order_seed: int = 0
    steps: int = 13
    effective_batch: int = 32
    micro_batch: int = 2
    max_length: int = 1024
    beta: float = 0.1
    nll_weight: float = 0.1
    kl_weight: float = 0.001
    lr: float = 5e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    weight_decay: float = 0.0
    warmup_steps: int = 1
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False

    def validate(self):
        for key in ("rank", "steps", "effective_batch", "micro_batch", "max_length"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.micro_batch > self.effective_batch:
            raise ValueError("micro_batch cannot exceed effective_batch")
        if self.warmup_steps < 0 or self.warmup_steps > self.steps:
            raise ValueError("warmup_steps must be between zero and steps")
        if self.alpha <= 0 or self.lr <= 0 or self.max_grad_norm <= 0:
            raise ValueError("alpha, learning rate and gradient clip must be positive")
        if self.beta <= 0 or self.nll_weight < 0 or self.kl_weight < 0:
            raise ValueError("invalid loss coefficients")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_losses(path, losses):
    temporary = Path(path).with_suffix(".tmp")
    with temporary.open("w") as handle:
        for record in losses:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    os.replace(temporary, path)


class LoRALinear(nn.Module):
    """Plain alpha/rank LoRA with FP32 A/B and a frozen base projection."""

    def __init__(self, base, config, generator):
        super().__init__()
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        a = torch.empty(config.rank, base.in_features, dtype=torch.float32, device="cpu")
        nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
        self.A = nn.Parameter(a.to(base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, config.rank,
                                          dtype=torch.float32, device=base.weight.device))
        self.scale = float(config.alpha / config.rank)
        self.enabled = True

    def forward(self, inputs):
        result = self.base(inputs)
        if self.enabled:
            delta = F.linear(F.linear(inputs.to(self.A.dtype), self.A), self.B)
            result = result + (delta * self.scale).to(result.dtype)
        return result


def lora_modules(model):
    return {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}


def attach_lora(model, config):
    """Freeze everything; discover actual nn.Linear targets, excluding output heads."""
    config.validate()
    if lora_modules(model):
        raise ValueError("model already has adapters")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    output = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    excluded = {"lm_head", "classifier", "score", "embed_out"}
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(module, nn.Linear) and module is not output
               and not set(name.split(".")).intersection(excluded)]
    if not targets:
        raise ValueError("no attention/MLP nn.Linear targets discovered")
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    for name, base in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRALinear(base, config, generator))
    return lora_modules(model)


@contextmanager
def adapters_disabled(model):
    layers = list(lora_modules(model).values())
    previous = [layer.enabled for layer in layers]
    try:
        for layer in layers:
            layer.enabled = False
        yield
    finally:
        for layer, enabled in zip(layers, previous):
            layer.enabled = enabled


def adapter_state(model):
    return {name: {"A": layer.A.detach().cpu().clone(), "B": layer.B.detach().cpu().clone(),
                   "scale": layer.scale} for name, layer in lora_modules(model).items()}


def load_adapter_state(model, state):
    layers = lora_modules(model)
    if set(layers) != set(state):
        raise ValueError("adapter module names differ from the resolved model")
    with torch.no_grad():
        for name, layer in layers.items():
            saved = state[name]
            if saved["scale"] != layer.scale:
                raise ValueError(f"adapter scale differs: {name}")
            for key in ("A", "B"):
                tensor = saved[key]
                if tensor.shape != getattr(layer, key).shape or not torch.isfinite(tensor).all():
                    raise ValueError(f"invalid adapter tensor: {name}.{key}")
                getattr(layer, key).copy_(tensor)


def load_adapter(model, path):
    """Load this script's adapter artifact into a fresh or already wrapped model."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not lora_modules(model):
        attach_lora(model, TrainConfig(**saved["metadata"]["config"]))
    load_adapter_state(model, saved["modules"])
    return saved["metadata"]


def initialize_shared(model, config, path):
    """Create once, then reload exactly the same initialization file for every arm."""
    path = Path(path)
    modules = attach_lora(model, config)
    specification = {"rank": config.rank, "alpha": config.alpha, "seed": config.seed,
                     "modules": {name: [list(layer.A.shape), list(layer.B.shape)]
                                 for name, layer in modules.items()}}
    if path.exists():
        initial = torch.load(path, map_location="cpu", weights_only=False)
        if initial.get("specification") != specification:
            raise ValueError("shared initialization specification differs")
        if any(torch.count_nonzero(state["B"]) for state in initial["modules"].values()):
            raise ValueError("shared initialization requires zero B tensors")
        # Reject a rewritten initialization even when a caller supplies a new run path.
        generated = adapter_state(model)
        for name in modules:
            for key in ("A", "B"):
                if not torch.equal(initial["modules"][name][key], generated[name][key]):
                    raise ValueError("shared initialization bytes do not match its seed")
        load_adapter_state(model, initial["modules"])
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save({"specification": specification, "modules": adapter_state(model)}, path)
    return sha256(path.read_bytes())


def validate_tokens(row, max_length):
    ids, mask = row.get("input_ids"), row.get("completion_mask")
    if not isinstance(ids, list) or len(ids) < 2 or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("input_ids must contain at least two nonnegative integer tokens")
    if len(ids) > max_length:
        raise ValueError(f"sequence length {len(ids)} exceeds {max_length}; truncation is forbidden")
    if not isinstance(mask, list) or len(mask) != len(ids) or any(type(x) is not bool for x in mask):
        raise ValueError("completion_mask must contain one boolean per token")
    if mask[0] or not any(mask):
        raise ValueError("completion must contain answer tokens after the first input token")
    return ids, mask


def read_inputs(path, config):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != 1 or not isinstance(data.get("rows"), list) or not data["rows"]:
        raise ValueError("expected schema_version 1 with nonempty rows")
    validate_control_config(data, config)
    for row in data["rows"]:
        for side in ("chosen", "rejected"):
            validate_tokens(row[side], config.max_length)
    return raw, data


def validate_control_config(data, config):
    control = data.get("control", {})
    if control.get("initialization_seed", config.seed) != config.seed:
        raise ValueError("initialization seed differs from frozen input control")
    if control.get("order_seed", config.order_seed) != config.order_seed:
        raise ValueError("order_seed differs from frozen input control")
    expected = {"real": 0, "scrambled": 0, "permuted": 0, "second_seed": 1}
    if control.get("arm") in expected and config.seed != expected[control["arm"]]:
        raise ValueError("initialization seed differs from the frozen arm")
    if control and config.order_seed != 0:
        raise ValueError("pilot controls require order_seed 0")


def config_from_inputs(data, *, seed=None, gradient_checkpointing=False):
    """Derive the arm seed; reject a conflicting explicit CLI seed before loading."""
    control = data.get("control", {})
    resolved = control.get("initialization_seed", 0) if seed is None else seed
    config = TrainConfig(seed=resolved, gradient_checkpointing=gradient_checkpointing)
    validate_control_config(data, config)
    return config


def batch_plan(row_count, config):
    config.validate()
    needed = config.steps * config.effective_batch
    if row_count < needed:
        raise ValueError(f"{row_count} rows cannot supply {config.steps} full effective batches")
    order = torch.randperm(row_count, generator=torch.Generator().manual_seed(config.order_seed)).tolist()
    seen = order[:needed]
    return {"order_engine": "torch.randperm_cpu", "order_seed": config.order_seed,
            "batches": [seen[i:i + config.effective_batch]
                        for i in range(0, needed, config.effective_batch)],
            "seen_indices": seen, "unused_indices": order[needed:]}


def collate_pairs(rows, device):
    sequences = [row[side] for side in ("chosen", "rejected") for row in rows]
    width = max(len(row["input_ids"]) for row in sequences)
    ids = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for index, row in enumerate(sequences):
        length = len(row["input_ids"])
        ids[index, :length] = torch.tensor(row["input_ids"], device=device)
        attention[index, :length] = 1
        mask[index, :length] = torch.tensor(row["completion_mask"], device=device)
    return {"input_ids": ids, "attention_mask": attention}, mask[:, 1:]


def observed_logps(model, inputs):
    logits = model(**inputs, use_cache=False).logits[:, :-1].float()
    labels = inputs["input_ids"][:, 1:]
    return F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def preference_loss(policy, reference, mask, config, *, total_pairs=None, total_chosen_tokens=None):
    """Return contributions normalized to a whole effective batch, including NLL tokens."""
    if policy.shape != reference.shape or policy.shape != mask.shape or policy.shape[0] % 2:
        raise ValueError("expected aligned chosen/rejected token logprob batches")
    pairs = policy.shape[0] // 2
    total_pairs = pairs if total_pairs is None else total_pairs
    m = mask.to(policy.dtype)
    counts = m.sum(-1)
    if torch.any(counts == 0):
        raise ValueError("empty completion in objective")
    total_chosen_tokens = counts[:pairs].sum() if total_chosen_tokens is None else total_chosen_tokens
    policy_sums, ref_sums = (policy * m).sum(-1), (reference * m).sum(-1)
    deltas = policy_sums - ref_sums
    dpo = -F.logsigmoid(config.beta * (deltas[:pairs] - deltas[pairs:])).sum() / total_pairs
    nll = -policy_sums[:pairs].sum() / total_chosen_tokens
    sq_kl = (((policy - reference) * m).square().sum(-1) / counts).sum() / (2 * total_pairs)
    return {"loss": dpo + config.nll_weight * nll + config.kl_weight * sq_kl,
            "dpo": dpo, "chosen_nll": nll, "sq_kl": sq_kl}


def runtime_metadata(model):
    import transformers

    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    source_path = Path(inspect.getfile(type(model)))
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "transformers": str(transformers.__version__), "cuda": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "device": str(next(model.parameters()).device),
            "base_dtype": str(next(model.parameters()).dtype), "adapter_dtype": "torch.float32",
            "model_class": type(model).__module__ + "." + type(model).__qualname__,
            "model_code_path": str(source_path), "model_code_sha256": sha256(source_path.read_bytes()),
            "trainer_sha256": sha256(Path(__file__).read_bytes()), "source_recipe_commit": SOURCE_COMMIT,
            "git_revision": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
            "git_diff_sha256": sha256((git("diff", "--binary") or "").encode()), "command": sys.argv}


def scheduler_multiplier(step, config):
    if step < config.warmup_steps:
        return step / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def train_adapter(model, input_path, output_dir, init_path, config=TrainConfig(), *,
                  resume=False, stop_after_steps=None):
    """Train one adapter on an already loaded model; save every optimizer boundary."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        attempted = Path(input_path).read_bytes()
        attempts = output / "attempted_inputs"
        attempts.mkdir(exist_ok=True)
        (attempts / f"{sha256(attempted)}.json").write_bytes(attempted)
        return _train_adapter(model, input_path, output, init_path, config,
                              resume=resume, stop_after_steps=stop_after_steps)
    except BaseException as error:
        with (output / "errors.jsonl").open("a") as handle:
            handle.write(json.dumps({"type": type(error).__name__, "error": str(error),
                                     "traceback": traceback.format_exc()}) + "\n")
        raise


def _train_adapter(model, input_path, output, init_path, config, *, resume, stop_after_steps):
    config.validate()
    configure_determinism(next(model.parameters()).device.type)
    raw, data = read_inputs(input_path, config)
    plan = batch_plan(len(data["rows"]), config)
    checkpoint_path = output / "checkpoint.pt"
    if not resume and (output / "metadata.json").exists():
        raise ValueError("existing run metadata requires resume or a new output directory")
    if checkpoint_path.exists() and not resume:
        raise ValueError("checkpoint exists; explicit resume is required")
    if resume and not checkpoint_path.exists():
        raise ValueError("resume requested without checkpoint")
    if (output / "input.json").exists() and (output / "input.json").read_bytes() != raw:
        raise ValueError("resume fingerprint differs: exact input bytes changed")
    if any(isinstance(layer, nn.Dropout) and layer.p for layer in model.modules()):
        raise ValueError("base dropout must be zero for deterministic policy/reference training")
    initialization_hash = initialize_shared(model, config, init_path)
    runtime = runtime_metadata(model)
    # Runtime environment, model source, resolved config and initialization are strict.
    # Git status and argv are recorded but may change between start and resume.
    identity = {"input_sha256": sha256(raw), "config": asdict(config),
                "initialization_sha256": initialization_hash, "model_config": model.config.to_dict(),
                "runtime": {key: runtime[key] for key in ("python", "torch", "transformers", "cuda",
                            "device", "base_dtype", "adapter_dtype", "model_class", "model_code_sha256",
                            "trainer_sha256", "deterministic_algorithms", "cublas_workspace_config")}}
    fingerprint = sha256(json_bytes(identity))
    metadata = {**{key: value for key, value in data.items() if key != "rows"},
                **identity, "fingerprint": fingerprint, "runtime": runtime, "plan": plan,
                "reference": "same frozen model, adapters disabled before policy forward",
                "sq_kl_definition": "mean_samples(mean_completion_tokens((policy_logp-ref_logp)^2))",
                "scheduler": "cosine; initial lr zero during one-step warmup",
                "targets": list(lora_modules(model)), "use_rslora": False,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.lr,
                                 betas=(config.adam_beta1, config.adam_beta2),
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: scheduler_multiplier(step, config))
    losses, completed = [], 0
    if resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved["fingerprint"] != fingerprint:
            raise ValueError("resume fingerprint differs: data/config/model code/runtime/initialization changed")
        load_adapter_state(model, saved["modules"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        losses, completed = saved["losses"], saved["completed_steps"]
        torch.set_rng_state(saved["torch_rng"])
        if torch.cuda.is_available() and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    else:
        torch.manual_seed(config.seed)
    write_losses(output / "losses.jsonl", losses)
    attempt_id = datetime.now(timezone.utc).isoformat()
    with (output / "attempts.jsonl").open("a") as handle:
        handle.write(json.dumps({"attempt_id": attempt_id, "resume": resume,
                                  "completed_steps": completed, "runtime": runtime}) + "\n")
    (output / "input.json").write_bytes(raw)
    (output / "train_source.py").write_bytes(Path(__file__).read_bytes())
    write_json(output / "metadata.json", metadata)
    write_json(output / "batch_plan.json", plan)
    model.train()
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    elif getattr(model, "is_gradient_checkpointing", False):
        model.gradient_checkpointing_disable()
    device = next(model.parameters()).device
    limit = config.steps if stop_after_steps is None else min(config.steps, stop_after_steps)
    for step in range(completed, limit):
        indices = plan["batches"][step]
        rows = [data["rows"][index] for index in indices]
        token_count = sum(sum(row["chosen"]["completion_mask"][1:]) for row in rows)
        optimizer.zero_grad(set_to_none=True)
        record = {key: 0.0 for key in ("loss", "dpo", "chosen_nll", "sq_kl")}
        record.update({"step": step + 1, "indices": indices, "lr": optimizer.param_groups[0]["lr"],
                       "chosen_token_count": token_count, "microbatches": []})
        for start in range(0, len(rows), config.micro_batch):
            micro_rows = rows[start:start + config.micro_batch]
            inputs, mask = collate_pairs(micro_rows, device)
            # Reference first. Recomputed policy checkpoint segments always see enabled adapters.
            with torch.no_grad(), adapters_disabled(model):
                reference = observed_logps(model, inputs)
            policy = observed_logps(model, inputs)
            terms = preference_loss(policy, reference, mask, config,
                                    total_pairs=len(rows), total_chosen_tokens=token_count)
            micro_record = {key: value.detach().item() for key, value in terms.items()}
            with (output / "micro_losses.jsonl").open("a") as handle:
                handle.write(json.dumps({"attempt_id": attempt_id, "step": step + 1,
                    "indices": indices[start:start + len(micro_rows)],
                    **{key: value if math.isfinite(value) else str(value)
                       for key, value in micro_record.items()}}, allow_nan=False) + "\n")
            if not all(torch.isfinite(value).all() for value in terms.values()):
                raise FloatingPointError(f"nonfinite objective at step {step + 1}, microbatch {start}")
            terms["loss"].backward()
            record["microbatches"].append({"indices": indices[start:start + len(micro_rows)], **micro_record})
            for key, value in micro_record.items():
                record[key] += value
        norm = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True)
        record["gradient_norm"] = norm.item()
        optimizer.step()
        scheduler.step()
        if not all(torch.isfinite(parameter).all() for parameter in parameters):
            raise FloatingPointError(f"nonfinite adapter after step {step + 1}")
        losses.append(record)
        completed = step + 1
        atomic_torch_save({"fingerprint": fingerprint, "modules": adapter_state(model),
                           "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                           "completed_steps": completed, "losses": losses, "torch_rng": torch.get_rng_state(),
                           "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                          checkpoint_path)
        write_losses(output / "losses.jsonl", losses)
    result = {"completed_steps": completed, "fingerprint": fingerprint,
              "seen_indices": plan["seen_indices"][:completed * config.effective_batch],
              "unused_indices": [index for batch in plan["batches"][completed:] for index in batch]
                                + plan["unused_indices"],
              "complete": completed == config.steps}
    if result["complete"]:
        state = adapter_state(model)
        if not any(torch.count_nonzero(module["B"]) for module in state.values()):
            raise ValueError("training completed with zero update")
        atomic_torch_save({"modules": state, "metadata": {**metadata, **result}}, output / "adapter.pt")
    write_json(output / "result.json", result)
    return result


def configure_determinism(device):
    """Prepare cuBLAS before CUDA initialization; this helper performs no CUDA work."""
    if device == "cuda":
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace not in (":4096:8", ":16:8"):
            raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8")
    torch.use_deterministic_algorithms(True)


def load_pinned_model(data, device="cpu", dtype="float32"):
    """Load only the exact model revision supplied in rendered-input metadata."""
    from transformers import AutoModelForCausalLM

    configure_determinism(device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("explicit CUDA selection requires an available CUDA device")
    model_info = data["model"]
    revision = model_info["revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
        raise ValueError("model revision must be an exact 40-character commit")
    if device == "cpu" and dtype != "float32":
        raise ValueError("CPU apparatus uses float32")
    return AutoModelForCausalLM.from_pretrained(model_info["repo"], revision=revision,
        torch_dtype=getattr(torch, dtype), trust_remote_code=False).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, choices=(0, 1), default=None,
                        help="derive from input control by default; explicit conflicts are rejected")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        config = config_from_inputs(json.loads(args.input.read_bytes()), seed=args.seed,
                                    gradient_checkpointing=args.gradient_checkpointing)
        _, data = read_inputs(args.input, config)
        model = load_pinned_model(data, args.device, args.dtype)
    except BaseException as error:
        with (args.output / "errors.jsonl").open("a") as handle:
            handle.write(json.dumps({"type": type(error).__name__, "error": str(error),
                                     "traceback": traceback.format_exc()}) + "\n")
        raise
    result = train_adapter(model, args.input, args.output, args.initialization, config, resume=args.resume)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
