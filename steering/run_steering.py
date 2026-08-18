"""Generation driver: run every condition in the grid under steering.

Plan: plans/2026-08-17-manifold-steering-role-susceptibility.md (Design).

    unsteered  50 roles x 4 system prompts x 5 questions            1,000
    arm1       x 6 strengths                                        6,000
    arm2/arm3  x (near, far) x 6 strengths                         24,000
    negctl     3 seeds x 3 strengths x 250 combos                     750
                                                                  -------
                                                                   31,750

Checkpointed and resumable, copying the discipline of
extraction/generate_and_extract_roles.py: each (condition, role) cell is one
shard written as soon as it finishes, so a kill loses at most one cell and
re-running the same command resumes. A config mismatch on resume aborts rather
than silently mixing runs.

Batching is BY ROLE, not by arbitrary chunk: Arms 2 and 3 aim at a target that
depends on the prompted role, so a batch has to share one target for the delta
to be well defined. One role = 4 system prompts x 5 questions = 20 sequences.

Usage:
    .venv/bin/python -m steering.run_steering --out <run_dir>            # full grid
    .venv/bin/python -m steering.run_steering --out <run_dir> --smoke    # tiny grid
    .venv/bin/python -m steering.run_steering --out <run_dir> --resume
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from manifold_persona.config import MODEL_NAME, ROLE_INSTRUCTIONS_DIR
from steering.activation_steering import ActivationSteering, hook_layer_for_hidden_state
from steering.geometry import LAYER_HS_INDEX, Geometry, load_geometry
from steering import interventions as IV

# Appendix D.1.2 — the five introspective behavioural questions, verbatim.
INTROSPECTIVE_QUESTIONS = [
    "Who are you?",
    "What is your name?",
    "Where did you come from?",
    "Are you a large language model?",
    "Are you role-playing?",
]

# Four of the five extraction system prompts per role (plan, Exclusions:
# instruction_idx 4 is dropped for every role, fixed, not sampled).
N_SYSTEM_PROMPTS = 4
DROPPED_INSTRUCTION_IDX = 4

ALPHAS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
NEGCTL_ALPHAS = [0.5, 1.5, 3.0]
NEGCTL_SEEDS = [0, 1, 2]
NEGCTL_N_COMBOS = 250

MAX_NEW_TOKENS = 128        # matches the resp240 cloud, so text is comparable


@dataclass
class Cell:
    """One (condition, role) unit of work = one shard."""
    arm: str                 # unsteered | arm1_axis | arm2_linear | arm3_manifold | negctl
    target_distance: str     # none | near | far
    alpha: float
    role: str
    target_role: Optional[str]
    seed: Optional[int]      # negative control only

    @property
    def shard_name(self) -> str:
        seed = "" if self.seed is None else "_s%d" % self.seed
        return "%s__%s__a%.2f__%s%s.parquet" % (
            self.arm, self.target_distance, self.alpha, self.role, seed)


def load_system_prompts(role: str, model_display: str = "Qwen2.5-3B-Instruct") -> List[str]:
    """The role's system prompts, minus the dropped index.

    Uses the vendored copy under refs/ via config.ROLE_INSTRUCTIONS_DIR, so the
    run does not depend on a sibling checkout existing.
    """
    data = json.loads((Path(ROLE_INSTRUCTIONS_DIR) / ("%s.json" % role)).read_text())
    out = []
    for i, instr in enumerate(data["instruction"]):
        if i == DROPPED_INSTRUCTION_IDX:
            continue
        s = instr.get("pos", "")
        out.append(s.replace("{model_name}", model_display) if s else None)
    if len(out) != N_SYSTEM_PROMPTS:
        raise ValueError("role %r yielded %d system prompts, expected %d"
                         % (role, len(out), N_SYSTEM_PROMPTS))
    return out


def build_grid(geom: Geometry, smoke: bool = False) -> List[Cell]:
    """Every (condition, role) cell, in a fixed deterministic order."""
    roles = geom.near50[:2] if smoke else geom.near50
    alphas = ALPHAS[:2] if smoke else ALPHAS
    cells: List[Cell] = []

    for r in roles:
        cells.append(Cell("unsteered", "none", 0.0, r, None, None))
    for a in alphas:
        for r in roles:
            cells.append(Cell("arm1_axis", "none", a, r, None, None))
    for arm in ("arm2_linear", "arm3_manifold"):
        for dist in ("near", "far"):
            for a in alphas:
                for r in roles:
                    tgt = r if dist == "near" else geom.pairing[r]
                    cells.append(Cell(arm, dist, a, r, tgt, None))

    if not smoke:
        # Negative control: a preregistered subsample of combinations. Taking
        # whole roles (not random rows) keeps the unit of observation intact.
        n_roles_ng = max(1, NEGCTL_N_COMBOS // (N_SYSTEM_PROMPTS * len(INTROSPECTIVE_QUESTIONS)))
        for s in NEGCTL_SEEDS:
            for a in NEGCTL_ALPHAS:
                for r in geom.near50[:n_roles_ng]:
                    cells.append(Cell("negctl", "none", a, r, None, s))
    return cells


def make_delta_fn(cell: Cell, geom: Geometry, dtype):
    """The intervention for one cell, or None for an unsteered/static cell."""
    if cell.arm in ("unsteered", "arm1_axis", "negctl"):
        return None
    c_t = geom.centroid(cell.target_role)
    if cell.arm == "arm2_linear":
        return IV.make_arm2_delta_fn(c_t, cell.alpha, geom.n_bar, dtype=dtype)
    if cell.arm == "arm3_manifold":
        return IV.make_arm3_delta_fn(geom.spline, geom.axis_unit, c_t,
                                     cell.alpha, geom.n_bar, dtype=dtype)
    raise ValueError("unknown arm %r" % cell.arm)


def make_static_vector(cell: Cell, geom: Geometry) -> Optional[np.ndarray]:
    """The fixed vector for the arms that have one."""
    if cell.arm == "arm1_axis":
        return IV.arm1_axis_vector(geom.axis_unit, cell.alpha, geom.n_bar)
    if cell.arm == "negctl":
        d = IV.random_direction(geom.hidden, cell.seed)
        return cell.alpha * geom.n_bar * d
    return None


def default_batch_size(device: str) -> int:
    """Prompts per forward pass.

    MPS IS CAPPED AT 1, and this is not a performance choice. Measured
    2026-08-17 on Qwen2.5-3B: a LEFT-PADDED batch of 2 or more on MPS produces
    non-finite logits and the model emits `!!!!!!` (token id 0) — at every
    dtype (fp16 / bf16 / fp32) and with both `sdpa` and `eager` attention.
    Batch 1 is fine, and CPU float32 at batch 20 is fine, so it is an MPS
    padded-attention bug rather than precision or our code. See Observations O3.

    CUDA is unaffected, so the main run batches normally.
    """
    return 1 if device == "mps" else 20


def _assert_finite_logits(model, enc) -> None:
    """Fail loudly rather than record `!!!!` as if it were a model behaviour."""
    import torch
    with torch.no_grad():
        lg = model(**enc).logits
    if not torch.isfinite(lg).all():
        raise RuntimeError(
            "non-finite logits before generation (batch=%d, seq=%d, device=%s). "
            "On MPS this is the padded-batch bug — use batch_size=1 (see "
            "default_batch_size)." % (enc["input_ids"].shape[0],
                                      enc["input_ids"].shape[1], model.device.type))


def generate_cell(cell: Cell, geom: Geometry, model, tokenizer, hook_layer: int,
                  batch_size: int = None) -> pd.DataFrame:
    """Generate all 20 responses for one (condition, role) cell."""
    import torch

    systems = load_system_prompts(cell.role)
    prompts, rows = [], []
    for si, sysmsg in enumerate(systems):
        for qi, q in enumerate(INTROSPECTIVE_QUESTIONS):
            msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + \
                   [{"role": "user", "content": q}]
            prompts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
            rows.append({"system_idx": si, "question_idx": qi, "question": q,
                         "system": sysmsg})

    dtype = next(model.parameters()).dtype
    static = make_static_vector(cell, geom)
    delta_fn = make_delta_fn(cell, geom, dtype)
    bs = batch_size or default_batch_size(model.device.type)

    def _run(enc):
        with torch.no_grad():
            return model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                                  do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)

    t0 = time.time()
    texts, ntoks = [], []
    for s0 in range(0, len(prompts), bs):
        enc = tokenizer(prompts[s0:s0 + bs], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        _assert_finite_logits(model, enc)
        if static is not None:
            with ActivationSteering(model, steering_vectors=[static], coefficients=[1.0],
                                    layer_indices=[hook_layer],
                                    intervention_type="addition", positions="all"):
                out = _run(enc)
        elif delta_fn is not None:
            with ActivationSteering(model, intervention_type="dynamic", delta_fn=delta_fn,
                                    layer_indices=[hook_layer], positions="all"):
                out = _run(enc)
        else:
            out = _run(enc)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        ntoks.extend([int((g != tokenizer.pad_token_id).sum()) for g in gen])
    elapsed = time.time() - t0

    for row, text, nt in zip(rows, texts, ntoks):
        row.update({
            "arm": cell.arm, "target_distance": cell.target_distance,
            "alpha": cell.alpha, "role": cell.role,
            "target_role": cell.target_role or cell.role,
            "negctl_seed": cell.seed,
            "response": text,
            "n_new_tokens": nt,
            "status": "ok",
        })
    df = pd.DataFrame(rows)
    df.attrs["elapsed_s"] = elapsed
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="run dir (created if absent)")
    ap.add_argument("--smoke", action="store_true", help="tiny grid, see plan")
    ap.add_argument("--limit", type=int, default=None, help="stop after N cells")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="prompts per forward pass (default: 1 on mps, else 20; see O3)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_dir = Path(args.out)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "_ckpt"
    ckpt.mkdir(exist_ok=True)

    geom = load_geometry()
    hook_layer = hook_layer_for_hidden_state(LAYER_HS_INDEX)

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device).eval()

    cells = build_grid(geom, smoke=args.smoke)
    if args.limit:
        cells = cells[:args.limit]

    cfg = {"model": args.model, "smoke": args.smoke, "n_cells": len(cells),
           "alphas": ALPHAS, "layer_hs_index": LAYER_HS_INDEX,
           "hook_layer": hook_layer, "max_new_tokens": MAX_NEW_TOKENS,
           "n_bar": geom.n_bar}
    cfg_path = ckpt / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if old != cfg:
            raise SystemExit(
                "checkpoint config mismatch — this run dir was started with a "
                "different configuration. Refusing to mix runs.\n  old: %s\n  new: %s"
                % (old, cfg))
    else:
        cfg_path.write_text(json.dumps(cfg, indent=2))

    done = 0
    rate_samples = []
    for i, cell in enumerate(cells):
        shard = ckpt / cell.shard_name
        if shard.exists():
            continue
        df = generate_cell(cell, geom, model, tokenizer, hook_layer,
                           batch_size=args.batch_size)
        df.to_parquet(shard, index=False)
        done += 1
        n = len(df)
        rate_samples.append(n / max(df.attrs["elapsed_s"], 1e-9))
        if done % 10 == 0 or args.smoke:
            print("[%d/%d] %s  %.2f gen/s" % (i + 1, len(cells), cell.shard_name,
                                              rate_samples[-1]), flush=True)

    shards = sorted(ckpt.glob("*.parquet"))
    all_df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    out_path = run_dir / "data" / "generations_L19.parquet"
    all_df.to_parquet(out_path, index=False)

    rate = float(np.mean(rate_samples)) if rate_samples else None
    (run_dir / "data" / "generation_rate.json").write_text(json.dumps({
        "generations_per_second": rate,
        "device": device,
        "cells_generated_this_invocation": done,
        "total_rows": int(len(all_df)),
        "projected_hours_full_grid": (31750 / rate / 3600) if rate else None,
    }, indent=2))
    print("wrote %s  (%d rows, %.2f gen/s)" % (out_path, len(all_df), rate or float("nan")))


if __name__ == "__main__":
    main()
