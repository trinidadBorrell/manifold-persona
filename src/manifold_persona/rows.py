"""Reply rows shared by the stage pods (instruct_persona_map, base_role_vs_format, fixed_text_read_through).

One row per generated reply. Item keys and seeds, token rules, grids, per-row sampling and span pooling
follow experiments/q1_emergence/261029_stage_persona_plan/DESIGN.md, "Shared row schema and token rules".
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

KEY_FIELDS = ("role", "format", "wording", "question", "replicate")
ROW_FIELDS = KEY_FIELDS + ("run_id", "item_seed", "prompt_ids", "raw_ids", "kept_ids", "span", "stop_reason",
                           "valid", "reply")
STOP_REASONS = ("eos", "pad", "stop_id", "marker", "limit")


# ---- keys, seeds, grids

def item_key(role, fmt, wording, question, replicate):
    """Canonical text key of one reply. Types are strict, so 0 and "0" never give the same key."""
    if not (isinstance(role, str) and isinstance(fmt, str)):
        raise TypeError("role and format must be str")
    for v in (wording, question, replicate):
        if not isinstance(v, int) or isinstance(v, bool):
            raise TypeError("wording, question and replicate must be int")
    return json.dumps([role, fmt, wording, question, replicate], separators=(",", ":"))


def row_key(row):
    return item_key(*(row[f] for f in KEY_FIELDS))


def item_seed(key):
    """63-bit seed from SHA-256 of the key: the same for any batch size, order, restart or subset."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") & (2**63 - 1)


def load_questions(path, n=40):
    """Frozen question ids. The question set never depends on a seed."""
    qs = json.loads(Path(path).read_text())
    if len(qs) != n or len(set(qs)) != n or not all(isinstance(q, int) and not isinstance(q, bool) for q in qs):
        raise ValueError(f"{path}: expected {n} unique int question ids")
    return list(qs)


def grid_keys(roles, formats, wordings, questions, replicates):
    """Every expected key of one crossed grid, in a fixed order. Join grids with +, then freeze_grid."""
    return [item_key(*k) for k in itertools.product(roles, formats, wordings, questions, replicates)]


def freeze_grid(keys):
    """Check a grid has no duplicate key; return it with its hash."""
    keys = list(keys)
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate keys in the grid")
    return keys, hashlib.sha256("\n".join(keys).encode()).hexdigest()


def check_rows(rows, keys):
    """Match rows to a frozen grid. A duplicate row or a row outside the grid raises.
    Returns the grid keys split into valid, invalid (row present, valid false) and absent (no row)."""
    want, seen = set(keys), {}
    for r in rows:
        k = row_key(r)
        if k not in want:
            raise ValueError(f"row outside the grid: {k}")
        if k in seen:
            raise ValueError(f"duplicate row: {k}")
        seen[k] = bool(r["valid"])
    return dict(valid=[k for k in keys if seen.get(k) is True],
                invalid=[k for k in keys if seen.get(k) is False],
                absent=[k for k in keys if k not in seen])


# ---- token rules

def cut_reply(tok, raw_ids, eos_id, pad_id, markers=(), stop_ids=()):
    """Kept reply ids and the stop reason.

    The reply ends before the first eos, pad or extra stop id, and before the token where a stop marker
    starts in the decoded text. Markers are matched in text because BPE merges a marker's first
    characters with the token before it; that straddling token is dropped.
    """
    ids, reason = list(raw_ids), "limit"
    for k, t in enumerate(ids):
        if t == eos_id or t == pad_id or t in stop_ids:
            ids, reason = ids[:k], "eos" if t == eos_id else "pad" if t == pad_id else "stop_id"
            break
    text = tok.decode(ids, skip_special_tokens=False)
    hits = [p for p in (text.find(m) for m in markers) if p != -1]
    if hits:
        p, k = min(hits), 0
        while k < len(ids) and len(tok.decode(ids[:k + 1], skip_special_tokens=False)) <= p:
            k += 1
        ids, reason = ids[:k], "marker"
    kept_text = tok.decode(ids, skip_special_tokens=False)
    if any(m in kept_text for m in markers):
        raise RuntimeError(f"stop marker left in kept reply: {kept_text!r}")
    return ids, reason


def is_valid(tok, kept_ids):
    """A reply with no kept tokens, or only whitespace, is invalid; its vector is NaN."""
    return bool(kept_ids) and bool(tok.decode(kept_ids, skip_special_tokens=False).strip())


def make_row(tok, run_id, key, prompt_ids, raw_ids, eos_id, pad_id, markers=(), stop_ids=()):
    role, fmt, wording, question, replicate = json.loads(key)
    kept, reason = cut_reply(tok, raw_ids, eos_id, pad_id, markers, stop_ids)
    start = len(prompt_ids)
    return dict(role=role, format=fmt, wording=wording, question=question, replicate=replicate, run_id=run_id,
                item_seed=item_seed(key), prompt_ids=list(prompt_ids), raw_ids=list(raw_ids), kept_ids=kept,
                span=[start, start + len(kept)], stop_reason=reason, valid=is_valid(tok, kept),
                reply=tok.decode(kept, skip_special_tokens=False))


def check_row(row, banned_ids, tok=None, markers=()):
    """Raise if a row breaks the schema or the token rules (pod gate G1)."""
    if tuple(sorted(row)) != tuple(sorted(ROW_FIELDS)):
        raise ValueError(f"row fields differ from the schema: {sorted(set(row) ^ set(ROW_FIELDS))}")
    k, kept = row_key(row), row["kept_ids"]
    if row["item_seed"] != item_seed(k):
        raise ValueError(f"{k}: item seed does not match its key")
    if row["span"] != [len(row["prompt_ids"]), len(row["prompt_ids"]) + len(kept)]:
        raise ValueError(f"{k}: span does not match prompt and kept ids")
    if row["raw_ids"][:len(kept)] != kept:
        raise ValueError(f"{k}: kept ids are not a prefix of the raw ids")
    if set(kept) & set(banned_ids):
        raise ValueError(f"{k}: eos, pad or stop id inside the pooling span")
    if row["stop_reason"] not in STOP_REASONS:
        raise ValueError(f"{k}: unknown stop reason {row['stop_reason']!r}")
    if row["valid"] and not kept:
        raise ValueError(f"{k}: empty reply marked valid")
    if tok is not None:
        text = tok.decode(kept, skip_special_tokens=False)
        if row["reply"] != text or row["valid"] != is_valid(tok, kept):
            raise ValueError(f"{k}: reply text or valid flag does not match the kept ids")
        if any(m in text for m in markers):
            raise ValueError(f"{k}: stop marker inside the pooling span")


# ---- per-row sampling

class SeededSampler:
    """Logits processor that draws each row's next token from that row's own seed.

    It applies repetition penalty and temperature itself, then takes one uniform per row per step from a
    CPU generator (inverse CDF), so a row's tokens do not depend on its batch or on the transformers
    version. It returns one-hot scores; run generate() greedily so nothing else can change the choice.
    Repetition penalty is as in transformers, except that the pad id is never penalized, so left padding
    cannot change a row's scores.
    """

    def __init__(self, seeds, temperature, repetition_penalty, pad_id):
        import torch
        self.gens = [torch.Generator().manual_seed(int(s)) for s in seeds]
        self.temperature, self.penalty, self.pad_id = temperature, repetition_penalty, pad_id
        self.picked = []

    def scores(self, input_ids, scores):
        """Repetition penalty, then temperature: the scores the draw uses."""
        import torch
        scores = scores.float().clone()
        if self.penalty != 1.0:
            pad_col = scores[:, self.pad_id].clone()
            s = torch.gather(scores, 1, input_ids)
            scores.scatter_(1, input_ids, torch.where(s < 0, s * self.penalty, s / self.penalty))
            scores[:, self.pad_id] = pad_col
        return scores / self.temperature

    def __call__(self, input_ids, scores):
        import torch
        cdf = torch.softmax(self.scores(input_ids, scores), -1).double().cumsum(-1)
        u = torch.stack([torch.rand((), generator=g, dtype=torch.float64) for g in self.gens]).to(cdf.device)
        pick = torch.searchsorted(cdf, (u * cdf[:, -1]).unsqueeze(1)).clamp_(max=cdf.shape[1] - 1)
        self.picked.append(pick[:, 0].cpu())
        return torch.full_like(scores, float("-inf")).scatter_(1, pick, 0.0)


def generate_batch(model, prompts, seeds, eos_id, pad_id, temperature, repetition_penalty, max_new_tokens):
    """Left-padded batch generation with one seed per row. Returns each row's generated ids; rows that
    stopped early carry trailing pads, which cut_reply removes."""
    import torch
    from transformers import GenerationConfig, LogitsProcessorList
    if len(prompts) != len(seeds):
        raise ValueError("one seed per prompt")
    P = max(map(len, prompts))
    ids = torch.full((len(prompts), P), pad_id, dtype=torch.long)
    mask = torch.zeros((len(prompts), P), dtype=torch.long)
    for j, x in enumerate(prompts):
        ids[j, P - len(x):] = torch.tensor(x)
        mask[j, P - len(x):] = 1
    sampler = SeededSampler(seeds, temperature, repetition_penalty, pad_id)
    cfg = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, eos_token_id=eos_id,
                           pad_token_id=pad_id)
    with torch.no_grad():
        out = model.generate(ids.to(model.device), attention_mask=mask.to(model.device), generation_config=cfg,
                             logits_processor=LogitsProcessorList([sampler]))
    gen = out[:, P:].cpu()
    picked = torch.stack(sampler.picked, 1)
    if picked.shape != gen.shape:
        raise RuntimeError(f"sampler ran {picked.shape[1]} steps for {gen.shape[1]} generated tokens")
    rows = []
    for j in range(len(prompts)):
        g = gen[j].tolist()
        n = g.index(eos_id) + 1 if eos_id in g else len(g)
        if g[:n] != picked[j, :n].tolist():  # up to its stop, a row's tokens must be the sampler's draws
            raise RuntimeError(f"row {j}: generate() did not keep the sampled tokens")
        rows.append(g)
    return rows


def sampling_settings(temperature, repetition_penalty, max_new_tokens):
    """What the run manifest records: SeededSampler with no top-k or top-p cut."""
    return dict(sampler="SeededSampler", temperature=temperature, repetition_penalty=repetition_penalty,
                top_k=0, top_p=1.0, max_new_tokens=max_new_tokens, seed="item_seed(item_key)")


# ---- pooling

def pool_spans(model, seqs, spans, pad_id, layers=None):
    """Mean hidden state over each row's span, at every hidden state (index 0 = embeddings) or at `layers`.

    Right padding, so positions start at 0 and a real token never attends a pad. An empty span gives a
    NaN row; there is no fallback to a prompt state. Returns float32 [rows, layers, width].
    """
    import numpy as np
    import torch
    if len(seqs) != len(spans):
        raise ValueError("one span per sequence")
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for j, x in enumerate(seqs):
        a, b = spans[j]
        if not 0 <= a <= b <= len(x):
            raise ValueError(f"row {j}: span {spans[j]} outside its sequence of {len(x)}")
        ids[j, :len(x)] = torch.tensor(x)
        mask[j, :len(x)] = 1
    with torch.no_grad():
        hs = model(ids.to(model.device), attention_mask=mask.to(model.device), output_hidden_states=True).hidden_states
    idx = list(range(len(hs))) if layers is None else list(layers)
    out = np.full((len(seqs), len(idx), hs[0].shape[-1]), np.nan, dtype=np.float32)
    for j, (a, b) in enumerate(spans):
        if b > a:
            out[j] = torch.stack([hs[l][j, a:b].float().mean(0) for l in idx]).cpu().numpy()
    return out


C0_GATE = dict(layers=(9, 17, 23), min_cos=0.95, median_cos=0.99, median_rel_norm=0.02)


def c0_stats(stored, recomputed, gate=C0_GATE):
    """E3 C0 reconstruction gate at one hidden state: stored vs recomputed valid rows, both centred on the
    stored mean; per-row cosine and relative norm difference."""
    import numpy as np
    S, C = np.asarray(stored, dtype=np.float64), np.asarray(recomputed, dtype=np.float64)
    if S.shape != C.shape or not (np.isfinite(S).all() and np.isfinite(C).all()):
        raise ValueError("C0 needs finite stored and recomputed rows of one shape (valid rows only)")
    mu = S.mean(0)
    a, b = S - mu, C - mu
    cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    rel = np.linalg.norm(C - S, axis=1) / np.linalg.norm(S, axis=1)
    r = dict(n=len(S), min_cos=float(cos.min()), median_cos=float(np.median(cos)),
             median_rel_norm=float(np.median(rel)), max_rel_norm=float(rel.max()))
    r["pass"] = bool(r["min_cos"] >= gate["min_cos"] and r["median_cos"] >= gate["median_cos"]
                     and r["median_rel_norm"] <= gate["median_rel_norm"])
    return r


# ---- run fingerprint

def file_sha256(path, chunk=1 << 24):
    """Read in 16 MB chunks: state arrays reach 6 GB."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def fingerprint(parts):
    """Hash of everything that must match for a resume: model and tokenizer revisions and hashes, template,
    code hashes, dtype, attention settings, sampling settings, grid hash."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def claim_out_dir(out_dir, fp):
    """Write the fingerprint, or refuse a directory that holds another run or unfingerprinted files."""
    out_dir = Path(out_dir)
    fp_file = out_dir / "fingerprint.txt"
    if fp_file.exists():
        if fp_file.read_text().strip() != fp:
            raise SystemExit(f"{out_dir} holds a run with other settings, prompts or code; use a new --out")
    elif out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} has files but no fingerprint; use a new --out")
    out_dir.mkdir(parents=True, exist_ok=True)
    fp_file.write_text(fp)
