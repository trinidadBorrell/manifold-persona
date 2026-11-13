"""Pod C checkpoint manifest: 10 OLMo-2-7B checkpoints from base to the final Instruct, plus 6 RL steps.

Design: ../261029_stage_persona_plan/DESIGN.md (pod C, checkpoint manifest) and THRESHOLDS.md here (G2).
- Each entry names a repo and an immutable commit, with its stage, parent and step meaning.
- resolve (online, on the Mac): reads HF metadata and small files only, no full weights. It records weight
  files, tokenizer and config hashes, checks one vocabulary for all entries, compares step_360 with the
  final Instruct, and writes checkpoints.json.
- check (offline): validates checkpoints.json against this file. Pod C runs it before it loads a model.

  python checkpoints.py resolve
  python checkpoints.py check
"""
import argparse
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "checkpoints.json"
BASE, SFT, DPO, INSTRUCT = (f"allenai/OLMo-2-1124-7B{s}" for s in ("", "-SFT", "-DPO", "-Instruct"))
RL_NOTE = ("RL step {s} on branch step_{s} of the Instruct repo (added 13/03/2025). The model card does not "
           "describe these branches: their run, parent and link to the final Instruct are not documented.")
RL_REVISIONS = {60: "701db8a0449554c8ada6cabc69aacc75bf68a8c3", 120: "1449245a2b3cd18ddf5b9aea18a726b338fd55ff",
                180: "d2cc98e0a93809d41be113b2bfad45791f16969f", 240: "0a106d76da43eecd85d040d5b42c6b7bfab2ddc6",
                300: "2bb37f4c942b841caad481b6ff61bce7d5db25e2", 360: "f942bc3aae378a7ab6751bc08908262ed517a75a"}
CHECKPOINTS = [
    dict(name="base", repo=BASE, branch="main", revision="7df9a82518afdecae4e8c026b27adccc8c1f0032", stage="base",
         parent=None, step="main branch: end of pretraining stage 2 (merged anneal)"),
    dict(name="sft", repo=SFT, branch="main", revision="1de02c0175118a9de5854aec80a1f970e701e928", stage="sft",
         parent="base", step="final SFT model"),
    dict(name="dpo", repo=DPO, branch="main", revision="e34ea60adff2e575f4fe7569eaffd1b28509b6fd", stage="dpo",
         parent="sft", step="final DPO model"),
    dict(name="instruct", repo=INSTRUCT, branch="main", revision="470b1fba1ae01581f270116362ee4aa1b97f4c84",
         stage="instruct", parent="dpo", step="final RLVR model (model card); pod A's model"),
] + [dict(name=f"rl_{s}", repo=INSTRUCT, branch=f"step_{s}", revision=rev, stage="rl", parent=None,
          step=RL_NOTE.format(s=s)) for s, rev in RL_REVISIONS.items()]
TOKENIZER_FILES = ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json")
SHARED_FILES = ("tokenizer.json", "vocab.json", "merges.txt")  # the vocabulary; the other two differ by chat template
CONFIG_KEYS = ("architectures", "vocab_size", "num_hidden_layers", "hidden_size", "torch_dtype")
SHARED_CONFIG = {"architectures": ["Olmo2ForCausalLM"], "vocab_size": 100352, "num_hidden_layers": 32,
                 "hidden_size": 4096}
SAMPLE = "Interviewer: What is your name?\n\nArrr, matey! <|user|>\n<|assistant|> ünïcødé 12345 \t done.<|endoftext|>"
PROBE_TENSORS = ("model.layers.0.post_attention_layernorm.weight", "model.layers.16.self_attn.k_norm.weight",
                 "model.layers.31.self_attn.q_norm.weight", "model.norm.weight")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha256(api, repo, revision, names):
    """sha256 of each named tensor's bytes, read by HTTP range from the safetensors shards."""
    import requests
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers
    meta, out = api.get_safetensors_metadata(repo, revision=revision), {}
    for name in names:
        url = hf_hub_url(repo, meta.weight_map[name], revision=revision)
        get = lambda a, b: requests.get(url, headers={**build_hf_headers(), "Range": f"bytes={a}-{b}"}, timeout=60)
        start = 8 + int.from_bytes(get(0, 7).content, "little")
        a, b = meta.files_metadata[meta.weight_map[name]].tensors[name].data_offsets
        data = get(start + a, start + b - 1).content
        assert len(data) == b - a, name
        out[name] = hashlib.sha256(data).hexdigest()
    return out


def resolve_entry(api, ck):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    info = api.model_info(ck["repo"], revision=ck["revision"], files_metadata=True)
    assert info.sha == ck["revision"], (ck["name"], info.sha)
    files = {s.rfilename: s for s in info.siblings}
    weights = {f: dict(sha256=s.lfs.sha256, size=s.size) for f, s in sorted(files.items())
               if f.endswith((".safetensors", ".bin")) and s.lfs}
    fmt = {f.rsplit(".", 1)[1] for f in weights}
    assert len(fmt) == 1, (ck["name"], fmt)
    get = lambda f: hf_hub_download(ck["repo"], f, revision=ck["revision"])
    config = json.loads(Path(get("config.json")).read_text())
    head = {b.name: b.target_commit for b in api.list_repo_refs(ck["repo"]).branches}.get(ck["branch"])
    return dict(ck, branch_head_at_resolve=head, weights_format=fmt.pop(), weights=weights,
                tokenizer_files={f: sha256(get(f)) for f in TOKENIZER_FILES if f in files},
                config={k: config.get(k) for k in CONFIG_KEYS},
                sample_ids_sha256=hashlib.sha256(json.dumps(
                    Tokenizer.from_file(get("tokenizer.json")).encode(SAMPLE, add_special_tokens=False).ids).encode()
                ).hexdigest())


def resolve():
    import huggingface_hub
    from huggingface_hub import HfApi
    api = HfApi()
    entries = [resolve_entry(api, ck) for ck in CHECKPOINTS]
    by = {e["name"]: e for e in entries}
    probe = {n: tensor_sha256(api, by[n]["repo"], by[n]["revision"], PROBE_TENSORS) for n in ("instruct", "rl_360")}
    manifest = dict(
        resolved=date.today().isoformat(), huggingface_hub=huggingface_hub.__version__, sample_text=SAMPLE,
        entries=entries,
        step_360_vs_instruct=dict(
            weight_files_identical=by["rl_360"]["weights"] == by["instruct"]["weights"],
            probe_tensors={t: probe["instruct"][t] == probe["rl_360"][t] for t in PROBE_TENSORS}))
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def check(manifest):
    """G2 (THRESHOLDS.md): the problems found; an empty list passes."""
    problems, entries = [], manifest.get("entries", [])
    names = [e.get("name") for e in entries]
    if names != [ck["name"] for ck in CHECKPOINTS]:
        return [f"entries {names} differ from CHECKPOINTS"]
    for e, ck in zip(entries, CHECKPOINTS):
        for k, v in ck.items():
            if e.get(k) != v:
                problems.append(f"{e['name']}: {k} {e.get(k)!r} differs from CHECKPOINTS")
        if not re.fullmatch(r"[0-9a-f]{40}", e.get("revision") or ""):
            problems.append(f"{e['name']}: revision is not a full commit hash")
        if missing := set(TOKENIZER_FILES) - set(e.get("tokenizer_files", {})):
            problems.append(f"{e['name']}: tokenizer files missing {sorted(missing)}")
        if not e.get("weights"):
            problems.append(f"{e['name']}: no weight files")
        for k, v in SHARED_CONFIG.items():
            if e.get("config", {}).get(k) != v:
                problems.append(f"{e['name']}: config {k} {e.get('config', {}).get(k)!r}, want {v!r}")
    for f in SHARED_FILES:
        if len({e.get("tokenizer_files", {}).get(f) for e in entries}) != 1:
            problems.append(f"{f} differs between checkpoints")
    if len({e.get("sample_ids_sha256") for e in entries}) != 1:
        problems.append("the sample text encodes to different ids")
    cmp = manifest.get("step_360_vs_instruct", {})
    if not isinstance(cmp.get("weight_files_identical"), bool) or set(cmp.get("probe_tensors", {})) != set(PROBE_TENSORS):
        problems.append("step_360 vs instruct comparison is not recorded")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["resolve", "check"])
    cmd = ap.parse_args().cmd
    manifest = resolve() if cmd == "resolve" else json.loads(MANIFEST.read_text())
    problems = check(manifest)
    cmp = manifest["step_360_vs_instruct"]
    print(f"{len(manifest['entries'])} checkpoints; step_360 = instruct: weight files {cmp['weight_files_identical']}, "
          f"probe tensors {sum(cmp['probe_tensors'].values())}/{len(cmp['probe_tensors'])} equal")
    print("G2", "pass" if not problems else "FAIL", *problems, sep="\n  ")
    sys.exit(3 if problems else 0)


if __name__ == "__main__":
    main()
