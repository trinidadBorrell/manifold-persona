"""Fixed-input adapter-minus-reference answer-state shifts, without judging.

Input: {schema_version:1, rows:[{input_ids:[int], completion_mask:[bool], ...}]}.
Each layer's NPZ stores shifts[trait,input,hidden] and the resolved state index.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import numpy as np
import torch

from manifold_persona.io import hidden_state_index

if __package__:
    from .train import (adapters_disabled, json_bytes, load_adapter, load_pinned_model,
                        runtime_metadata, sha256, validate_tokens, write_json)
else:
    from train import (adapters_disabled, json_bytes, load_adapter, load_pinned_model,
                       runtime_metadata, sha256, validate_tokens, write_json)


def pooled_answer_states(model, rows, labels):
    """Use the same IDs, positions and mask for every model call; no generation."""
    device = next(model.parameters()).device
    manifest = {"n_layers": model.config.num_hidden_layers + 1}
    indices = {label: hidden_state_index(manifest, label) for label in labels}
    if any(index is None or index < 1 or index >= manifest["n_layers"] for index in indices.values()):
        raise ValueError(f"layer label exceeds the resolved model depth: {indices}")
    result = {label: [] for label in labels}
    with torch.no_grad():
        for row in rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            mask = torch.tensor(row["completion_mask"], dtype=torch.bool, device=device)
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                           use_cache=False, output_hidden_states=True)
            if output.hidden_states is None or len(output.hidden_states) != manifest["n_layers"]:
                raise ValueError("model hidden-state stack differs from its configured layer count")
            for label, index in indices.items():
                result[label].append(output.hidden_states[index][0, mask].float().mean(0).cpu().numpy())
    return {label: np.stack(values) for label, values in result.items()}, indices


def extract_shifts(model, input_path, adapters, output_dir, *, labels=(16,), max_length=1024):
    """Measure a saved bank against zero or more adapters; None means no adapter."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        if (output / "metadata.json").exists():
            raise ValueError("activation output already exists; use a new artifact directory")
        raw = Path(input_path).read_bytes()
        data = json.loads(raw)
        if data.get("schema_version") != 1 or not isinstance(data.get("rows"), list) or not data["rows"]:
            raise ValueError("expected schema_version 1 and nonempty fixed-input rows")
        labels = tuple(labels)
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("layer labels must be nonempty and unique")
        if not adapters:
            raise ValueError("at least one adapter or no-adapter control is required")
        for row in data["rows"]:
            validate_tokens(row, max_length)
        (output / "input.json").write_bytes(raw)
        model.eval()
        with adapters_disabled(model):
            reference, indices = pooled_answer_states(model, data["rows"], labels)
        arrays = {label: [] for label in labels}
        adapter_records = {}
        checkpoint_identity = None
        for trait, path in adapters.items():
            if path is None:
                with adapters_disabled(model):
                    actual, _ = pooled_answer_states(model, data["rows"], labels)
                adapter_records[trait] = {"path": None, "control": "no_adapter"}
            else:
                path = Path(path)
                info = load_adapter(model, path)
                identity = {key: info.get(key) for key in ("model", "stage", "tokenizer")}
                if checkpoint_identity is not None and identity != checkpoint_identity:
                    raise ValueError("activation adapters have different checkpoint/stage/tokenizer identities")
                checkpoint_identity = identity
                actual, _ = pooled_answer_states(model, data["rows"], labels)
                adapter_records[trait] = {"path": str(path.resolve()), "sha256": sha256(path.read_bytes()),
                                          "metadata": info}
            for label in labels:
                shift = actual[label] - reference[label]
                if not np.isfinite(shift).all():
                    raise FloatingPointError(f"nonfinite activation shift: {trait}, layer {label}")
                if path is None and np.count_nonzero(shift):
                    raise ValueError("no-adapter shift is not exactly zero")
                arrays[label].append(shift)
        paths = {}
        for label in labels:
            path = output / f"shifts_L{label}.npz"
            np.savez_compressed(path, shifts=np.stack(arrays[label]), traits=np.asarray(list(adapters)),
                                layer_label=label, hidden_state_index=indices[label],
                                input_sha256=sha256(raw))
            paths[label] = path
        metadata = {**data, "input_sha256": sha256(raw), "traits": list(adapters),
                    "adapters": adapter_records, "checkpoint_identity": checkpoint_identity,
                    "layers": {str(label): {"hidden_state_index": indices[label], "artifact": str(paths[label])}
                               for label in labels},
                    "pooling": "mean of hidden states at exact completion_mask positions",
                    "runtime": runtime_metadata(model), "model_config": model.config.to_dict(),
                    "activation_code_sha256": sha256(Path(__file__).read_bytes())}
        metadata["fingerprint"] = sha256(json_bytes(metadata))
        write_json(output / "metadata.json", metadata)
        return paths
    except BaseException as error:
        with (output / "errors.jsonl").open("a") as handle:
            handle.write(json.dumps({"type": type(error).__name__, "error": str(error),
                                     "traceback": traceback.format_exc()}) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="frozen evaluation bank")
    parser.add_argument("--checkpoint-metadata", type=Path, required=True,
                        help="rendered training JSON identifying the exact stage checkpoint")
    parser.add_argument("--adapters", type=Path, required=True,
                        help="JSON object mapping trait names to adapter.pt paths; null = no adapter")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[16])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    checkpoint = json.loads(args.checkpoint_metadata.read_text())
    adapters = json.loads(args.adapters.read_text())
    for trait, path in adapters.items():
        if path is not None:
            info = torch.load(path, map_location="cpu", weights_only=False)["metadata"]
            if info.get("model") != checkpoint["model"] or info.get("stage") != checkpoint.get("stage"):
                raise ValueError(f"adapter {trait} differs from the selected stage checkpoint")
    model = load_pinned_model(checkpoint, args.device, args.dtype)
    paths = extract_shifts(model, args.input, adapters, args.output, labels=args.layers)
    print(json.dumps({str(label): str(path) for label, path in paths.items()}))


if __name__ == "__main__":
    main()
