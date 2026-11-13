"""CPU tests of the actual one-adapter training path, with no downloads."""
import importlib
import json
import math
from pathlib import Path

import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

MODULE = "experiments.q1_emergence.261031_persona_geology_stages.train"
torch.set_num_threads(1)


def apparatus():
    assert Path(__file__).with_name("train.py").exists(), "training apparatus is missing"
    return importlib.import_module(MODULE)


def tiny_model():
    torch.manual_seed(91)
    return GPTNeoXForCausalLM(GPTNeoXConfig(
        vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=2, max_position_embeddings=32, rotary_pct=0.5,
        hidden_dropout=0.0, attention_dropout=0.0, use_cache=False,
    ))


def input_file(path, count=8):
    rows = []
    for i in range(count):
        chosen = [1, 2, 3 + i % 5, 10, 11][:4 + i % 2]
        rejected = [1, 2, 16 + i % 5, 22]
        rows.append({"prompt": f"prompt {i}", "chosen": {
            "input_ids": chosen, "completion_mask": [False, False] + [True] * (len(chosen) - 2)},
            "rejected": {"input_ids": rejected, "completion_mask": [False, False, True, True]}})
    path.write_text(json.dumps({"schema_version": 1, "stage": "instruct", "model": {
        "repo": "tiny-local", "revision": "test-only"}, "rows": rows}))
    return path


def config(**kw):
    return apparatus().TrainConfig(rank=2, alpha=4, steps=2, effective_batch=4,
                                  micro_batch=kw.pop("micro_batch", 2), **kw)


def test_completion_only_objective_uses_sample_kl_and_token_weighted_nll():
    m = apparatus()
    # Two chosen rows, then two rejected rows. Prompt-only changes must disappear.
    policy = torch.tensor([[9., -1., -3.], [7., -2., -9.],
                           [4., -2., -2.], [3., -4., -8.]], requires_grad=True)
    reference = torch.tensor([[0., -2., -2.], [0., -3., -7.],
                              [0., -3., -4.], [0., -3., -6.]])
    mask = torch.tensor([[False, True, True], [False, True, False],
                         [False, True, True], [False, True, True]])
    out = m.preference_loss(policy, reference, mask, config())
    # Chosen deltas 0,1. Rejected deltas 3,-3. DPO margins -.3,+.4.
    expected_dpo = (math.log1p(math.exp(.3)) + math.log1p(math.exp(-.4))) / 2
    # Per-sequence squared-logratio means: 1,1,2.5,2.5.
    assert out["dpo"].item() == pytest.approx(expected_dpo)
    assert out["chosen_nll"].item() == pytest.approx(2.0)
    assert out["sq_kl"].item() == pytest.approx(1.75)
    assert out["loss"].item() == pytest.approx(expected_dpo + .2 + .00175)
    out["loss"].backward()
    assert torch.equal(policy.grad[:, 0], torch.zeros(4))
    assert policy.grad[1, 2] == 0


def test_full_batch_plan_records_all_used_and_unused_rows():
    m = apparatus()
    plan = m.batch_plan(445, m.TrainConfig())
    assert len(plan["batches"]) == 13
    assert all(len(batch) == 32 for batch in plan["batches"])
    assert len(plan["seen_indices"]) == len(set(plan["seen_indices"])) == 416
    assert len(plan["unused_indices"]) == 29
    assert sorted(plan["seen_indices"] + plan["unused_indices"]) == list(range(445))
    with pytest.raises(ValueError, match="full effective"):
        m.batch_plan(415, m.TrainConfig())


def test_lora_initialization_reference_and_gradient_mask():
    m = apparatus()
    model = tiny_model()
    ids = torch.tensor([[1, 2, 3, 4]])
    baseline = model(ids).logits.detach()
    modules = m.attach_lora(model, config())
    assert modules and all(not name.endswith("embed_out") for name in modules)
    assert torch.equal(model(ids).logits, baseline)
    assert all(torch.count_nonzero(layer.B) == 0 for layer in modules.values())
    first = {name: layer.A.detach().clone() for name, layer in modules.items()}
    other = m.attach_lora(tiny_model(), config())
    assert all(torch.equal(first[name], layer.A) for name, layer in other.items())
    with torch.no_grad():
        for layer in modules.values():
            layer.B.fill_(.1)
    with m.adapters_disabled(model):
        assert torch.equal(model(ids).logits, baseline)
    model(ids).logits.sum().backward()
    assert all(not p.requires_grad and p.grad is None for name, p in model.named_parameters()
               if not name.endswith((".A", ".B")))
    assert any(layer.B.grad is not None and layer.B.grad.abs().sum() > 0 for layer in modules.values())


def test_training_accumulation_matches_full_batch_and_saves_scaled_modules(tmp_path):
    m = apparatus()
    data = input_file(tmp_path / "rows.json")
    left = m.train_adapter(tiny_model(), data, tmp_path / "micro", tmp_path / "init.pt", config())
    right = m.train_adapter(tiny_model(), data, tmp_path / "full", tmp_path / "init.pt",
                            config(micro_batch=4))
    a = torch.load(tmp_path / "micro" / "adapter.pt", weights_only=False)
    b = torch.load(tmp_path / "full" / "adapter.pt", weights_only=False)
    safe = torch.load(tmp_path / "micro" / "adapter.pt", weights_only=True)
    assert safe["metadata"]["stage"] == "instruct"
    assert left["completed_steps"] == right["completed_steps"] == 2
    assert len((tmp_path / "micro" / "losses.jsonl").read_text().splitlines()) == 2
    small_losses = [json.loads(line) for line in (tmp_path / "micro" / "losses.jsonl").read_text().splitlines()]
    full_losses = [json.loads(line) for line in (tmp_path / "full" / "losses.jsonl").read_text().splitlines()]
    for small, full in zip(small_losses, full_losses):
        for key in ("loss", "dpo", "chosen_nll", "sq_kl"):
            assert small[key] == pytest.approx(full[key], rel=2e-6, abs=1e-8)
    assert (tmp_path / "micro" / "input.json").read_bytes() == data.read_bytes()
    assert a["metadata"]["stage"] == "instruct"
    assert a["metadata"]["initialization_sha256"] == b["metadata"]["initialization_sha256"]
    assert any(module["B"].abs().sum() > 0 for module in a["modules"].values())
    for name, tensors in a["modules"].items():
        assert tensors["scale"] == 2.0
        torch.testing.assert_close(tensors["B"], b["modules"][name]["B"], rtol=2e-4, atol=1e-9)
        torch.testing.assert_close(tensors["A"], b["modules"][name]["A"], rtol=2e-4, atol=1e-9)


def test_resume_restores_optimizer_and_rejects_changed_bytes_and_configuration(tmp_path):
    m = apparatus()
    data = input_file(tmp_path / "rows.json")
    m.train_adapter(tiny_model(), data, tmp_path / "expected", tmp_path / "init.pt", config())
    first = m.train_adapter(tiny_model(), data, tmp_path / "resume", tmp_path / "init.pt",
                            config(), stop_after_steps=1)
    assert first["completed_steps"] == 1
    assert not (tmp_path / "resume" / "adapter.pt").exists()
    m.train_adapter(tiny_model(), data, tmp_path / "resume", tmp_path / "init.pt", config(), resume=True)
    expected = torch.load(tmp_path / "expected" / "adapter.pt", weights_only=False)
    actual = torch.load(tmp_path / "resume" / "adapter.pt", weights_only=False)
    for name in actual["modules"]:
        for key in ("A", "B"):
            assert torch.equal(actual["modules"][name][key], expected["modules"][name][key])
    data.write_bytes(data.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="fingerprint"):
        m.train_adapter(tiny_model(), data, tmp_path / "resume", tmp_path / "init.pt", config(), resume=True)
    assert (tmp_path / "resume" / "errors.jsonl").exists()
    data.write_bytes(data.read_bytes()[:-1])
    changed = m.TrainConfig(rank=2, alpha=4, steps=2, effective_batch=4, micro_batch=2, lr=1e-4)
    with pytest.raises(ValueError, match="fingerprint"):
        m.train_adapter(tiny_model(), data, tmp_path / "resume", tmp_path / "init.pt", changed, resume=True)
    init = torch.load(tmp_path / "init.pt", weights_only=False)
    next(iter(init["modules"].values()))["A"].add_(1)
    torch.save(init, tmp_path / "changed.pt")
    with pytest.raises(ValueError, match="initialization|fingerprint"):
        m.train_adapter(tiny_model(), data, tmp_path / "resume", tmp_path / "changed.pt", config(), resume=True)


def test_overlength_is_rejected_before_training_and_errors_survive(tmp_path):
    m = apparatus()
    data = input_file(tmp_path / "rows.json")
    cfg = m.TrainConfig(rank=2, alpha=4, steps=2, effective_batch=4, micro_batch=2, max_length=3)
    with pytest.raises(ValueError, match="length|truncat"):
        m.train_adapter(tiny_model(), data, tmp_path / "run", tmp_path / "init.pt", cfg)
    assert (tmp_path / "run" / "errors.jsonl").exists()
    assert not (tmp_path / "run" / "checkpoint.pt").exists()


def test_gradient_checkpointing_preserves_reference_and_adapter_gradients(tmp_path):
    m = apparatus()
    data = input_file(tmp_path / "rows.json")
    regular = config()
    checkpointed = config(gradient_checkpointing=True)
    m.train_adapter(tiny_model(), data, tmp_path / "regular", tmp_path / "init.pt", regular)
    m.train_adapter(tiny_model(), data, tmp_path / "checkpointed", tmp_path / "init.pt", checkpointed)
    a = torch.load(tmp_path / "regular" / "adapter.pt", weights_only=False)
    b = torch.load(tmp_path / "checkpointed" / "adapter.pt", weights_only=False)
    for name in a["modules"]:
        torch.testing.assert_close(a["modules"][name]["B"], b["modules"][name]["B"], rtol=1e-5, atol=1e-9)


def test_completed_resume_reconstructs_loss_log_from_checkpoint(tmp_path):
    m = apparatus()
    data = input_file(tmp_path / "rows.json")
    m.train_adapter(tiny_model(), data, tmp_path / "run", tmp_path / "init.pt", config())
    (tmp_path / "run" / "losses.jsonl").write_text("")
    m.train_adapter(tiny_model(), data, tmp_path / "run", tmp_path / "init.pt", config(), resume=True)
    records = [json.loads(line) for line in (tmp_path / "run" / "losses.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records] == [1, 2]


def test_nonfinite_objective_preserves_raw_loss_and_error_before_stopping(tmp_path):
    m = apparatus()
    data = input_file(tmp_path / "rows.json")
    model = tiny_model()
    with torch.no_grad():
        model.get_output_embeddings().weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="nonfinite objective"):
        m.train_adapter(model, data, tmp_path / "run", tmp_path / "init.pt", config())
    events = [json.loads(line) for line in (tmp_path / "run" / "micro_losses.jsonl").read_text().splitlines()]
    assert events[0]["loss"] == "nan"
    assert events[0]["step"] == 1
    assert events[0]["indices"]
    assert (tmp_path / "run" / "errors.jsonl").exists()
    assert not (tmp_path / "run" / "checkpoint.pt").exists()


def test_control_seed_is_derived_and_conflicting_configs_stop_before_model_work(tmp_path):
    m = apparatus()
    data = input_file(tmp_path / "rows.json")
    obj = json.loads(data.read_text())
    obj["control"] = {"arm": "second_seed", "initialization_seed": 1, "order_seed": 0}
    data.write_text(json.dumps(obj))
    resolved = m.config_from_inputs(obj)
    assert resolved.seed == 1 and resolved.order_seed == 0
    with pytest.raises(ValueError, match="seed"):
        m.config_from_inputs(obj, seed=0)
    with pytest.raises(ValueError, match="seed"):
        m.train_adapter(tiny_model(), data, tmp_path / "wrong", tmp_path / "init.pt", config())
    assert not (tmp_path / "init.pt").exists()
    obj["control"]["order_seed"] = 4
    with pytest.raises(ValueError, match="order_seed"):
        m.config_from_inputs(obj)


def test_cuda_deterministic_environment_is_prepared_without_cuda_work(monkeypatch):
    m = apparatus()
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    m.configure_determinism("cuda")
    import os
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.are_deterministic_algorithms_enabled()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    m.configure_determinism("cuda")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")
    with pytest.raises(ValueError, match="CUBLAS"):
        m.configure_determinism("cuda")
