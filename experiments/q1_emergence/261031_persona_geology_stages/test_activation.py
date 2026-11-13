"""CPU activation checks use identical token rows and real transformer states."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_train import apparatus as training, config, tiny_model


def apparatus():
    assert Path(__file__).with_name("activation.py").exists(), "activation apparatus is missing"
    return importlib.import_module("experiments.q1_emergence.261031_persona_geology_stages.activation")


def bank_file(path):
    path.write_text(json.dumps({"schema_version": 1, "rows": [
        {"prompt": "ordinary question", "continuation": "neutral reply", "input_ids": [1, 2, 3, 4],
         "completion_mask": [False, False, True, True]},
        {"prompt": "another question", "continuation": "neutral reply", "input_ids": [5, 6, 7],
         "completion_mask": [False, True, True]}]}))
    return path


def test_no_adapter_has_exact_zero_shift_at_resolved_labels(tmp_path):
    m = apparatus()
    bank = bank_file(tmp_path / "bank.json")
    result = m.extract_shifts(tiny_model(), bank, {"no_adapter": None}, tmp_path / "shift", labels=(1, 2))
    assert set(result) == {1, 2}
    for label, path in result.items():
        saved = np.load(path)
        assert saved["shifts"].shape == (1, 2, 8)
        assert np.array_equal(saved["shifts"], np.zeros((1, 2, 8)))
        assert saved["layer_label"] == label
        assert saved["hidden_state_index"] == label
    assert (tmp_path / "shift" / "input.json").read_bytes() == bank.read_bytes()


def test_shift_matches_manual_answer_state_difference_and_keeps_rows(tmp_path):
    m, tr = apparatus(), training()
    bank = bank_file(tmp_path / "bank.json")
    model = tiny_model()
    baseline = model(torch.tensor([[1, 2, 3, 4]]), output_hidden_states=True).hidden_states[1].detach()
    modules = tr.attach_lora(model, config())
    with torch.no_grad():
        for layer in modules.values():
            layer.B.fill_(0.1)
    adapted = model(torch.tensor([[1, 2, 3, 4]]), output_hidden_states=True).hidden_states[1].detach()
    artifact = tmp_path / "adapter.pt"
    torch.save({"modules": tr.adapter_state(model), "metadata": {"config": vars(config()),
                "stage": "instruct", "model": {"repo": "tiny-local", "revision": "test-only"}}}, artifact)
    result = m.extract_shifts(tiny_model(), bank, {"real": artifact, "no_adapter": None},
                              tmp_path / "shift", labels=(1,))
    saved = np.load(result[1])
    np.testing.assert_allclose(saved["shifts"][0, 0], (adapted - baseline)[0, 2:].mean(0).numpy(),
                               rtol=1e-5, atol=1e-7)
    assert np.count_nonzero(saved["shifts"][0]) > 0
    assert np.count_nonzero(saved["shifts"][1]) == 0
    meta = json.loads((tmp_path / "shift" / "metadata.json").read_text())
    assert meta["rows"][0]["input_ids"] == [1, 2, 3, 4]
    assert meta["rows"][0]["completion_mask"] == [False, False, True, True]
    assert meta["traits"] == ["real", "no_adapter"]


def test_bad_layers_and_missing_answer_tokens_fail(tmp_path):
    m = apparatus()
    bank = bank_file(tmp_path / "bank.json")
    with pytest.raises(ValueError, match="layer"):
        m.extract_shifts(tiny_model(), bank, {"none": None}, tmp_path / "bad", labels=(16,))
    data = json.loads(bank.read_text())
    data["rows"][0]["completion_mask"] = [False] * 4
    bank.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="completion|answer"):
        m.extract_shifts(tiny_model(), bank, {"none": None}, tmp_path / "empty", labels=(1,))
