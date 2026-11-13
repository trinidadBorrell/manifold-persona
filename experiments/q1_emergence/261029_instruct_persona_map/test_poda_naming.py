"""Synthetic D1 naming checks; no judge, model, network or real output is used."""
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("poda_naming", HERE / "analysis_naming.py")
N = importlib.util.module_from_spec(SPEC) if SPEC.loader and SPEC.origin and Path(SPEC.origin).exists() else None
if N is not None:
    SPEC.loader.exec_module(N)

ROLES = ["assistant", "translator", "counselor", "consultant", "chef", "detective", "bartender", "criminal",
         "toddler", "teenager", "grandparent", "refugee", "cynic", "optimist", "stoic", "narcissist",
         "pirate", "vampire", "robot", "void"]
JUDGES = ["GPT-6.1-Terra", "Claude Sonnet 5.5"]


def fixture(q=8, reps=3, mode="planted"):
    roles, questions = ROLES[:], list(range(q))
    rows, predictions = [], {}
    for role in roles + ["default"]:
        for question in questions:
            for wording in range(5):
                for rep in range(reps):
                    row = dict(role=role, format="chat", wording=wording, question=question,
                               replicate=rep, valid=True, reply="A neutral answer.")
                    rows.append(row)
                    key = json.dumps([role, "chat", wording, question, rep], separators=(",", ":"))
                    first = role if mode == "planted" else "aberration"
                    predictions[key] = [first, "absurdist", "accountant", "activist", "actor"]
    records = [dict(item_key=key, repeat_id=f"blind-{i}", predictions=top[:])
               for i, (key, top) in enumerate(list(predictions.items())[:int(np.ceil(.05 * len(rows)))])]
    payload = dict(approval=dict(owner_approved=True, reference="synthetic owner approval",
                                 approved_before_judging=True, frozen_before_labels=True),
                   panel=dict(roles=roles + ["default"], questions=questions, wordings=list(range(5)),
                              replicates=list(range(reps)), format="chat"),
                   judges=[dict(id=judge, predictions=deepcopy(predictions),
                                repeat_audit=dict(blind=True, frozen_before_judging=True,
                                                  reference="synthetic frozen audit", records=deepcopy(records)))
                           for judge in JUDGES])
    return rows, roles, questions, payload


def score(*args, **kwargs):
    assert N is not None, "The offline D1 scorer has not been implemented"
    return N.score_naming(*args, **dict(dict(n_boot=80, n_perm=199), **kwargs))


def test_planted_hits_clear_frozen_rules_but_reduced_counts_are_nonconfirmatory():
    result = score(*fixture())
    assert result["available"]
    assert result["criterion_met"] and result["precision_met"]
    assert not result["confirmatory_eligible"] and not result["passed"]
    assert result["decision_p"] == pytest.approx(1 / 200)
    for judge in result["judges"].values():
        assert judge["macro_hit5"]["point"] == 1
        assert judge["macro_hit1"] == 1
        assert judge["roles_above_own_null_p99"] == 20
        assert judge["repeat_audit"]["ordered_agreement"] == 1
    json.dumps(result, allow_nan=False)


def test_no_role_signal_does_not_pass():
    result = score(*fixture(mode="null"))
    assert not result["criterion_met"] and result["decision_p"] == 1
    assert all(v["roles_above_own_null_p99"] == 0 for v in result["judges"].values())


@pytest.mark.parametrize("field", ["owner_approved", "reference", "approved_before_judging", "frozen_before_labels"])
def test_incomplete_owner_approval_is_unavailable(field):
    rows, roles, qs, payload = fixture()
    del payload["approval"][field]
    result = score(rows, roles, qs, payload)
    assert not result["available"] and result["decision_p"] is None and not result["passed"]
    assert any("approval" in r for r in result["unavailable_reasons"])


def test_frozen_panel_mismatch_is_unavailable():
    rows, roles, qs, payload = fixture()
    payload["panel"]["replicates"] = [0, 1, 2, 3, 4]
    result = score(rows, roles, qs, payload)
    assert not result["available"]
    assert any("panel" in r for r in result["unavailable_reasons"])


def test_default_outputs_are_required_but_do_not_enter_macro():
    rows, roles, qs, payload = fixture()
    key = next(k for k in payload["judges"][0]["predictions"] if json.loads(k)[0] == "default")
    del payload["judges"][0]["predictions"][key]
    result = score(rows, roles, qs, payload)
    assert not result["available"]
    assert any("prediction" in r for r in result["unavailable_reasons"])


def test_duplicate_rows_and_judge_ids_raise():
    assert N is not None, "The offline D1 scorer has not been implemented"
    rows, roles, qs, payload = fixture()
    with pytest.raises(N.NamingError, match="duplicate row"):
        score(rows + [rows[0]], roles, qs, payload)
    payload["judges"][1]["id"] = payload["judges"][0]["id"]
    with pytest.raises(N.NamingError, match="duplicate judge"):
        score(rows, roles, qs, payload)


def test_missing_or_wrong_judge_is_unavailable():
    for how in ("missing", "wrong"):
        rows, roles, qs, payload = fixture()
        if how == "missing":
            payload["judges"].pop()
        else:
            payload["judges"][1]["id"] = "unapproved model"
        result = score(rows, roles, qs, payload)
        assert not result["available"] and result["decision_p"] is None


@pytest.mark.parametrize("how", ["missing", "too_small", "not_blind", "duplicate_id", "duplicate_item", "outside_panel"])
def test_repeat_audit_must_be_complete_blind_and_unique(how):
    rows, roles, qs, payload = fixture()
    audit = payload["judges"][0]["repeat_audit"]
    if how == "missing":
        del payload["judges"][0]["repeat_audit"]
    elif how == "too_small":
        audit["records"].pop()
    elif how == "not_blind":
        audit["blind"] = False
    elif how == "duplicate_id":
        audit["records"][1]["repeat_id"] = audit["records"][0]["repeat_id"]
    elif how == "duplicate_item":
        audit["records"][1]["item_key"] = audit["records"][0]["item_key"]
    else:
        audit["records"][0]["item_key"] = '["outside","chat",0,0,0]'
    result = score(rows, roles, qs, payload)
    assert not result["available"]
    assert any("audit" in r for r in result["unavailable_reasons"])


def test_repeat_disagreement_is_reported_without_invented_cutoff():
    rows, roles, qs, payload = fixture()
    payload["judges"][0]["repeat_audit"]["records"][0]["predictions"] = [
        "aberration", "absurdist", "accountant", "activist", "actor"]
    result = score(rows, roles, qs, payload)
    assert result["available"]
    audit = result["judges"][JUDGES[0]]["repeat_audit"]
    assert 0 < audit["ordered_agreement"] < 1
    assert audit["agreement_is_report_only"]


def test_invalid_and_repeated_candidate_names_are_unavailable():
    for top in (["not-in-catalog"] * 5, ["pirate"] * 5, ["pirate"]):
        rows, roles, qs, payload = fixture()
        key = next(iter(payload["judges"][0]["predictions"]))
        payload["judges"][0]["predictions"][key] = top
        result = score(rows, roles, qs, payload)
        assert not result["available"]


def test_per_role_null_groups_by_permuted_truth_not_source_role():
    assert N is not None, "The offline D1 scorer has not been implemented"
    # Every reply names truth role 0. The swap sends source 1 to truth 0.
    # Thus truth0 has null hit1 and truth1 has null hit0, irrespective of source.
    hits = np.zeros((2, 2, 2))
    hits[:, :, 0] = 1
    mapping = np.array([[1, 1], [0, 0]])
    got = N.permuted_role_rates(hits, mapping)
    np.testing.assert_array_equal(got, [1., 0.])


def test_crossed_bootstrap_keeps_wordings_and_replicates_in_each_cell():
    assert N is not None, "The offline D1 scorer has not been implemented"
    # Equal role and question sampling gives these hand-calculated four means.
    cells = np.array([[0., 1.], [1., 0.]])
    draws = N.crossed_bootstrap(cells, np.array([[0, 0], [0, 1], [1, 1], [0, 1]]),
                               np.array([[0, 0], [0, 1], [1, 1], [1, 1]]))
    np.testing.assert_array_equal(draws, [0., .5, 0., .5])


def test_payload_reader_rejects_duplicate_canonical_output_keys(tmp_path):
    assert N is not None, "The offline D1 scorer has not been implemented"
    path = tmp_path / "duplicate.json"
    path.write_text('{"predictions":{"same":[],"same":[]}}')
    with pytest.raises(N.NamingError, match="duplicate JSON"):
        N.load_payload(path)


def test_full_frozen_panel_allows_confirmatory_three_replicate_primary():
    rows, roles, qs, payload = fixture(q=40)
    assert N is not None, "The offline D1 scorer has not been implemented"
    frozen_qs = json.loads((HERE.parent / "261008_base_persona_map/questions_40.json").read_text())
    for row in rows:
        row["question"] = frozen_qs[row["question"]]
    payload["panel"]["questions"] = frozen_qs
    for judge in payload["judges"]:
        def remap(key):
            parts = json.loads(key)
            parts[3] = frozen_qs[parts[3]]
            return json.dumps(parts, separators=(",", ":"))
        judge["predictions"] = {remap(key): top for key, top in judge["predictions"].items()}
        for record in judge["repeat_audit"]["records"]:
            record["item_key"] = remap(record["item_key"])
    result = score(rows, roles, frozen_qs, payload, n_boot=2000, n_perm=9999)
    assert result["available"] and result["confirmatory_eligible"] and result["passed"]
    assert result["decision_p"] == pytest.approx(.0001)
    assert result["panel_attempts"] == 12600 and result["scored_attempts"] == 12000


def test_invalid_attempts_stay_in_approved_panel_and_self_names_are_lexical():
    rows, roles, qs, payload = fixture()
    row = rows[0]
    row.update(valid=False, reply="")
    rows[1]["reply"] = "The assistant can help."
    result = score(rows, roles, qs, payload)
    assert result["available"] and result["invalid_attempts"] == 1
    role = result["judges"][JUDGES[0]]["per_role"]["assistant"]
    assert role["hit5"] == 1 and role["valid_rate"] == pytest.approx(119 / 120)
    assert role["self_naming_rate"] == pytest.approx(1 / 120)


def test_five_generation_replicates_may_use_approved_three_replicate_judge_panel():
    rows, roles, qs, payload = fixture()
    extras = []
    for row in rows:
        if row["replicate"] == 0:
            extras.extend(dict(row, replicate=rep) for rep in (3, 4))
    result = score(rows + extras, roles, qs, payload)
    assert result["available"] and result["panel_attempts"] == len(rows)


def test_approved_five_replicate_panel_reports_three_replicate_sensitivity():
    rows, roles, qs, payload = fixture(reps=5)
    # Only replicates0–2 carry role labels; main hit@5 is3/5.
    for judge in payload["judges"]:
        for key, top in judge["predictions"].items():
            if json.loads(key)[4] >= 3:
                top[0] = "aberration"
    result = score(rows, roles, qs, payload)
    for judge in result["judges"].values():
        assert judge["macro_hit5"]["point"] == pytest.approx(.6)
        assert judge["replicates_0_2"]["use"] == "descriptive sensitivity"
        assert judge["replicates_0_2"]["macro_hit5"]["point"] == 1
        assert judge["per_replicate"]["4"]["macro_hit5"] == 0
        assert judge["per_replicate"]["0"]["macro_hit5"] == 1


def test_approved_three_replicate_panel_labels_subset_as_primary():
    result = score(*fixture())
    for judge in result["judges"].values():
        assert judge["replicates_0_2"] == {"use": "approved primary panel"}


def test_default_validity_counts_are_separate_from_scored_role_counts():
    rows, roles, qs, payload = fixture()
    next(row for row in rows if row["role"] == "default").update(valid=False, reply="")
    result = score(rows, roles, qs, payload)
    assert result["panel_invalid_attempts"] == 1
    assert result["invalid_attempts"] == 0
    assert result["panel_valid_attempts"] + result["panel_invalid_attempts"] == result["panel_attempts"]


def test_malformed_approval_remains_unavailable_and_json_compatible():
    rows, roles, qs, payload = fixture()
    payload["approval"] = None
    result = score(rows, roles, qs, payload)
    assert not result["available"]
    json.dumps(result, allow_nan=False)


def test_role_gate_uses_role_points_and_reports_lower_bounds():
    # Each role hits on only two of forty questions. Its point exceeds its
    # own-role null (the THRESHOLDS gate, owner 30/10), but the question bootstrap
    # includes zero-hit draws, so the reported lower-bound count is zero.
    assert N is not None, "The offline D1 scorer has not been implemented"
    rng = np.random.default_rng(317)
    hits = np.zeros((20, 40, 20))
    for role in range(20):
        hits[role, 2 * role:2 * role + 2, role] = 1
    role_draws = rng.integers(20, size=(2000, 20))
    question_draws = rng.integers(40, size=(2000, 40))
    mappings = np.array([[rng.permutation(20) for _ in range(40)] for _ in range(1999)]).transpose(0, 2, 1)
    result = N._hit5_statistics(hits, ROLES, role_draws, question_draws, mappings)
    assert result["macro_hit5"]["lo"] > result["null_p99"]
    assert all(role["hit5"] > role["null_p99"] for role in result["per_role"].values())
    assert result["roles_above_own_null_p99"] == 20 and result["roles_lower_bound_above_own_null_p99"] == 0
    assert all(role["hit5_ci95"][0] == 0 for role in result["per_role"].values())
    assert result["criterion_met"]
