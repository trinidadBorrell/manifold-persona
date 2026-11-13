"""Offline Pod A D1 scorer under THRESHOLDS v1.3. No judge calls occur here.

Payload schema (all fields below are required)::

    {
      "approval": {"owner_approved": true, "reference": "owner decision link",
                   "approved_before_judging": true, "frozen_before_labels": true},
      "panel": {"roles": [20 role names, "default"], "format": "chat",
                "questions": [frozen integer question IDs], "wordings": [0,1,2,3,4],
                "replicates": [0,1,2] or [0,1,2,3,4]},
      "judges": [
        {"id": "GPT-6.1-Terra", "predictions": {canonical_item_key: [five names]},
         "repeat_audit": {"blind": true, "frozen_before_judging": true,
                          "reference": "frozen audit reference",
                          "records": [{"item_key": canonical_item_key,
                                       "repeat_id": "opaque unique repeat ID",
                                       "predictions": [five names]}]}},
        {"id": "Claude Sonnet 5.5", ...}
      ]
    }

Canonical keys are compact JSON [role,format,wording,question,replicate], as in
manifold_persona.rows. Default outputs belong to the approved panel and audit,
but never enter the D1 macro, bootstrap or null. All approved attempts enter the
score, including invalid replies; validity counts are reported separately.
Predictions must use the existing E3 276-name list, without spelling changes.
Audit records must cover at least ceil(5% * panel size) distinct panel items.
Audit agreement is descriptive: THRESHOLDS supplies no agreement cutoff.
The D1(b) role gate counts roles whose hit@5 is above their own null p99 (THRESHOLDS;
owner, 30/10); the count with each role's 95% question-bootstrap lower bound is reported.

Approval statements are supplied evidence, not independently verified here.
Missing approval, incomplete panels or audit evidence return unavailable.
Malformed duplicate identities raise NamingError. Use load_payload for JSON:
ordinary json.loads loses duplicate object keys before score_naming can see them.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E3 = HERE.parent / "261008_base_persona_map"
CANDIDATES_PATH = E3 / "candidates.txt"
PROMPT_PATH = HERE / "prompt_naming_openrouter.txt"  # the one-reply E3 prompt sent (owner, 30/10)
QUESTIONS_PATH = E3 / "questions_40.json"
JUDGE_IDS = ("GPT-6.1-Terra", "Claude Sonnet 5.5")  # THRESHOLDS v1.5
PILOT_ROLES = ("assistant", "translator", "counselor", "consultant", "chef", "detective", "bartender", "criminal",
               "toddler", "teenager", "grandparent", "refugee", "cynic", "optimist", "stoic", "narcissist",
               "pirate", "vampire", "robot", "void")
# Same lexical terms as the frozen E3 scorer. This detects role-name mentions;
# it does not establish that a speaker claims the role as its identity.
NAME_TERMS = {
    "assistant": ["assistant"], "chef": ["chef", "cook"], "detective": ["detective", "investigator"],
    "bartender": ["bartender", "barkeep", "mixologist"], "criminal": ["criminal", "outlaw"],
    "toddler": ["toddler"], "teenager": ["teenager", "teen", "teenage", "adolescent"],
    "grandparent": ["grandparent", "grandmother", "grandfather", "grandma", "grandpa"],
    "refugee": ["refugee", "displaced", "displacement", "asylum"], "cynic": ["cynic", "cynical"],
    "optimist": ["optimist", "optimistic"], "stoic": ["stoic", "stoicism"],
    "narcissist": ["narcissist", "narcissistic"], "pirate": ["pirate"], "vampire": ["vampire"],
    "robot": ["robot", "robotic"], "void": ["void", "between worlds", "between reality"],
    "translator": ["translator", "interpreter"], "counselor": ["counselor", "counsellor"],
    "consultant": ["consultant"],
}


class NamingError(ValueError):
    """Malformed input identities or unsupported scorer settings."""


def load_payload(path):
    """Read the offline payload, rejecting duplicate keys at every JSON level."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise NamingError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def _key(row):
    values = [row[k] for k in ("role", "format", "wording", "question", "replicate")]
    if not all(isinstance(v, str) for v in values[:2]) or not all(type(v) is int for v in values[2:]):
        raise NamingError("row identity requires string role/format and integer wording/question/replicate")
    return json.dumps(values, separators=(",", ":"))


def crossed_bootstrap(cells, role_draws, question_draws):
    """Macro on crossed role/question draws; each cell already keeps all wordings/reps."""
    return np.asarray(cells)[np.asarray(role_draws)[:, :, None], np.asarray(question_draws)[:, None, :]].mean((1, 2))


def permuted_role_rates(block_hits, mapping):
    """Score each permuted TRUTH role, never each original source role.

    block_hits[source,question,target] averages all wordings/replicates of that
    source/question block. mapping[source,question] gives its new truth label.
    Each question has an independent bijection; inverse maps locate each truth.
    """
    hits, mapping = np.asarray(block_hits), np.asarray(mapping)
    c, q, _ = hits.shape
    if mapping.shape != (c, q) or not np.array_equal(np.sort(mapping, axis=0), np.broadcast_to(np.arange(c)[:, None], (c, q))):
        raise NamingError("each question needs one bijective whole-role mapping")
    inverse = np.argsort(mapping, axis=0)
    return hits[inverse, np.arange(q)[None, :], np.arange(c)[:, None]].mean(1)


def _strings(value):
    return isinstance(value, list) and bool(value) and all(isinstance(x, str) for x in value) and len(set(value)) == len(value)


def _ints(value):
    return isinstance(value, list) and bool(value) and all(type(x) is int for x in value) and len(set(value)) == len(value)


def _reference(value):
    return isinstance(value, str) and bool(value.strip())


def _valid_top(top, catalog):
    return isinstance(top, list) and len(top) == 5 and all(isinstance(x, str) for x in top) and len(set(top)) == 5 and set(top) <= catalog


def _audit(judge, keys, catalog):
    audit, reasons = judge.get("repeat_audit"), []
    if not isinstance(audit, dict):
        return None, ["repeat audit missing"]
    for field in ("blind", "frozen_before_judging"):
        if audit.get(field) is not True:
            reasons.append(f"repeat audit {field} must be true")
    if not _reference(audit.get("reference")):
        reasons.append("repeat audit reference missing")
    records = audit.get("records")
    required = math.ceil(.05 * len(keys))
    if not isinstance(records, list):
        return None, reasons + ["repeat audit records missing"]
    seen_items, seen_ids, ordered, sets, first, hit5 = set(), set(), [], [], [], []
    for record in records:
        if not isinstance(record, dict):
            reasons.append("repeat audit record is not an object")
            continue
        key, rid, top = record.get("item_key"), record.get("repeat_id"), record.get("predictions")
        if not isinstance(key, str) or key not in keys or key in seen_items:
            reasons.append("repeat audit item missing, duplicated or outside panel")
            continue
        seen_items.add(key)
        if not _reference(rid) or rid in seen_ids or rid in keys:
            reasons.append("repeat audit ID missing, duplicated or not opaque")
            continue
        seen_ids.add(rid)
        if not _valid_top(top, catalog):
            reasons.append("repeat audit predictions must contain five distinct candidate names")
            continue
        original = judge["predictions"].get(key)
        if not _valid_top(original, catalog):
            continue  # Main-output validation supplies the reason.
        truth = json.loads(key)[0]
        ordered.append(top == original)
        sets.append(set(top) == set(original))
        first.append(top[0] == original[0])
        hit5.append((truth in top) == (truth in original))
    if len(seen_items) < required or len(ordered) < required:
        reasons.append(f"repeat audit needs {required} distinct complete panel items")
    result = dict(required_items=required, completed_items=len(ordered), panel_fraction=len(ordered) / len(keys),
                  ordered_agreement=float(np.mean(ordered)) if ordered else None,
                  set_agreement=float(np.mean(sets)) if sets else None,
                  top1_agreement=float(np.mean(first)) if first else None,
                  hit5_agreement=float(np.mean(hit5)) if hit5 else None,
                  agreement_is_report_only=True, reference=audit.get("reference"))
    return result, reasons


def _self_named(role, reply):
    terms = "|".join(re.escape(t) for t in NAME_TERMS.get(role, [role]))
    return bool(re.search(rf"\b({terms})(s|es)?\b", reply, re.I))


def _hit5_statistics(block_hits, roles, role_draws, question_draws, mappings):
    c, q, _ = block_hits.shape
    cells = block_hits[np.arange(c)[:, None], np.arange(q)[None, :], np.arange(c)[:, None]]
    point = float(cells.mean())
    draws = crossed_bootstrap(cells, role_draws, question_draws)
    lo, hi = [float(x) for x in np.percentile(draws, [2.5, 97.5])] if len(draws) > 1 else (None, None)
    null_roles = np.array([permuted_role_rates(block_hits, mapping) for mapping in mappings])
    null_macro = null_roles.mean(1)
    null_p99, role_p99 = float(np.percentile(null_macro, 99)), np.percentile(null_roles, 99, axis=0)
    observed_roles = cells.mean(1)
    # A role keeps its identity; shared question draws retain every fixed
    # wording and observed replicate mean in each sampled role/question cell.
    role_intervals = np.percentile(cells[:, question_draws].mean(2), [2.5, 97.5], axis=1) if len(draws) > 1 else None
    # D1(b) as THRESHOLDS words it (owner, 30/10): each role's hit@5 above its own null p99; the stricter count
    # with each role's 95% lower bound is reported beside it
    above = observed_roles > role_p99
    above_lo = role_intervals[0] > role_p99 if role_intervals is not None else np.zeros(c, dtype=bool)
    p = float((1 + (null_macro >= point).sum()) / (1 + len(mappings)))
    return dict(macro_hit5=dict(point=point, lo=lo, hi=hi), null_p99=null_p99,
                null_mean=float(null_macro.mean()), p=p,
                roles_above_own_null_p99=int(above.sum()),
                role_gate_basis="role hit@5 point above own null p99 (THRESHOLDS D1(b); owner, 30/10)",
                roles_lower_bound_above_own_null_p99=int(above_lo.sum()),
                criterion_met=bool(lo is not None and lo > null_p99 and int(above.sum()) >= 15),
                precision_met=bool(lo is not None and hi - lo <= .15 + 1e-12),
                per_role={role: dict(hit5=float(observed_roles[i]),
                                    hit5_ci95=[float(x) for x in role_intervals[:, i]] if role_intervals is not None else [None, None],
                                    null_p99=float(role_p99[i]),
                                    above_own_null_p99=bool(above[i]),
                                    lower_bound_above_own_null_p99=bool(above_lo[i])) for i, role in enumerate(roles)})


def score_naming(rows, roles, questions, payload, *, n_boot=2000, n_perm=9999, seed=20261030):
    """Compute D1 only. Parent analysis applies Holm with D2 and run integrity gates.

    ``passed`` is an unadjusted D1 threshold result, never the Holm family result.
    Synthetic or reduced panels/counts expose descriptive statistics but cannot
    yield a primary pass. ``available=False`` means no statistical verdict.
    """
    if type(n_boot) is not int or type(n_perm) is not int or n_boot < 1 or n_perm < 1 or type(seed) is not int or seed < 0:
        raise NamingError("n_boot/n_perm must be positive integers; seed must be a nonnegative integer")
    roles, questions, rows = list(roles), list(questions), list(rows)
    if not _strings(roles) or "default" in roles or not _ints(questions):
        raise NamingError("roles/questions must be unique nonempty lists; roles exclude default")
    result = dict(available=False, unavailable_reasons=[], criterion_met=None, precision_met=None,
                  confirmatory_eligible=False, eligibility_reasons=[], passed=False, decision_p=None, p=None,
                  judges={}, settings=dict(n_boot=n_boot, n_perm=n_perm, seed=seed),
                  family_adjustment="none here; parent combines D1 and D2 with Holm")
    reasons = result["unavailable_reasons"]
    if not isinstance(payload, dict):
        reasons.append("naming payload missing")
        return result
    approval = payload.get("approval", {})
    if not isinstance(approval, dict) or any(approval.get(f) is not True for f in ("owner_approved", "approved_before_judging", "frozen_before_labels")) or not _reference(approval.get("reference")):
        reasons.append("owner approval incomplete or not frozen before judging/labels")
    if not isinstance(approval, dict):
        approval = {}
    panel = payload.get("panel")
    if not isinstance(panel, dict):
        reasons.append("frozen judge panel missing")
        return result
    if (not _strings(panel.get("roles")) or set(panel["roles"]) != set(roles + ["default"])
            or not _ints(panel.get("questions")) or set(panel["questions"]) != set(questions)
            or not _ints(panel.get("wordings")) or panel["wordings"] != list(range(5))
            or not _ints(panel.get("replicates")) or panel["replicates"] not in (list(range(3)), list(range(5)))
            or panel.get("format") != "chat"):
        reasons.append("frozen judge panel does not match roles/questions/wordings/approved replicates/format")
        return result
    result["panel"] = {key: panel[key] for key in ("roles", "format", "questions", "wordings", "replicates")}
    result["approval"] = {key: approval.get(key) is True for key in ("owner_approved", "approved_before_judging", "frozen_before_labels")}
    result["approval"]["reference"] = approval.get("reference") if _reference(approval.get("reference")) else None
    expected = {_key(dict(zip(("role", "format", "wording", "question", "replicate"), values)))
                for values in itertools.product(panel["roles"], ["chat"], range(5), questions, panel["replicates"])}
    seen, selected = set(), {}
    for row in rows:
        key = _key(row)
        if key in seen:
            raise NamingError(f"duplicate row: {key}")
        seen.add(key)
        if key in expected:
            if type(row.get("valid")) is not bool or not isinstance(row.get("reply"), str):
                raise NamingError(f"row requires boolean valid and string reply: {key}")
            selected[key] = row
        elif row["replicate"] in panel["replicates"]:
            reasons.append(f"row outside frozen panel: {key}")
    if set(selected) != expected:
        reasons.append(f"frozen panel rows missing: {len(expected - set(selected))}")
    judges = payload.get("judges")
    if not isinstance(judges, list) or not all(isinstance(j, dict) and isinstance(j.get("id"), str) for j in judges):
        reasons.append("frozen judge IDs missing")
        return result
    ids = [j["id"] for j in judges]
    if len(ids) != len(set(ids)):
        raise NamingError("duplicate judge ID")
    if set(ids) != set(JUDGE_IDS):
        reasons.append("judge panel must contain GPT-6.1-Terra and Claude Sonnet 5.5 exactly")
    catalog = set(CANDIDATES_PATH.read_text().splitlines())
    if len(catalog) != 276 or not set(roles) <= catalog:
        raise NamingError("E3 candidate list must contain 276 unique names and every scored role")
    result["references"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (CANDIDATES_PATH, PROMPT_PATH, QUESTIONS_PATH)}
    audits = {}
    for judge in judges:
        predictions = judge.get("predictions")
        if not isinstance(predictions, dict):
            reasons.append(f"{judge['id']}: predictions missing")
            continue
        if set(predictions) != expected:
            reasons.append(f"{judge['id']}: prediction panel differs (missing={len(expected - set(predictions))}, extra={len(set(predictions) - expected)})")
        malformed = sum(not _valid_top(top, catalog) for top in predictions.values())
        if malformed:
            reasons.append(f"{judge['id']}: {malformed} predictions lack five distinct candidate names")
        audits[judge["id"]], failures = _audit(judge, expected, catalog)
        reasons.extend(f"{judge['id']}: {failure}" for failure in failures)
    if reasons:
        return result
    c, q, w, k = len(roles), len(questions), 5, len(panel["replicates"])
    ordered = [[[[selected[_key(dict(role=role, format="chat", wording=wi, question=question, replicate=rep))]
                  for rep in panel["replicates"]] for wi in range(w)] for question in questions] for role in roles]
    valid = np.array([[[[row["valid"] for row in word] for word in question] for question in role] for role in ordered])
    names = np.array([[[[_self_named(row["role"], row["reply"]) for row in word] for word in question] for question in role] for role in ordered])
    rng = np.random.default_rng(seed)
    role_draws = rng.integers(c, size=(n_boot, c))
    question_draws = rng.integers(q, size=(n_boot, q))
    # Share the exact null maps and bootstrap draws between judges.
    mappings = np.array([[rng.permutation(c) for _ in range(q)] for _ in range(n_perm)]).transpose(0, 2, 1)
    for judge in judges:
        pred = judge["predictions"]
        tops = [[[[pred[_key(row)] for row in word] for word in question] for question in role] for role in ordered]
        membership = np.array([[[[[target in top for target in roles] for top in word] for word in question] for question in role] for role in tops], dtype=float)
        hit1 = np.array([[[[top[0] == role for top in word] for word in question] for question in role_tops] for role, role_tops in zip(roles, tops)])
        block_hits = membership.mean((2, 3))  # replicate means inside each wording, fixed wordings averaged
        stats = _hit5_statistics(block_hits, roles, role_draws, question_draws, mappings)
        for i, role in enumerate(roles):
            stats["per_role"][role].update(hit1=float(hit1[i].mean()), output_coverage=1.,
                                           valid_rate=float(valid[i].mean()), self_naming_rate=float(names[i].mean()),
                                           attempts=q*w*k, valid_attempts=int(valid[i].sum()),
                                           invalid_attempts=int((~valid[i]).sum()))
        per_replicate = {}
        for ki, rep in enumerate(panel["replicates"]):
            rep_hits = membership[:, :, :, ki, :]
            diagonal = rep_hits[np.arange(c)[:, None, None], np.arange(q)[None, :, None],
                                np.arange(w)[None, None, :], np.arange(c)[:, None, None]]
            per_replicate[str(rep)] = dict(macro_hit5=float(diagonal.mean()), macro_hit1=float(hit1[..., ki].mean()),
                                           per_role={role: dict(hit5=float(diagonal[i].mean()),
                                                                hit1=float(hit1[i, ..., ki].mean()))
                                                     for i, role in enumerate(roles)})
        sensitivity = dict(use="approved primary panel")
        if k == 5:
            sensitivity = _hit5_statistics(membership[:, :, :, :3, :].mean((2, 3)), roles,
                                           role_draws, question_draws, mappings)
            sensitivity.update(use="descriptive sensitivity", replicates=[0, 1, 2],
                               macro_hit1=float(hit1[..., :3].mean()), passed=False)
        stats.update(macro_hit1=float(hit1.mean()), macro_self_naming_rate=float(names.mean()),
                     output_coverage=1., repeat_audit=audits[judge["id"]], per_replicate=per_replicate,
                     replicates_0_2=sensitivity)
        result["judges"][judge["id"]] = stats
    eligibility = result["eligibility_reasons"]
    if (n_boot, n_perm) != (2000, 9999):
        eligibility.append("reduced resampling counts are non-confirmatory; requires 2000 bootstrap and 9999 permutations")
    if set(roles) != set(PILOT_ROLES) or set(questions) != set(json.loads(QUESTIONS_PATH.read_text())):
        eligibility.append("reduced/changed role or question panel is non-confirmatory")
    result.update(available=True, criterion_met=all(j["criterion_met"] for j in result["judges"].values()),
                  precision_met=all(j["precision_met"] for j in result["judges"].values()),
                  decision_p=max(j["p"] for j in result["judges"].values()), confirmatory_eligible=not eligibility,
                  panel_attempts=len(expected), scored_attempts=c*q*w*k,
                  valid_attempts=int(valid.sum()), invalid_attempts=int((~valid).sum()),
                  panel_valid_attempts=sum(row["valid"] for row in selected.values()),
                  panel_invalid_attempts=sum(not row["valid"] for row in selected.values()),
                  self_naming_method="E3 lexical name terms; name mentions, not validated identity claims",
                  null_method="independent whole-role permutation per question; group by permuted truth role",
                  bootstrap_scope="macro: roles and questions crossed; per role: shared question draws; five wordings fixed; observed replicate means retained")
    result["p"] = result["decision_p"]
    result["passed"] = bool(result["criterion_met"] and result["precision_met"] and result["confirmatory_eligible"])
    return result
