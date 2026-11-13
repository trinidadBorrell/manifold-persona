"""Role SD on Han Xiao's axis with no judge filter (Olmo 3 7B, layer 16). Exploratory check, 13/11/2026.

Part B reports the role SD on judge-filtered role vectors (0.613 at SFT, 0.715 at DPO). This asks whether the roles
still move further apart at DPO when every reply counts. Same definitions as Part B (np.std of raw axis projections
over the roles present in both stages) and Part D (unfiltered role vector = mean over all replies; the default is
never filtered). CPU only, reads the local release copy.

    .venv/bin/python experiments/q1_emergence/261112_han_reanalysis/role_sd_unfiltered.py
"""
import json
from pathlib import Path

import numpy as np

import han_reanalysis as hr
from han_reanalysis import pa

DATA = hr.HERE / "data/han_release"
OUT = hr.HERE / "results/role_sd_unfiltered.json"


def stage_projection(stage):
    roles = hr.activation_roles(DATA, stage)
    vecs = []
    for role in roles + ["default"]:
        _, acts = hr.load_acts(DATA, stage, role)
        vecs.append(acts.mean(axis=0))
    vecs = np.stack(vecs)
    return roles, pa.axis_projections(vecs[:-1], vecs[-1])


proj = {}
for s in hr.STAGES:
    roles, p = stage_projection(s)
    proj[s] = dict(zip(roles, p))
    print(s, len(roles), "roles", flush=True)

out = dict(label=hr.LABEL, check="role SD on Han's axis, no judge filter", layer=hr.LAYER, exploratory=True, pairs={})
for a, b in hr.PAIRS:
    common = sorted(set(proj[a]) & set(proj[b]))
    out["pairs"][f"{a}->{b}"] = dict(n_common=len(common),
                                     role_sd_raw={s: float(np.std([proj[s][r] for r in common])) for s in (a, b)})
OUT.write_text(json.dumps(out, indent=1))
print(json.dumps(out["pairs"], indent=1))
