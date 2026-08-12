"""Fetch the 40q response cloud, keep only the primary layer, drop the 8.4GB blob.

NOTE: STALE. The cloud actually in use,
``data/embeddings_roles_resp_40q``, keeps ALL 37 layers (manifest
primary_layer=19). This slicer writes a single-layer copy to a different
dir (``data/embeddings_roles_resp40``) that no current script reads.
Kept for provenance of how a slimmed cloud could be produced.
"""
import json, shutil
from pathlib import Path
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download
from manifold_persona.hf_utils import read_token

RID = "triniborrell/manifold-persona-roles-response-40q"
OUT = Path("data/embeddings_roles_resp40")
OUT.mkdir(parents=True, exist_ok=True)
tok = read_token()

man = json.load(open(hf_hub_download(RID, "manifest.json", repo_type="dataset", token=tok)))
meta = pd.read_csv(hf_hub_download(RID, "metadata.csv", repo_type="dataset", token=tok))
L = man["primary_layer"]
print(f"manifest primary_layer={L}  n_records={man['n_records']}  n_layers={man['n_layers']}")

print("downloading prompt_avg.npy (8.4 GB) ...", flush=True)
p = hf_hub_download(RID, "prompt_avg.npy", repo_type="dataset", token=tok)
arr = np.load(p, mmap_mode="r")
print("downloaded shape", arr.shape, arr.dtype, flush=True)
assert arr.shape == (man["n_records"], man["n_layers"], man["hidden"])

# Keep the primary layer only, but preserve the [N, n_layers, hidden] loader
# contract with n_layers=1 so io.load_layer works unchanged.
sub = np.ascontiguousarray(arr[:, L, :])[:, None, :]
np.save(OUT / "prompt_avg.npy", sub)
print("wrote", OUT / "prompt_avg.npy", sub.shape, sub.nbytes / 1e6, "MB")

meta.to_parquet(OUT / "metadata.parquet", index=False)
man2 = dict(man)
man2.update({"n_layers": 1, "primary_layer": 0, "source_layer": L,
             "source_repo": RID, "views": ["prompt_avg"],
             "note": f"single layer {L} extracted from the 37-layer HF cloud"})
json.dump(man2, open(OUT / "manifest.json", "w"), indent=2)

del arr
Path(p).unlink(missing_ok=True)     # drop the 8.4 GB blob
blobs = Path(p).resolve().parent
print("done. cache blob removed.")
