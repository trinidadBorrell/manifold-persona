"""Pick k personas by k-medoid on the clean role cloud. `default` is forced in."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from manifold import pipeline as P
from manifold.subsets import kmeans_medoid_roles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cloud = P.load_cloud(view="prompt_avg", seed=args.seed)
    if not cloud.manifest.get("sink_factor"):
        raise RuntimeError("cloud has no sink_factor — it predates the attention-sink fix")
    sel = kmeans_medoid_roles(cloud, args.k, seed=args.seed)
    roles = sel["roles"]
    Path(args.out).write_text(json.dumps(roles, indent=1))
    print(f"model={cloud.manifest.get('model_name')} layer={cloud.layer} "
          f"roles={len(cloud.role_names)}")
    print(f"selected {len(roles)}: {roles}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
