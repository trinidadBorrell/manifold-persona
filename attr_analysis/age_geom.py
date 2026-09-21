"""Is the age ladder actually monotone, or does it double back?"""
import sys
import numpy as np
sys.path.insert(0, "attr_analysis")
from loader import LAYERS, load_all

LADDER = ["infant", "toddler", "adolescent", "teenager", "graduate",
          "newlywed", "parent", "grandparent", "retiree", "elder"]
AGES   = [1, 3, 13, 16, 22, 28, 38, 62, 66, 78]

means, _p, all_p = load_all(need_pts=False)
for L in LAYERS:
    a = means[L]["default"]
    miss = [p for p in LADDER if p not in means[L]]
    if miss:
        print("missing:", miss); break
    D = np.stack([means[L][p] - a for p in LADDER])
    print(f"\n=== layer {L} ===")
    print("  step norms between consecutive rungs:")
    steps = [np.linalg.norm(D[i+1]-D[i]) for i in range(len(LADDER)-1)]
    for i, s in enumerate(steps):
        print(f"    {LADDER[i]:12}->{LADDER[i+1]:12} {s:6.2f}")
    # turning: angle between consecutive step vectors. 0 deg = straight line
    print("  turn angle at each interior rung (0 = perfectly straight):")
    for i in range(1, len(LADDER)-1):
        u = D[i]-D[i-1]; v = D[i+1]-D[i]
        ang = np.degrees(np.arccos(np.clip(u@v/np.linalg.norm(u)/np.linalg.norm(v), -1, 1)))
        flag = "  <-- reverses" if ang > 90 else ""
        print(f"    {LADDER[i]:12} {ang:5.1f} deg{flag}")
    # how much of the ladder is captured by the straight infant->elder line
    line = D[-1] - D[0]; u = line/np.linalg.norm(line)
    proj = [(D[i]-D[0]) @ u for i in range(len(LADDER))]
    resid = [np.linalg.norm((D[i]-D[0]) - proj[i]*u) for i in range(len(LADDER))]
    print(f"  straight infant->elder line: |line| = {np.linalg.norm(line):.2f}")
    print("  rung:  along-line / off-line distance")
    for i, p in enumerate(LADDER):
        mono = "" if i == 0 or proj[i] >= proj[i-1] else "  <-- goes BACKWARD"
        print(f"    {p:12} {proj[i]:7.2f} {resid[i]:7.2f}{mono}")
