# Jev role judge vs owner role scores: rule fixed before comparison (16/10/2026)

Written after the owner finished 50 role scores and before any Jev role output was read.

- Data: the owner's 50 scores (human/owner_role50_scores_2026-10-16.json; subset human/role50_subset_2026-10-16.json: 35 target, 8 close, 7 distant) and Jev role answers (jev/role_full).
- Primary: quadratic-weighted Cohen's kappa between the owner score (0-3) and Jev's enactment level rounded to the nearest level. Usable if kappa >= 0.6 (the same bar as the broken judges).
- Secondary: Spearman correlation between the owner score and Jev's probability-weighted score; binary agreement "owner 3" vs "Jev plays_role > 0.5"; confusion table; 95% bootstrap CIs (2,000 draws).
- No threshold or prompt is tuned on these 50 scores.
