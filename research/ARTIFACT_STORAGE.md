# Experiment artifact storage

This project uses an experiment-first layout. The [Hughes–Perez workflow](https://www.lesswrong.com/posts/6P8GYb4AjtPXx6LLB/tips-and-code-for-empirical-research-workflows) recommends dated experiment folders and light scripts in Git. It distinguishes fast de-risk work from extended projects. The Hugging Face paths below are this project's storage adaptation, not a rule from their article.

## New experiments

| Place | Path | Contents |
|---|---|---|
| Git | `experiments/<topic>/<YYMMDD_name>/` | Numbered scripts or notebooks, a short run note, and settings. |
| Private Hugging Face | `experiments/<topic>/<YYMMDD_name>/` | Large inputs, saved state, and outputs when needed. |
| Project index | `research/EXPERIMENT_INDEX.md` | Question, mode, code or note, exact artifact paths, and next decision. |

De-risk runs can use a short notebook and note. They do not need a full pipeline or a Hub upload. For an extended run, keep reused code in stable modules and record the code revision, inputs, settings, seed, command, and outputs. When a run uses the Hub, record its dataset ID, exact revision, and path in the run note. Never use `main` as the only record of an input or result.

## Historical outputs

The private dataset `brend0nc/manifold-persona-experiment-artifacts` uses `legacy/output/<original_path>` for old files that lack a reliable dated experiment record. This preserves the old path. It does not invent a run date, method, or scientific result. The 2026-10-05 transfer verified all 15,536 selected files (30.155 GiB logical) at data revision `ed246c1629cfd1fd223e7f8516d9d6f60fd94e08`.

The initial local map is `output/release_audit_2026-10-02/research_artifact_map_2026-10-05.jsonl`. The final lookup table is `output/release_audit_2026-10-02/research_artifact_locations_2026-10-05.jsonl`. Private copies are at `catalog/2026-10-05/` in the new dataset. Each row records a source path, SHA-256, storage status, and exact remote location when verified. The local verification receipt is `output/release_audit_2026-10-02/research_archive_verification_2026-10-05.json`. The map also records 2,322 files whose bytes matched existing private Hub files, 2,368 local cache files, 11 files in external mirrors, and two local links. These counts describe storage only; they cover the map snapshot, not later output changes.

Existing private dataset files remain at these pinned revisions:

| Dataset | Revision |
|---|---|
| `brend0nc/manifold-persona-roles-response-40q-base` | `b7c6b77d5dbfabd06cdd8ffe6d76ae6f9d932beb` |
| `brend0nc/manifold-persona-olmo1b-stages` | `22250ccc62345c7ae84ba1cb86e52100ad83ed54` |
| `brend0nc/manifold-persona-olmo7b-stages` | `9ef49b677fb4b7bf46d32d468d11efb927d1db37` |
| `brend0nc/manifold-persona-organisms` | `f8e5a5e57d22c30c6bf4f7b4e38fe2efc9077e9e` |

The map compares local bytes with remote file hashes at these revisions. A later test restored 17 selected files (41.907 GiB) across all five private datasets. It checked full SHA-256 values and opened the files. This does not test every file or validate a scientific claim. Six external-mirror files matched a public source by hash; that source is not an independent backup. Do not upload external mirrors again without source and rights checks. Do not delete local caches or links based on this map.

The cleanup review is `output/release_audit_2026-10-02/CLEANUP_READINESS_2026-10-05.md`. It holds 24 paths changed after the map, 31 new paths, and 2,381 local-only or unverified paths. Seven files under `output/olmo7b_backups/` passed full restore and current-local-hash checks. They remain local until an independent backup and exact removal approval are in place.

## Find one historical file

Search the final lookup table for its original `source_path`:

```sh
rg 'olmo7b_backups/source_sha256.json' output/release_audit_2026-10-02/research_artifact_locations_2026-10-05.jsonl
```

Use the row's `remote_repo`, `remote_revision`, and `remote_path` to download the exact bytes. For example:

```sh
hf download brend0nc/manifold-persona-experiment-artifacts \
  legacy/output/olmo7b_backups/source_sha256.json \
  --type dataset \
  --revision ed246c1629cfd1fd223e7f8516d9d6f60fd94e08 \
  --local-dir ./restored
```

## Two separate decisions

**Storage status** says whether exact bytes have a known remote location. **Research status** says what the data supports after method checks and independent review. A successful upload does not change research status. The historical release audit puts all groups on HOLD for public release. Keep the new dataset private.

Do not remove any local source file until its remote bytes match, a restore test succeeds, and the owner approves the exact removal list. Keep a second independent backup for data that cannot be recreated.
