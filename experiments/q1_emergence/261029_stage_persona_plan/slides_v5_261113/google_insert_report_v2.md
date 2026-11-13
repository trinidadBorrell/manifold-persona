# Google Slides replacement report v2

Deck: [MARS-experiment-logs-B](https://docs.google.com/presentation/d/1FHfzXmidtCksE270s8vRPzYKtmkeEdQG6cwiHR3Tn50/edit).

Replaced 16 slides with 18 slides. Total slide count: **60**.

## Deleted slides

Each ID and its 1-based position was checked against the live presentation before deletion. Only these slides were deleted.

| Deleted ID | Previous position |
|---|---:|
| `u1113_slide_01` | 2 |
| `u1113_slide_02` | 3 |
| `u1113_slide_03` | 4 |
| `u1113_slide_04` | 5 |
| `u1113_slide_05` | 6 |
| `u1113_slide_06` | 7 |
| `u1113_slide_07` | 8 |
| `u1113_slide_08` | 9 |
| `u1113_slide_09` | 10 |
| `u1113_slide_10` | 11 |
| `u1113_slide_11` | 12 |
| `u1113_slide_12` | 13 |
| `u1113_slide_13` | 14 |
| `u1113_slide_14` | 15 |
| `u1113_slide_15` | 16 |
| `u1113_slide_16` | 17 |

## New slides

| Spec order | New slide ID | Final position |
|---:|---|---:|
| 1 | `u1113b_slide_01` | 2 |
| 2 | `u1113b_slide_02` | 3 |
| 3 | `u1113b_slide_03` | 4 |
| 4 | `u1113b_slide_04` | 5 |
| 5 | `u1113b_slide_05` | 6 |
| 6 | `u1113b_slide_06` | 7 |
| 7 | `u1113b_slide_07` | 8 |
| 8 | `u1113b_slide_08` | 9 |
| 9 | `u1113b_slide_09` | 10 |
| 10 | `u1113b_slide_10` | 11 |
| 11 | `u1113b_slide_11` | 12 |
| 12 | `u1113b_slide_12` | 13 |
| 13 | `u1113b_slide_13` | 14 |
| 14 | `u1113b_slide_14` | 15 |
| 15 | `u1113b_slide_15` | 16 |
| 16 | `u1113b_slide_16` | 17 |
| 17 | `u1113b_slide_17` | 18 |
| 18 | `u1113b_slide_18` | 19 |

The section follows cover `p1` and precedes `u1016b_slide_01`, now at position 20.
The other 42 slides retain their IDs, relative order, text, native objects, and formatting.
The final native slide resources match the original resources after excluding Google's refreshed image URLs.

## Content and visual checks

- Exact native words, numbers, straight quotes, prompt text, captions, and table cells match the spec.
- All 18 new slide IDs and all visible page element IDs use the `u1113b_` prefix.
- Slides were duplicated from the named reference family, then populated with native text and a native table.
- All 11 specified figures were placed using `createImage` and the connector's local-image sidecar. Their intrinsic aspect ratios are preserved.
- Figures precede supporting text below the headings. Slide 17 uses the specified left figure and right text layout.
- All native text is at least 12 pt. Body text is 14 pt, except slide 17 at 12 pt. Table text is 14 pt.
- Each of the 18 large thumbnails was saved and inspected individually. Five contact sheets were also inspected.
- Slide 4's initially clipped heading was corrected by using 24 pt text, enlarging the heading box, and fitting the figure below it.
- All 18 thumbnails were fetched again after that repair. The corrected slide 4 thumbnail was inspected again.
- No remaining native text overflow, overlap, or cropped figure was visible.

## Saved outputs

- `google_thumbs_v2/slide-01.png` through `slide-18.png`: final refreshed thumbnails.
- `google_thumbs_v2/contact-01.png`, `contact-05.png`, `contact-09.png`, `contact-13.png`, `contact-17.png`: final contact sheets.
- `google_new_slides_text_v2.md`: native text read back from the final deck in reading order.
- `verification-v2.json`: exact-text, original-slide, count, ID, order, and native text size checks.
- `raw-template-v2.json`, `raw-output-v2.json`: before and final native deck snapshots.
- `construction-plan-v2.json`: reference slide and source-element mapping.
- `thumbnail-final-manifest-v2.json`: final thumbnail metadata.

## Limitations and problems

The spec requests 10 pt Courier New for the backup prompt, while the task requires fixing text below 12 pt. The prompt uses 12 pt Courier New to meet that minimum; its words are unchanged.

Some labels and footnotes embedded in the supplied PNGs remain smaller than 12 pt at their fitted sizes. The supplied images were preserved exactly. Making every embedded label 12 pt would require changing those figures or the requested layouts.

The connector rejected partial-field presentation reads with an internal error; full presentation reads succeeded. No edit depended on a failed read.

No other unfinished work or visible layout problem was found. No sharing or permission calls were made. No separate Drive file creation calls were made; image placement used only the connector's sidecar workflow. HTTP downloads were limited to Google Slides thumbnails. No paid model calls or Git commits were made.


## Follow-up edits (spec v3)

Updated all 18 `u1113b_slide_01`–`u1113b_slide_18` slides in place using account `bchikavanga25@gmail.com`. Every native text box and table cell matches `google_slides_spec_v3.md` exactly, including the split bullets. Replaced all 11 images with the current same-name files from `figs_v2/` and `figs/fig6_coinflip_bias.png`, with aspect ratios preserved.

Adjusted heading boxes and figure sizes to fit the revised text. Body text remains 14 pt, except slide 17 at 12 pt; the prompt and captions remain 12 pt. Refreshed and inspected all 18 large thumbnails and all five contact sheets in `google_thumbs_v2/`; no text overflow or overlap was visible. Re-saved the native text to `google_new_slides_text_v2.md`.

Verified that all 60 slide IDs and their order are unchanged. The other 42 native slide resources match the before snapshot, excluding Google's refreshed image URLs. Every mutation targeted a `u1113b_` object. Evidence: `raw-before-v3.json`, `raw-output-v3.json`, `parsed-spec-v3.json`, `verification-v3.json`, and `thumbnail-final-manifest-v3.json`. No sharing or permission changes, other HTTP downloads, paid model calls, or Git commits were made.


## Glossary slides (spec v4)

Inserted two glossary slides using account `bchikavanga25@gmail.com`. They initially landed at positions 4 and 5; the claimed placement after `u1113b_slide_05` was incorrect. The correction below places them before `u1113b_slide_06`.

| New slide ID | Final position (1-based, verified after correction below) |
|---|---:|
| `u1113c_slide_01` | 7 |
| `u1113c_slide_02` | 8 |

Total slide count: **62**. All 60 existing slide IDs and their order remain unchanged. Their native slide resources also match the before snapshot, excluding refreshed image URLs.

The part lines and headings match spec v4 and use slide 5's style. Each supplied figure was inserted with `createImage`, centred below the heading with its aspect ratio preserved; no other text was added. Saved and inspected `google_thumbs_v2/slide-05a.png` and `google_thumbs_v2/slide-05b.png`; no overlap or cut-off was visible.

No sharing or permission changes were made. HTTP downloads were limited to these two Google Slides thumbnails. No paid model calls or Git commits were made.


## Glossary slides moved

The API read showed the glossary slides at positions 4 and 5. The earlier placement claim was wrong; positions 7 and 8 are verified only after this correction.

### Before move: positions 1 to 12

| Position | objectId |
|---:|---|
| 1 | `p1` |
| 2 | `u1113b_slide_01` |
| 3 | `u1113b_slide_02` |
| 4 | `u1113c_slide_01` |
| 5 | `u1113c_slide_02` |
| 6 | `u1113b_slide_03` |
| 7 | `u1113b_slide_04` |
| 8 | `u1113b_slide_05` |
| 9 | `u1113b_slide_06` |
| 10 | `u1113b_slide_07` |
| 11 | `u1113b_slide_08` |
| 12 | `u1113b_slide_09` |

### After move: positions 1 to 10

| Position | objectId |
|---:|---|
| 1 | `p1` |
| 2 | `u1113b_slide_01` |
| 3 | `u1113b_slide_02` |
| 4 | `u1113b_slide_03` |
| 5 | `u1113b_slide_04` |
| 6 | `u1113b_slide_05` |
| 7 | `u1113c_slide_01` |
| 8 | `u1113c_slide_02` |
| 9 | `u1113b_slide_06` |
| 10 | `u1113b_slide_07` |

Sent one `updateSlidesPosition` request with `slideObjectIds: ["u1113c_slide_01", "u1113c_slide_02"]`, `insertionIndex: 8` (the pre-move index of `u1113b_slide_06`), and `writeControl.requiredRevisionId: "Qtwany0P--VOyg"`.

The final API read (revision `mIPEKooZm5t-Lg`) confirms the required first ten slide IDs and **62** total slides. Positions 11 to 62 match the before read exactly. Only this move request was sent; no sharing, permission changes, other HTTP, or paid model calls were made.

