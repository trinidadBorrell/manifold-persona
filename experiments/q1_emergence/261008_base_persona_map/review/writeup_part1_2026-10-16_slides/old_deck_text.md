# Old deck text

Source: https://docs.google.com/presentation/d/1FHfzXmidtCksE270s8vRPzYKtmkeEdQG6cwiHR3Tn50/edit

Slide indices are 1-based positions in the full deck. Slides whose object IDs start with `u1016_` are excluded. Native text is copied without rewriting, including source control characters. Image descriptions below use associated native text and source image metadata; image pixels and image titles could not be verified through the returned asset pointers. No additional HTTP requests were made.

## Slide 1 — p1

### Text — p1_i85

```text
Persona geometry - experiment log
```

### Text — p1_i86

```text
Brendon
```

### Notes

_No native notes text._

## Slide 15 — g38b9c492913_0_0

### Text — g38b9c492913_0_3

```text
Update 9/11
```

### Text — g38b9c492913_0_4

```text
reliability check: I ran a reliability check on the assistant axis, to confirm if the cosines from the last meet were noise
RESULT: the measure of the assistant direction seems random in the base. SFT is where it becomes steady enough to trust it 
supervisor probe: built a supervised probe, to help the simple axis find anything of use in the base model
I wanted to answer if there really no assistant direction in base or if it’s hidden
RESULT: what was weird is that the supervised linear probe was able to find a direction in the base model, it seems that the simple axis I was working with was too weak to see it 
text-control: ran a text control on that probe, because the probe can cheat by reading the answer wording rather than the real internal states
RESULT: the answer text on its own couldn’t identify the assistant, 
and the direction still held after I stripped some of the wording out of the activations,so just further confirms that the probes results are dependent on the assistant direction itself
assistant direction: measured the probe to probe rotation across the various post training stages
RESULT: confirmed the base model’s directions but it seems to be orthogonal to the final model, where most of the rotating happens in SFT
I measured persona cloud shape across all the training stages against a noise band
RESULT: the cloud shape doesn’t stay steady across training but the biggest change happens at SFT
I started running a fine grained pretraining sweep, so that I can work with more checkpoints
RESULT: STILL RUNNING
Been looking to work with Trinidad’s steering code on OLMo, to check whether the base model is less robust to steering
RESULT: TO BE RAN
```

### Notes

_No native notes text._

## Slide 16 — g38b9c492913_0_15

### Text — g38b9c492913_0_18

```text
Experiment setup
```

### Text — g38b9c492913_0_21

```text
OLMo-2-7B, 276 roles, 200 prompts per role, response-token activations read at three layers (9, 17, 23 = early/mid/late)
The base/SFT/DPO/RLVR clouds are the n_q=40 ones from prior experiments
```

### Text — g38b9c492913_0_24

```text
reliability check: I split each role's answers into two random piles and worked out the assistant direction from each pile on its own 
If the two come out the same, the direction is solid
I ran that 40 times and averaged, across three layers
supervised probe: Instead of the simple average, I trained a small classifier to tell the assistant's answers from the personas' answers, using the model's internals. It can weigh which dimensions matter, so it catches a weak signal the usual method misses. I checked it holds up on fresh data and repeats across splits of role’s answers
text-control: I turned each answer into a text fingerprint using an outside model. Then I checked two things, whether that text alone can spot the assistant, and whether the probe still works once I take the text part back out of the activations
assistant direction: I worked out the probe direction for each training stage and measured the angle between them
persona cloud shape: I simplified each stage down to its 276 persona centres and measured how spread out they are and how many separate groups they form. Then I compared the change across stages to a noise band from reshuffling the data, to see if the shape really moves
```

### Notes

_No native notes text._

## Slide 17 — g38b9c492913_0_12

### Text — g38b9c492913_0_30

```text
When the assistant direction stops being noise…
```

### Text — g38b9c492913_0_38

```text
At base, the direction is noise. The two halves of data disagree.
It becomes reliable at SFT, and stays reliable through DPO and RLVR.
Same story at all three depths
later layers seems to be more reliable
```

### Notes

_No native notes text._

Image `g38b9c492913_0_35`: Chart of assistant-direction agreement between split halves across Base, SFT, DPO, and RLVR at layers 9, 17, and 23; associated text says agreement becomes reliable at SFT.

## Slide 18 — g38b9c492913_0_9

### Text — g38b9c492913_0_42

```text
Base already separates default assistant from personas  (87% correct) The concept is there in pretraining but doesn’t hold up 
SFT roughly doubles the gap (100%), then DPO and RLVR hold it
The simple axis couldn't see this at base, but the probe can

```

### Notes

_No native notes text._

Image `g38b9c492913_0_43`: Chart of supervised separation between the default Assistant and personas across training stages; associated text reports 87% at Base and 100% at SFT.

## Slide 19 — g38b9c492913_0_6

### Text — g38b9c492913_0_47

```text
Text alone scores 50%, similar to guessing. The wording does not reveal the assistant
The probe on the activations scores 78%. With the text removed it stays at 89%.
Removing the text takes out 99.9% of the text variance, and the probe still works. The result is the same at layers 9, 17, and 23.
```

### Notes

_No native notes text._

Image `g38b9c492913_0_44`: Chart comparing assistant detection from text, activations, and activations with the text component removed; associated text reports 50%, 78%, and 89%.

## Slide 20 — g38b9c492913_0_48

_No native slide text._

### Notes

_No native notes text._

Image `g38b9c492913_0_52`: Image-only slide; the connector returned an image asset pointer that could not be visually read here. Chart contents and title are unverified.

## Slide 21 — g38b9c492913_0_53

### Text — g38b9c492913_0_58

```text
Finish pretraining
Steering test 
Sycophancy per stage 
```

### Text — g38b9c492913_0_59

```text
Next steps
```

### Notes

_No native notes text._

## Slide 22 — g3f9b6cfdfa2_0_0

### Text — g3f9b6cfdfa2_0_3

```text
Update 8/31
```

### Text — g3fa1d3531c0_0_6

```text
I ran the same stage study on the released 7B checkpoints.
The measured assistant direction lines up with its final orientation by SFT in 7B and by DPO in 1B.
The default Assistant stays fairly loose in 7B. A text-only control reproduces most H0, entropy and MLE results.
These releases differ in training data and recipe. The scaffold also changes the 1B result.


```

### Notes

_No native notes text._

## Slide 23 — wk831_setup

### Text — wk831_setup_title

```text
Experiment setup
```

### Text — wk831_setup_body

```text
Model stages: Base, SFT, DPO and RLVR in the 1B and 7B releases.
Panel: 275 role prompts plus the paper's default Assistant; 200 answers per condition.
Names: default Assistant = five neutral baseline instructions. The role named assistant is a separate role.
Prompt scaffold used in every run: User: What causes the seasons on Earth? Assistant: The tilt of Earth's axis... User: How do you make a cup of tea? Assistant: Boil water...

```

### Notes

_No native notes text._

## Slide 24 — wk831_direction

### Text — wk831_direction_title

```text
The 7B direction lines up by SFT
```

### Text — wk831_direction_result

```text
Result: At SFT, cosine with the final RLVR direction is 0.75 in 7B and 0.44 in 1B.
```

### Notes

_No native notes text._

Image `wk831_direction_img`: Chart comparing alignment with the final RLVR assistant direction across training stages for 1B and 7B; associated text reports SFT cosines of 0.44 and 0.75.

## Slide 25 — wk831_rank

### Text — wk831_rank_title

```text
The default Assistant stays fairly loose in 7B
```

### Text — wk831_rank_result

```text
Result: At SFT, its H0 rank is 80 in 7B and 95 in 1B. Lower means tighter than more roles.
```

### Notes

_No native notes text._

Image `wk831_rank_img`: Chart comparing the default Assistant's H0 rank across training stages for 1B and 7B; associated text reports SFT ranks of 95 and 80.

## Slide 26 — wk831_link

### Text — wk831_link_title

```text
The H0 link appears earlier in 7B
```

### Text — wk831_link_result

```text
Result: 7B moves from +0.10 at Base to -0.46 at SFT. 1B reaches -0.34 at DPO. Negative means nearby roles have tighter clouds.
```

### Text — g3fa21bb9584_8_10

```text
grey = 1B blue = 7B
Controls: cloud variance and distance from origin.

```

### Notes

_No native notes text._

Image `g3fa21bb9584_8_9`: Chart comparing the assistant-axis/H0 link across training stages for 1B (grey) and 7B (blue), controlling for cloud variance and distance from origin.

## Slide 27 — wk831_text

### Text — wk831_text_title

```text
Answer text reproduces most H0 rankings
```

### Text — wk831_text_result

```text
Result: From SFT onward, H0 rank agreement is 0.94 to 0.97. MLE is 0.91 to 0.96. Participation ratio is 0.22 to 0.40.
```

### Notes

_No native notes text._

Image `wk831_text_img`: Chart comparing text-based and activation-based rankings across training stages for H0, MLE, and participation ratio.

## Slide 28 — wk831_voice

### Text — wk831_voice_title

```text
Backup: role voices move most at SFT
```

### Text — wk831_voice_result

```text
Result: In both releases, role-to-default text distance peaks at SFT and drops at DPO.
```

### Notes

_No native notes text._

Image `wk831_voice_img`: Chart comparing role-to-default text distance across training stages in the 1B and 7B releases; associated text says distance peaks at SFT.

## Slide 29 — wk831_scaffold

### Text — wk831_scaffold_title

```text
The prompt scaffold changes the 1B result
```

### Text — wk831_scaffold_result

```text
Result: Removing the two example answers moves the 1B RLVR H0 rank from 60 to 10.5. The full 7B run is still missing.
```

### Notes

_No native notes text._

Image `wk831_scaffold_img`: Chart comparing the 1B RLVR default Assistant H0 rank with and without two scaffold example answers; associated text reports a change from 60 to 10.5.

## Slide 30 — wk831_next

### Text — wk831_next_title

```text
Current reading and next test
```

### Text — wk831_next_body

```text
With the shared prompt scaffold, the 7B direction lines up by SFT. The 1B direction lines up by DPO.
Answer text reproduces most H0, entropy and MLE results.
The releases also differ in training data and recipe. I would run the full 7B panel without the scaffold next.
To discuss: should that run happen before steering?

```

### Notes

_No native notes text._

## Slide 31 — p2

### Text — p2_i92

```text
Update 8/26
```

### Text — p2_i93

```text
•  Ran the robustness checks from the thread. Three pass, one flagged the base model.
•  Found a silent bug in skdim on the way. Fixed it; results identical to the last digit.
•  New control: a sentence embedder on the answer text recovers most of our metrics. In line with the axis paper’s persona drift, embedding text with a small embedder + ridge regression predicts the position on the axis well
•  Next up: prompt-side clouds, where the text control can't apply.
```

### Notes

### Text — p2:notes_i89

```text
The week's summary, then the details, most important first.
```

## Slide 32 — p3

### Text — p3_i99

```text
Revisiting the naming issue discussed on slack
```

### Text — p3_i100

```text
The paper's term is the default Assistant, its default persona. That's our `default` condition (the five no-role instruction sets). I'll say "the default Assistant" from here on.
"assistant", one of the 275 characters. This mix-up is what made rank 1 look weird a couple of weeks ago.
Metrics: I'm dropping the names I made up (bunched-ness, evenness, simplicity, rank). They confused more than they helped. Back to the panel names: H0 persistence (how tight the cloud is), persistence entropy (how even), MLE intrinsic dimension , participation ratio.

```

### Notes

### Text — p3:notes_i96

```text
Two naming problems have already cost us thread time. Settling both here.
```

## Slide 33 — p4

### Text — p4_i106

```text
Recomputing the metrics from answer text alone
```

### Text — p4_i107

```text
Experiment:
1.   Take the instruct run: 276 characters × 200 answers each.
2.   Embed each answer's text with all-mpnet-base-v2. No activations anywhere in this.
3.   Compute the same per-character metrics on the text vectors, and the same axis-metric partial correlations (answer length held fixed, as usual).
4.   Compare with the activation versions: same character ordering, same links?
```

### Notes

### Text — p4:notes_i103

```text
Step 2 is the point: there are no activations anywhere in this pipeline.
```

## Slide 34 — p5

### Text — p5_i113

```text
Recomputing the metrics from answer text alone
```

### Text — p5_i114

```text
Result: the text recovers most of it.
```

### Notes

### Text — p5:notes_i110

```text
Participation ratio is the one metric the text can't reproduce. Worth a follow-up.
```

Image `p5_i115`: Chart comparing cloud metrics and axis links computed from answer-text embeddings with those computed from activations; source image description: fig_q9.png.

## Slide 35 — p6

### Text — p6_i121

```text
Consequences for our claims
```

### Text — p6_i122

```text
The axis paper shows text embeddings track the axis (R² 0.53–0.77 for where a persona sits). The new part is that H0 persistence, entropy and MLE are mostly recoverable from the text too, along with their links to the axis. (vec2vec found that when different models read the same text, they all end up organizing it the same way so, alike that you can convert one model's representation into another's without any matching examples to learn from )
So response clouds don’t seem to back a claim about internal structure beyond what the words already show
The staircase exp results are unaffected. It shows up in the words and the geometry alike.
The paper gets past this with steering. Our version is prompt-side clouds: if the H0 link is there before the model has written anything, it isn't the words.
```

### Notes

### Text — p6:notes_i118

```text
The paper has the same measurement basis, so the same gap. Their answer is steering; ours can be prompt-side clouds.
```

## Slide 36 — p7

### Text — p7_i128

```text
Checks 
```

### Text — p7_i129

```text
Experiment:
1.   k in the MLE estimator: sweep 5 to 40, watch the ordering and the axis link.
2.   skdim bug: the K you pass the constructor is silently ignored (neighborhood_based=True path). Fix it, rerun everything.
3.   Duplicate question: q7 and q227 are the same string in the 240-question set. Drop each in turn, rerun.
4.   Truncation: answers that hit the 128-token cap end mid-sentence and look alike. Hold the capped share fixed, see if the axis link survives.
```

### Notes

### Text — p7:notes_i125

```text
One line per check. All from the thread with Thomas.
```

## Slide 37 — p8

### Text — p8_i135

```text
Checks - continued
```

### Text — p8_i136

```text
Result: three pass; truncation kills the base-model numbers.
```

### Table — p8_i137

Cell (0, 0):

```text
Check
```

Cell (0, 1):

```text
Outcome
```

Cell (0, 2):

```text
Verdict
```

Cell (1, 0):

```text
1. k sweep (5 to  40)
```

Cell (1, 1):

```text
ordering k10 to k20 agrees 0.994; axis link −0.887 vs −0.906
```

Cell (1, 2):

```text
pass
```

Cell (2, 0):

```text
2. skdim bug
```

Cell (2, 1):

```text
fixed; JSON md5 and PNG byte-identical
```

Cell (2, 2):

```text
pass
```

Cell (3, 0):

```text
3. duplicate question
```

Cell (3, 1):

```text
≤ 0.001 change anywhere
```

Cell (3, 2):

```text
pass
```

Cell (4, 0):

```text
4. truncation - instruct
```

Cell (4, 1):

```text
H0 link −0.863 to −0.782 with length + cap fixed
```

Cell (4, 2):

```text
pass
```

Cell (5, 0):

```text
4. truncation -  Qwen base
```

Cell (5, 1):

```text
+0.144 to −0.272,  sign flip?
```

Cell (5, 2):

```text
don't trust base stage
```

### Notes

### Text — p8:notes_i132

```text
Files: output/mle_k_sweep, dup_question_check, trunc_link_qwen, stage2/trunc_link_control.json. Each check is a script + JSON.
```

## Slide 38 — p9

### Text — p9_i143

```text
Next
```

### Text — p9_i144

```text
1.   Prompt-side clouds: activations before the model writes anything, so the text control can't apply. The clean test of whether the geometry is real.
2.   Text control across the four OLMo stages (about an hour): does the text's share of the signal grow with training?
3.   Steering: Trinidad's track, unchanged.

To discuss: is prompt-side the right decisive test, and how do we word the text result in the write-up.
```

### Notes

### Text — p9:notes_i140

```text
If we're short on time, land here.
```

## Slide 39 — p10

### Text — p10_i150

```text
Update 8/14 (recap)
```

### Text — p10_i151

```text
•  The headline result. Nothing this week changed it.
```

### Notes

### Text — p10:notes_i147

```text
Older section, kept below per the log format.
```

## Slide 40 — p11

### Text — p11_i157

```text
The default Assistant becomes special at DPO
```

### Text — p11_i158

```text
Result: its H0-persistence percentile goes 100 → 95 → 61 → 60 across base → SFT → DPO → RLVR.
```

### Text — p11_i160

```text
The y-axis says "bunched-ness rank": that's this percentile (older figure, old name). Lower = more special.
```

### Notes

### Text — p11:notes_i154

```text
Grey = before preference training, green = after. Qwen is the finished-model reference. Error bars: bootstrap over questions.
```

Image `p11_i159`: Chart of the default Assistant's H0-persistence percentile across Base, SFT, DPO, and RLVR, with Qwen as a reference; source image description: fig_headline.png.

## Slide 41 — p12

### Text — p12_i166

```text
Pirate voice in the raw text
```

### Text — p12_i167

```text
Result: zero-shot, strict pirate words go 0 → 43.5 → 75.5 → 77.5 across the stages (±6 pts, N = 200).
```

### Notes

### Text — p12:notes_i163

```text
Same DPO weights in the left and middle panels, only the prompt changed. The two example Q&As were muzzling the voice.
```

Image `p12_i168`: Chart of pirate-word frequency across training stages and prompt conditions; source image description: fig_pirate_evidence.png.

## Slide 42 — p13

### Text — p13_i174

```text
The direction and the answers move at the same stage
```

### Notes

### Text — p13:notes_i171

```text
Right panel is recounted - the first version of this figure had stale numbers (74.6/23.0/23.7). Recount lives in pirate_dup_recount.json.
```

Image `p13_i175`: Chart comparing assistant-direction change with answer changes across training stages; source image description: fig_axis_dup_corrected.png.

## Slide 43 — p14

### Text — p14_i181

```text
Measurement details
```

### Text — p14_i182

```text
Clouds:  200 layer-19 activation points per character, mean over response tokens. Same basis as the paper, which is why the text gap applies to both.
Text control:  all-mpnet-base-v2 on answer text; links are partial correlations with answer length held fixed.  output/q9_embedder_null/
Pirate prompt, verbatim:  "You are a pirate captain who has sailed the seven seas in search of treasure and adventure."  40 questions × 5 phrasings, chat template, zero-shot, greedy, 128 new tokens. Metric: share of 200 answers with ahoy / matey / arr / scallywag. SE = √(p(1−p)/N) → ±6.9 / ±6.0 / ±5.8 points.
Check files:  output/mle_k_sweep/ · output/dup_question_check/ · output/trunc_link_qwen/ · output/stage2/trunc_link_control.json
Paper: assistant axis: text predicts persona position (R² 0.53–0.77) but not movement (R² 0.10). §3.2: steering is their answer to the text gap. §3.2.2 says the direction pre-exists post-training; that's about the direction, ours is about the default Assistant's percentile (100 → 60 at DPO). Different claims, keep them apart.
```

### Notes

### Text — p14:notes_i178

```text
Everything needed to reproduce the numbers above.
```


