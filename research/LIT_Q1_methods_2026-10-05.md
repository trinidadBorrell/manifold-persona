# Q1 methods: literature review, 2026-10-05

Question: how did other work measure persona structure in base models and across training stages, and how did they make base models play a persona?

Method: arXiv API searches with `curl`, then a full-text read of 20 papers from arXiv HTML. The full texts are in `output/lit_q1_2026-10-05/`. The keyword searches can miss relevant work. Some table numbers were lost in the text extraction.

## 1. Persona geometry in base models and across stages

| Paper | Models and stages | How the base got persona text | Judge | Finding |
|---|---|---|---|---|
| Assistant Axis, 2601.10387 (Jan 2026), Apx B.3 | Gemma 2 27B base vs instruct | Teacher forcing: base reads the instruct model's judge-filtered role replies | gpt-4.1-mini, 0–3 role score, on the instruct replies | Top 3 PCs match at cosine 0.93, 0.87, 0.83. The paper says the axes are "likely already present in base models". This test cannot show that the base forms roles by itself. |
| Tracing Persona Vectors, 2605.13329 (May 2026) | OLMo-3-7B, 17 pretraining checkpoints, plus SFT, DPO, RLVR; Apertus-8B | Own generations. The system prompt becomes a third-person character description ("Alex operates with intent to cause harm…"). Apx tests dialogue and narration formats. | gpt-4.1-mini; keep replies with trait > 50 and coherence > 50; 91% human agreement | Trait vectors form within 0.22% of pretraining and still steer the final model. Four traits only, not a role map. |
| Simulation to Enaction, 2605.25459 (May 2026) | OLMo-3-32B Base, SFT, DPO, RLVR; Llama-3.1-70B | Same texts run as assistant turn, user turn and no template | None (entropy) | Base: no self/other gap. SFT: gap only in the assistant turn. DPO: gap in all formats. |
| Introspective Awareness, 2603.21396 (Mar 2026), §3.3 | OLMo-3.1-32B Base, SFT, DPO, Instruct | Base prompt with "User:" / "Assistant:" lines | gpt-4.1-mini | DPO is the first stage with 0% false positives. |
| Harmlessness on the User Turn, 2609.23935 (Sep 2026) | Base/instruct pairs; OLMo-3 Base, SFT, DPO, RLVR | Same 400 fixed items on all models; base gets plain text | None (next-token probability) | Bias is near 0 in base and appears at DPO. |
| Transplanting a misalignment persona, 2607.04510 (Jul 2026) | Qwen2.5-32B base and instruct | Plain template with no chat tokens, the same for base and instruct | Qwen3-Next-80B | A probe reads the EM direction in base (AUC 0.89). Steering the base with it mostly breaks coherence. |
| Persona effects generalize, 2609.32758 (Sep 2026) | 7 base/post-trained pairs; OLMo-2, OLMo-3, Tulu-3 stages | Persona prompts, A/B first-token log odds, 4 few-shot examples | None | Training scales persona effects 0.76–4.96× and keeps their direction. Behaviour only. |

Persona Vectors (2507.21509), Persona Features (2506.19823), Data Attribution of EM (2608.11025) and Auditing with Persona Vectors (2607.13162) use only post-trained models, or a base-trained SAE applied to the chat model. They assume, but do not test, that persona features form in pretraining.

## 2. How to make a base model play a persona

None of these papers fed a chat template to a base model. Each one wrote the persona as a document type that the base model knows.

| Paper | Base prompt format | Fidelity check |
|---|---|---|
| Out of One, Many, 2209.06899 | First-person backstory, or `Interviewer: …\nMe: …` | Match to survey data; human raters |
| Anthology, 2407.06576 | `Question: Tell me about yourself.\nAnswer: <long first-person story>` | Match to Pew surveys. Says instruct models are "completely unsuitable" for this. |
| Emulate or Estimate, 2608.03044 | `INTERVIEWER: …\nPARTICIPANT: I am a Democrat.\nINTERVIEWER: <q>\nPARTICIPANT:` | LLM judge maps text to options, checked against 200 human labels (74% agreement) |
| Identity, Cooperation, 2601.16355 | Long generated life-story interview ("DeepBind") | Match to human game studies |
| Persona Mixture Models, 2609.22607 | Serialized dialogue or forum thread ending in the speaker's prefix | No LLM judge; perplexity, MAUVE, diversity |
| URIAL, 2312.01552 | Preamble + 3 `# Query:` / `# Answer:` examples | GPT-4 scores. The preamble names the model "an AI assistant (you)". |
| Shanahan 2305.16367; Andreas 2212.01681 | Script or author description | Theory. Short descriptions bind weakly; later text can overwrite the role. |

Judge warning: PersonaEval (2508.10014) reports that the best LLM judges identify the speaking role about 69% of the time, against 90.8% for humans. We read only the abstract.

## 3. What this means for our Q1 design

1. **E1 (fixed text) has a direct precedent:** Assistant Axis Apx B.3. It shows only that the base can *represent* role text. It does not show that the base forms roles by itself.
2. **E0 (base-native prompts) has candidate formats:** an interview turn with a first-person role line, a long first-person backstory, a third-person narration ("Alex"), or a forum or script dialogue.
3. **Do not use URIAL for the assistant axis.** Its preamble tells the base model that it is an AI assistant. That pushes the default toward the assistant and confounds the measure.
4. **Control the format:** run the same text as an assistant turn, a user turn and with no template, as 2605.25459 did.
5. **Validate the judge against human labels** before we trust role-play rates.
6. **Other stage results point to DPO,** not SFT, for self-related effects in OLMo-3 (2605.25459, 2603.21396, 2609.23935). These are different measures from ours. Do not use them as an expected result.

## Gap

In these searches, no paper ran a full role map (about 275 roles) across base, SFT, DPO and RL with both fixed text and base-native own generations. The keyword searches were short, so this is not a complete novelty check.
