# Deck outline v2 (13 Nov): the story, for a reader new to the topic

Brendon's 8-part structure, one row per slide. Sources are the vault paper notes (`Reference/2026-10-23-paper-audit/papers/`, `Reference/2026-11-05 Literature stages/notes/`), the weekly log, and draft v5. Every result number stays from v5.

| # | Part | Slide title (the finding) | What is on it | Source |
|---|---|---|---|---|
| 1 | Opener | Update 11/13: when does the Assistant become privileged? | Section header, 3 lines | v5 |
| 2 | 1. Field | Persona research rests on a small set of papers | **Paper mountain**: cards from the oldest foundations at the base to the newest work at the peak; colour = theme; the three scoop papers outlined | paper notes |
| 3 | 2. Latest research | Recent work: personas exist before chat training, form a map, and post-training picks the Assistant | Small table: claim, evidence, paper (Moskvoretskii et al.; Lu et al.; Chen et al.; Xu et al.; Ratnam and Africa; Zhao et al.) | P01, P02, P03, P10, P11, Persona Geology note |
| 4 | 3. Safety | Persona geometry gives handles on misalignment | Four rows: misalignment acts like a persona (Wang et al.); trait shifts can be monitored and prevented (Chen et al.); distance from the default persona predicts how far fine-tuned behaviour spreads (Zhao et al.); the Assistant's harmlessness preference shows on the user's turn (Nguyen) | P03, P06, P07, P10 |
| 5 | 4. Unknowns | No paper yet says at which training stage the Assistant becomes special | Open questions from the 5 Nov review: stage of privilege; own replies per stage; geometry beyond the words (Persona Geology's 94% bar); whether map, role play and default change together | 5 Nov literature review |
| 6 | 5. My question | I ask at which training stage the Assistant becomes privileged | Why (PSM predicts selection at post-training; nobody showed "this starts at this checkpoint"); how it helps (tells safety work which stage to audit); OLMo-2 releases every stage; the two meanings (position, spread); exact prompt | v5, 17 Sep meeting |
| 7 | 6. Pivot | Three concurrent works covered parts of my plan, so I narrowed it | Timeline: 28 Sep PersonaManifold (scoop found 2 Oct); Oct Persona Geology (I prepared a per-stage replication on 31 Oct, not run); 5 Nov Han Xiao (same stage question on Olmo 3 7B) | W40, W44, W46 logs |
| 8 | 7. What they say, what I test | Each concurrent result makes a claim I can test on OLMo-2 | Table: work, its claim, what I found so far, the test I run, how it helps if it holds | v5, paper notes |
| 9-14 | 8. Results | One finding per slide, figure first, terms defined in plain words, one "what this means for behaviour, geometry and safety" line | The six figure/table slides already built (base role play; words carry the role; same text; Olmo 3 default cloud; roles move together; RLVR) | v5 |
| 15 | 8. Results | What the results mean so far | Plain summary: behaviour vs geometry; possible safety use, marked as my reading | v5 + my reading |
| 16-17 | Close | Next steps; questions for Thomas | As now | v5 |
| B | Backup | Coin-flip, own-reply status, method check, judge check, words used | As now | v5 |

Open points for Brendon are in the chat.
