# Survey protocol: assumptions and methods for testing curved representations (05/11/2026)

Fixed before the search. Owner request: go beyond the first 11 papers.

## Questions

1. What do researchers assume when they say a representation forms a manifold, and is curved?
2. Which assumptions are stated, which are tacit, and which have been tested or challenged?
3. Which methods test curved vs flat vs clustered vs noise, with which nulls?
4. How is geometry compared across training checkpoints?

## Strands

| ID | Strand | Anchors to start from |
|---|---|---|
| S1 | Statistics of the manifold hypothesis | Fefferman et al. 2016; Genovese et al.; Aamari and Levrard; Lim et al. 2023; Robinson et al. 2025 |
| S2 | Neural manifolds in neuroscience | Cunningham and Yu 2014; Gallego et al. 2017; Elsayed and Cunningham 2017; Chaudhuri et al. 2019; Jazayeri and Ostojic 2021; Chung et al. |
| S3 | Geometry of deep network representations | Ansuini et al. 2019; Cohen et al. 2020; Hénaff et al. 2019; Hosseini and Fedorenko 2023; Valeriani et al. 2023; Brown et al. 2023 |
| S4 | Concept geometry in LLMs | Park et al. 2024; Engels et al. 2024; Modell et al. 2025; Gurnee et al. 2025; Lu et al. 2026; Xu et al. 2026 |
| S5 | Nonlinear model fitting and its validation | Lawrence 2005; Titsias and Lawrence 2010; Hastie and Stuetzle 1989; Owen and Perry 2009; Chari and Pachter 2023; Ghahramani and Hinton (mixtures of factor analysers) |
| S6 | Geometry across training and across models | Li et al. 2025; Kornblith et al. 2019; Ding et al. 2021; Davari et al. 2023; Kriegeskorte et al. 2008; Saxe et al. 2019 |

## Search

- Sources: arXiv, Semantic Scholar API, Google Scholar through web search, journal pages.
- Per strand: at least 6 distinct queries, plus forward and backward citation steps from each anchor.
- Include: a paper that states, tests or challenges an assumption, or gives a test or null for shape.
- Exclude: pure applications with no method or assumption content; blog posts unless they are the primary source.
- Prefer reviews, highly cited work, and critiques. Search for critiques on purpose, with words such as artifact, spurious, null, pitfall, fails, byproduct.

## Reading and records

- Owner rule (05/11/2026): fetch each paper with curl and read the whole text, appendices included. No summaries, snippets or abstracts in place of a full read.
- Record the read level per paper: `full`, `main text`, or `abstract`. Use the lower levels only when curl cannot get the full text, and log why.
- Only `full` and `main text` papers count toward saturation.
- Record for each paper: assumptions (stated or tacit), test or null, what it challenges, section numbers.
- Separate what the paper states from the reader's inference.

## Stop rule

A strand stops when the last 5 papers read at `full` or `main text` add no new assumption and no new test family, or at 25 papers read, whichever comes first. Each strand reports which applied. A strand that stops at 25 is not saturated, and the report says so.

## Limits to report

No search proves completeness. The survey claims coverage only of what this protocol reached. English sources. Results are exploratory, not external claims.
