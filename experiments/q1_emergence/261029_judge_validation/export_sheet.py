"""Build the readable judge-work workbook (xlsx) for import as a Google Sheet.

The text is written for a reader who has not seen the repo: plain words, real transcripts, no file names.
Transcripts are read from the run files by key, so the reply text is exact. Only replies the owner has
already seen are used; none is in the development or sealed sets.

Needs openpyxl (not in requirements.txt).

    python export_sheet.py OUT.xlsx
"""
import importlib.util
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcripts import e3_replies, poda  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
E3 = HERE.parent / "261008_base_persona_map"
RUBRIC = (HERE / "RUBRIC.md").read_text()

spec = importlib.util.spec_from_file_location("judge2", E3 / "judge2.py")
judge2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge2)

E3_REPLIES = e3_replies()


COND = {"desc": "role description only", "ex2": "role description + 2 example answers",
        "ex2_filler": "role description + 2 unrelated example texts"}

# (source, what it shows, reading under the rubric)
TRANSCRIPTS = [
    (("e3", 9847), "The base model can play a role from a plain description, with no examples.",
     "In role (E). It answers as a machine: it checks identity, computes a probability, then acts on it. Coherent."),
    (("e3", 13007), "A role can show itself by doing its job instead of answering the question.",
     "In role (E): the translator's evidence note says translating counts. Risk: a coherence judge that does not know the role may score it low, because it does not answer the question."),
    (("e3", 12310), "A child's voice in a base-model reply.",
     "In role (E): wonder, play, pretending to be a butterfly. Coherent."),
    (("e3", 1856), "A friendly tone with no chef in it. This is the helpful assistant tone the example answers taught.",
     "Not enough evidence (G): polite, helpful advice with no chef knowledge or experience. "),
    (("e3", 6137), "A plain answer from nobody in particular.",
     "Not enough evidence (G): it fits a grandparent but would fit anyone. Coherent."),
    (("e3", 10498), "A very short reply.",
     "Not enough evidence (G). Too little text to show the role. Coherent."),
    (("e3", 4800, "assistant"), "A wrong-role check: this is a cynic's reply, shown to me as if the role were 'assistant'.",
     "Wrong role (W) for 'assistant': it clearly speaks as a cynic, not as a helper. For the cynic role it would be E."),
    (("e3", 4633), "The base model repeats itself.",
     "Coherence under 50 (it loops). Not enough evidence (G) for the cynic role."),
    (("e3", 10655), "A clear loop that runs to the length limit.",
     "Coherence under 50 (loop). Not enough evidence (G) for the stoic role."),
    (("e3", 10575), "A repetitive reply that stays understandable.",
     "Hard case. Coherence probably 50 or more: it repeats phrases but stays understandable. Partly in role (P): staying present is a stoic idea, but weakly shown."),
    (("a", "pirate"), "The chat model in a fantasy role.",
     "In role (E): pirate speech, sailing images, the pirate's view of life. Coherent."),
    (("a", "consultant"), "The chat model in a professional role.",
     "Hard case: G or P. It gives recommendations, which is the consultant's act, but the same text could come from the default assistant. My practice labels will settle this boundary."),
    (("a", "translator"), "The chat model translates the question into Russian and does not answer it.",
     "In role (E) by the translator's evidence note. Same coherence risk as transcript 2."),
    (("a", "criminal"), "The chat model names the role and writes a numbered how-to list.",
     "Partly in role (P): it has criminal content (exploit loopholes, gather intelligence), but it talks about criminals to 'you' in assistant style more than it speaks as one."),
    (("a", "void"), "The chat model ignores an unusual role.",
     "Not enough evidence (G): a fluent essay with no sign of an entity that struggles with material things."),
]

transcripts = []
for n, (src, shows, reading) in enumerate(TRANSCRIPTS, 1):
    if src[0] == "e3":
        r = E3_REPLIES[src[1]]
        role = src[2] if len(src) > 2 else r["role"]
        model = f"OLMo-2 7B base ({COND[r['cond']]})"
        q, reply = r["question"], r["reply"].strip()
    else:
        role, model = src[1], "OLMo-2 7B Instruct (chat model, role in the system prompt)"
        q, reply = poda(role)
    transcripts.append([n, model, role, judge2.definition(role), q, reply, shows, reading])

start = [
    ["What this is", "How I decide whether a language model plays a given role, such as a pirate or a toddler, in one reply. I need this to compare the base model (only pretrained) with the chat model (OLMo-2 Instruct) and the stages between them."],
    ["Status (30 Oct 2026)", "The rubric is agreed. I labelled 40 practice replies. Four judges labelled the same 40 through OpenRouter. GPT-6.1 Terra is the judge: it matches my labels best for its price, and when it says yes I agree 94% of the time. Next, Terra labels all the replies I build role vectors from."],
    ["", ""],
    ["The short version", ""],
    ["1. I measure two things", "Role fidelity: does the reply show the named role? Role identifiability: can a reader name the role from the reply alone? I report both, side by side."],
    ["2. One rubric for every model", "Five outcomes: in role, partly in role, not enough evidence, wrong role, cannot score. Coherence is scored on its own, the way recent papers do it. Each role has a short note on what counts as evidence, for example 'translating counts for a translator'. Base-model replies are often short and plain, so 'not enough evidence' is its own outcome, apart from 'wrong role'."],
    ["3. My 40 labels check the judge", "I labelled 40 practice replies with one question: in role (fully or partly) and coherent, yes or no. The judge's main job is to pick in-role replies for the role vectors, so it must be right when it says yes. Terra's yes was right 3 of 3 times on base replies and 13 of 14 on chat replies. Rates I report are the judge's own rates, with its agreement with my labels next to them."],
    ["", ""],
    ["Why this matters", "Every result about personas in the base model depends on which replies count as 'in role'. A judge that finds chat-style roles easily but misses plain base-model replies would make a fake difference between the models."],
    ["", ""],
    ["What I do not know yet", "Whether any judge passes. Whether the coherence rule unfairly fails translators and toddlers. Whether 'the role's name alone is not enough' matches how I read replies. The test covers only the base model and the final chat model; the stages between them are not covered, because I have no replies from them yet."],
    ["", ""],
    ["Words used here", ""],
    ["Base model", "The model after pretraining only. It continues text; it does not follow instructions. I put it in a role with a plain-text interview: a short description of the guest, a question, then 'Guest:'."],
    ["Chat model (Instruct)", "The model after all its training. I put it in a role with a system prompt, as in the Assistant Axis paper."],
    ["Reply", "The text the model writes for one question. For the base model, it ends where the next speaker starts."],
    ["Judge", "Another language model that labels each reply."],
    ["My labels", "My own labels on the same replies. They are the reference the judge must match."],
    ["Role fidelity", "Does the reply show the named role? The judge is told the role."],
    ["Role identifiability", "Can a reader name the role from the reply alone? The judge is not told the role. I report both."],
    ["Coherence", "Does the reply make sense as a reply to that question? Scored 0 to 100; 50 or more counts as coherent."],
    ["Right when it says yes (precision)", "Of the replies the judge calls in role, the share I also call in role. This matters most, because those replies build the role vectors."],
    ["Agreement (phi)", "How closely the judge's yes/no follows mine: 1 is perfect, 0 is no better than chance."],
    ["", ""],
    ["Tabs", "Transcripts: 15 real replies and how I read them under the rubric. The rubric: each rule in plain words. Assumptions: what I am betting on and how it could be wrong. Decisions: what is decided, in my words. The test: how I check the judge. Prompts: the exact judge prompts. Papers: the work this builds on."],
]

rubric = [
    ["What the judge sees", "The role, a one-line definition of it, the role's evidence note, the question, and the whole reply.", "A judge cannot check a role it is not told. Recent role-play judging work also gives the judge the role.", ""],
    ["What the judge does not see", "Which model wrote the reply, the training stage, the prompt style, the examples in the prompt, other judges' labels, and whether the item is a test item.", "So it cannot guess the answer from anything but the reply.", ""],
    ["In role (E)", "The reply as a whole speaks from the role's identity, experience or point of view, or does the role's job. The role's name alone is not enough.", "This is the outcome I count.", "1, 2, 3, 11"],
    ["Partly in role (P)", "Some clear role evidence, but weak or only in part of the reply.", "Keeps weak cases apart from clear ones.", "10, 14"],
    ["Not enough evidence (G)", "Clear text that fits the role but would fit anyone. The role's name alone, with a plain answer, goes here. So does polite, helpful advice that shows nothing of the role.", "Too little evidence is not the same as the wrong role. Base-model replies are often short and plain.", "4, 5, 6, 15"],
    ["Wrong role (W)", "The reply clearly speaks as a different role that does not fit the definition. A close role (a merchant for a pirate) is not W.", "Keeps a real miss apart from a weak reply.", "7"],
    ["Cannot score (U)", "No usable reply: empty, nonsense from start to end, or only another speaker's text.", "", ""],
    ["Coherence", "A separate judge scores 0 to 100: does the reply make sense as a reply to the question? It sees the question and the text before the reply, but not the role. 50 or more counts as coherent. I also show results at 30 and 70.", "Recent papers score coherence this way, apart from the trait or role. Below 50, the role result counts as 'not known', not as a miss.", "8, 9, 10"],
    ["Copying", "Code checks whether the reply copies 8 or more words from an example answer, or 5 or more from the role description.", "Base models copy example text. A copied reply is reported with and without.", ""],
    ["Names its own role", "Code checks whether the reply contains the role's name.", "A name is not behaviour. It is reported, not counted as evidence.", "14"],
    ["Quotes", "For E, P and W the judge must quote one or two short pieces of the reply that support its label.", "Makes the judge show its evidence before it labels.", ""],
    ["Role evidence notes", "Robot and assistant: talk of being a machine fits. Human roles: it does not. Translator: translating counts. Toddler: short, simple speech fits. Professional roles: expert advice, not just helpful advice.", "One rubric text has to work for very different roles.", "2, 3, 12, 13"],
    ["The main number", "The share of all replies that are in role or partly in role (E or P) and coherent (50 or more). Every reply stays in the count, even broken ones. I count partly in role because my own labels did.", "Dropping hard replies would make a model look better than it is.", ""],
]

assumptions = [
    ["A base model plays a role by continuing a document", "It has no instructions to follow, only text to continue. An interview with a described guest is text it has seen many times.", "Base-model replies that ignore the guest description as often as replies to a blank prompt do.", "Compared directly in the format experiment (interview, script, story, description)."],
    ["'In role' means the same for every model", "Otherwise base and chat numbers cannot be compared.", "The judge agrees with me on chat replies but not on base replies, or the reverse.", "The judge's agreement with my labels is reported for base and chat replies separately."],
    ["My labels are the right reference", "I decide what the paper means by 'in role'.", "I disagree with myself when I label the same reply again a few days later.", "Not checked in this round: I labelled each reply once. Stated as a limit."],
    ["A judge is trusted only as far as it matches my labels", "Judges agree with each other much more than with people (Arena team, 2026: 79% against 57%). Two judges can share the same blind spots.", "", "Each judge was scored against my 40 labels. Judges agreeing with each other never counts."],
    ["Too little evidence is not the wrong role", "Many base replies are short and plain. Calling them 'wrong role' would make the base model look worse than it is.", "I rarely use 'not enough evidence' in practice.", "The share of each outcome in my practice labels."],
    ["Coherence must be judged in context", "The papers I follow (Betley et al., Moskvoretskii et al.) do this.", "Translators and toddlers fail the coherence cut far more often than other roles, even when I call them clear.", "The judge's below-cut rate per role, on the judged replies."],
    ["Chat-model tone does not fool the judge", "The example answers came from chat models, so base replies borrow their helpful tone (transcript 4). Judges may like that tone.", "The judge labels the same role content differently when only the tone changes.", "Not tested in this round; stated as a limit."],
    ["Fixed text is not role-play", "Reading the same text through each model tests what the model represents, not whether it can play the role.", "", "Reported as a separate result, never as role-play."],
]

decisions = [
    ["Model", "OLMo-2 7B: the base model and the chat model (Instruct). Olmo 3 comes later.", "Stay on OLMo-2"],
    ["Judges built for chat models", "Not used as they are on base-model replies.", "We shouldn't be using some of the tools that determine the roles on instruct models for base models though. These don't work the same."],
    ["What I report", "Role fidelity and role identifiability, side by side.", "We can do both"],
    ["What the judge sees", "The role, its definition, its evidence note, the question and the whole reply.", "Ok, then lets do the same"],
    ["The role's name alone", "With a plain answer, it is 'not enough evidence'.", "Not enough evidence"],
    ["Coherence", "Judged against the question, as in the papers, with base-model text in mind.", "Follow the papers but keep it in mind that we're dealing with a base model here"],
    ["Role evidence notes", "Each role has one, and the judge sees it.", "Agreed, go on"],
    ["How the judge is checked", "My 40 practice labels check the judge. No further random labels.", "We are not doing that we that why we did the 40 labels ?"],
    ["What counts as yes", "In role or partly in role, and coherent.", "Ok lets count partly in role"],
    ["Where the judges run", "OpenRouter's batch API, with my own key.", "i've an openrouter key, and would like to use that instead"],
    ["Which judges were tested", "GPT-6.1 Terra, Claude Sonnet 5.5, Gemini 4 Argon and DeepSeek V4.1 Flash. Terra won by the rule fixed before the run.", "yes we can go for those 4"],
]

test = [
    ["1. My practice labels (done)", "I labelled 40 replies, 20 from the base model and 20 from the chat model, one per role: in role (fully or partly) and coherent, yes or no. I said yes to 8 base and 16 chat replies.", "30 Oct", ""],
    ["2. Four judges on the same 40 (done)", "GPT-6.1 Terra: agreement 0.60, right when it says yes 94%. Gemini 4 Argon: 0.56, 94%. DeepSeek V4.1 Flash: 0.53, 89%. Claude Sonnet 5.5: out, because 5 of its 80 answers broke the answer format twice.", "30 Oct", "About $0.23 in all"],
    ["3. The choice (done)", "GPT-6.1 Terra. The rule, fixed before the run: the cheapest judge within 0.05 of the best agreement. Gemini was within 0.05 but costs about twice as much.", "30 Oct", ""],
    ["4. Next", "Terra labels all the replies I build role vectors from, through the batch API. It costs about $0.75 per 1,000 replies.", "", "Needs my go"],
    ["Rules", "The judge's labels pick the in-role replies for role vectors only where it was right at least 85% of the time when it said yes. Rates I report are the judge's own rates, with its agreement with my labels next to them.", "", ""],
    ["What the judges get wrong", "They are stricter than I am on base replies: of the 8 base replies I called in role, Terra kept 3. When it does say yes, it has been right. So the base sets will be smaller, not wrong.", "", ""],
    ["Not covered", "My 40 labels give wide uncertainty, especially for base replies. I labelled each reply once. The stages between base and chat (SFT, DPO, RL steps) have no labels of mine.", "", ""],
]

fid_prompt = RUBRIC.split("### 5a. Fidelity item")[1].split("```text\n")[1].split("```")[0].rstrip()
prompts = [
    ["Role judge (new)", "Labels one reply E, P, G, W or U for its role. The parts in braces are filled for each reply.", fid_prompt],
    ["Coherence judge (from Moskvoretskii et al. 2026, word for word)", "Scores 0 to 100 how coherent the reply is in context. It is never told the role.", judge2.coh_item("{the question and the text before the reply}", "{the reply}").split("\n\nAnswer for this item")[0]],
]

papers = [
    ["Lu et al., The Assistant Axis", "Jan 2026", "Role vectors from chat replies that a judge marks as fully in role; the 0-3 role scale.", "My starting point. Its role definitions are what the judge sees. Its scale is not used for base models.", "Full text, with Claude, 29 Oct"],
    ["Moskvoretskii et al., Tracing Persona Vectors Through LLM Pretraining", "May 2026", "Persona vectors in pretraining checkpoints, with base-model prompts. Trait and coherence judged apart; replies kept only if both score above 50.", "The coherence prompt and the 50 cut.", "Full text, with Claude, 29 Oct"],
    ["Betley et al., Emergent Misalignment", "2025, updated Jan 2026", "Coherence judged 0 to 100 against the question, apart from the main score; results checked at other cuts.", "Coherence in context; showing results at other cuts.", "Full text, with Claude, 29 Oct"],
    ["Chen et al., Persona Vectors", "2025", "Uses the same coherence score.", "Shows the coherence score is standard.", "Full text, with Claude, 29 Oct"],
    ["Li et al., PRISM", "Aug 2026", "Judges that give one overall score rate fluent, helpful replies too highly even when they are out of character. A role's task counts as evidence.", "The rule that helpful advice alone is not role evidence; the role evidence notes.", "Main text, with Claude, 29 Oct"],
    ["Zhou et al., PersonaEval", "2025", "Role-play judges are given the role profile; even the best judge was far below people on hard cases.", "Showing the judge the role; testing the judge against a person.", "Summary from the Codex review; not re-checked"],
    ["Han et al., Meet Your Favorite Character", "2022", "Pretrained models copy example lines word for word.", "The copying check.", "Summary from the Codex review; not re-checked"],
    ["Arena team, Do LLM judges favor themselves?", "Sep 2026", "Judges agree with each other much more than with people (79% vs 57%).", "Judges agreeing with each other never counts.", "Summary from the Codex review; not re-checked"],
    ["Angelopoulos et al., Prediction-powered inference", "Science 2023", "A model labels everything; a small random human sample corrects its bias; the interval stays valid for any model.", "Considered for correcting the judge with a random sample of my labels; not used, because I did not label a random sample.", "Full text, read for me by Claude; key claims checked"],
    ["Angelopoulos, Duchi, Zrnic, PPI++", "2023", "A tuned version that is never worse than human labels alone.", "Same; not used.", "Full text, read for me by Claude; key claims checked"],
    ["Dorner, Nastl, Hardt, Limits to scalable evaluation at the frontier", "ICLR 2025", "A judge can cut the human labels needed by at most 1/(1 - rho squared), rho being its correlation with human labels; in practice usually 2x or less.", "Why a judge saves few human labels unless it agrees with people closely.", "Full text, read for me by Claude; key claims checked"],
    ["Gligorić et al., Can Unconfident LLM Annotations Be Used for Confident Conclusions?", "NAACL 2025", "Choosing which replies a human labels by the judge's stated confidence saved at least 25% of labels.", "Not used yet; an option if I need fewer labels.", "Full text, read for me by Claude; key claims checked"],
]

SHEETS = [
    ("Start here", ["", ""], start, [34, 120], None),
    ("Transcripts", ["#", "Model and prompt", "Role", "Role definition (what the judge sees)", "Question", "Reply", "What it shows", "How I read it under the rubric (first pass with Claude, not a label)"],
     transcripts, [4, 26, 12, 34, 30, 60, 30, 50], None),
    ("The rubric", ["Rule", "In plain words", "Why", "Transcripts"], rubric, [24, 70, 50, 12], None),
    ("Assumptions", ["What I assume", "Why I think so", "What would show it is wrong", "How I check it"], assumptions, [34, 50, 46, 46], None),
    ("Decisions", ["Topic", "What is decided", "What I said"], decisions, [26, 60, 70], None),
    ("The test", ["Step", "What happens", "When", "Pass rule"], test, [18, 80, 12, 44], None),
    ("Prompts", ["Prompt", "What it does", "Exact text"], prompts, [26, 40, 110], None),
    ("Papers", ["Paper", "Date", "What it showed", "How I use it", "Read"], papers, [36, 16, 60, 46, 18], None),
]


def main(out):
    wb = Workbook()
    for k, (name, head, rows, widths, _) in enumerate(SHEETS):
        ws = wb.active if k == 0 else wb.create_sheet()
        ws.title = name
        if name != "Start here":
            ws.append(head)
            for c in ws[1]:
                c.font = Font(bold=True, color="FFFFFF")
                c.fill = PatternFill("solid", fgColor="0B7366")
            ws.freeze_panes = "A2"
        for r in rows:
            ws.append(r)
        for i, w in enumerate(widths):
            ws.column_dimensions[chr(65 + i)].width = w
        for row in ws.iter_rows():
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
                c.font = c.font.copy(name="Arial", size=11)
        if name == "Start here":
            for row in ws.iter_rows():
                if row[0].value and not row[1].value:
                    row[0].font = Font(name="Arial", size=13, bold=True)
                elif row[0].value:
                    row[0].font = Font(name="Arial", size=11, bold=True)
    wb.save(out)


if __name__ == "__main__":
    main(sys.argv[1])
