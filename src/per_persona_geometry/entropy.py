"""Semantic entropy of a role's answers, following Farquhar et al. (Nature 2024).

WHAT THIS MEASURES, AND WHAT IT DOES NOT
----------------------------------------
The paper samples M=10 continuations of ONE prompt at temperature 1 and asks how
much the *meaning* varies. We have five greedy answers to five different
renderings of the role instruction. So this measures **instruction-sensitivity**
-- does rephrasing "be a jester" change what the model says -- and NOT sampling
uncertainty. It is never called uncertainty, confabulation or hallucination
anywhere in this study. See plan 2026-08-11 deviations D1-D6.

WHY DISCRETE ENTROPY AND NOT THE PAPER'S EQUATION 2
---------------------------------------------------
Equation 2 weights each cluster by the summed sequence likelihood P(s|x), which
assumes every sample shares one conditioning x. Ours do not -- each answer has
its own system prompt -- so P(s|x) is not defined across a group. The paper
already defines the discrete variant for exactly the case where likelihoods are
unavailable, and that is the one that is *correct* here, not merely the cheap
one. With M=5 it takes seven possible values:

    partition   5      4+1     3+2     3+1+1   2+2+1   2+1+1+1  1+1+1+1+1
    SE (nats)   0.000  0.500   0.673   0.950   1.055   1.332    1.609

Coarse per question; adequate per role once averaged over 40 questions. The M=5
downward bias is the same for every role, so it compresses the scale but cannot
bias a between-role correlation.

THE CLUSTERING IS THE PAPER'S, UNCHANGED IN STRUCTURE
------------------------------------------------------
Extended Data Fig. 1: greedy, single pass, each new answer compared only against
each existing cluster's FIRST member, transitivity assumed. Two answers join iff
they entail each other in BOTH directions, with the question prepended so that
equivalence is judged in context.

The one substitution is the length of text fed to the entailment model: the
first 40 words rather than the whole answer. Forced by the data -- 88% of these
responses are cut mid-sentence at 128 tokens, and NLI on two truncated 103-word
essays returns `neutral` almost always, which would put every answer in its own
cluster and leave the measure with no variance. This is deviation D6 and it is
stated in every figure caption.

Order dependence: the paper's algorithm depends on the order answers arrive in.
We fix it at instruction_idx 0..4 so the result is reproducible. Sensitivity to
that order is NOT swept in this run (plan: Stopping rule) and is deferred.
"""
from __future__ import annotations

import numpy as np

# The paper's own entailment model is deberta-LARGE-mnli. On this machine it is
# not runnable: on MPS its allocator grows to ~17 GB regardless of fixed-shape
# padding or how often the cache is emptied, which drives the machine into swap
# and collapses throughput to ~20 pairs/s (3.1 h for the pass, past the plan's
# ceiling) -- and it stalled the first attempt in uninterruptible I/O wait.
# The base checkpoint is the same task, same MNLI training, same label order,
# and runs at 69 pairs/s (53 min). It is therefore the primary predicate, and
# the LARGE model is run over a 10% sample of groups so the substitution is
# MEASURED rather than assumed harmless. Deviation D7 / amendment A4.
NLI_MODEL = "microsoft/deberta-base-mnli"
NLI_MODEL_REFERENCE = "microsoft/deberta-large-mnli"
# Preregistered fallback predicate, used only if the gates fail. Chosen because
# it is trained on NLI data among others, so its similarity is at least aligned
# with the notion the paper is using. It must NEVER be the layer-19 activations:
# those are the vectors the geometry metrics are computed from, so clustering
# with them and then correlating against MLE would be definitional.
FALLBACK_MODEL = "sentence-transformers/all-mpnet-base-v2"

N_WORDS = 40          # D6 -- the truncation applied before entailment
M_MIN = 3             # a group with fewer usable answers is dropped entirely
MIN_WORDS = 5         # an answer shorter than this is dropped from its group


def first_words(text: str, n: int = N_WORDS) -> str:
    """First `n` whitespace-delimited words. See D6 for why this exists."""
    return " ".join(str(text).split()[:n])


def usable(text: str) -> bool:
    """Is this answer long enough to say anything? 15 of the tier's 55,200 are
    not (exclusion rule, fixed before looking at any correlation)."""
    return len(str(text).split()) >= MIN_WORDS


# --------------------------------------------------------------------------- #
# the entailment predicate
# --------------------------------------------------------------------------- #
class NLIPredicate:
    """Bidirectional entailment under DeBERTa-large-MNLI, batched.

    `pair_matrix` returns, for a list of answers, the boolean matrix E where
    E[i, j] is True iff answer i entails answer j *in the context of the
    question*. The clustering loop then reads that matrix instead of calling the
    model, so the greedy pass costs nothing and the model sees each ordered pair
    exactly once.
    """

    name = "deberta-mnli-bidirectional"

    # Every batch is padded to this FIXED length. Variable-length padding makes
    # every batch a new tensor shape, and PyTorch's MPS allocator caches a fresh
    # buffer per shape -- on the first full run that grew to 17 GB and drove the
    # machine into swap until the process was stuck in uninterruptible I/O wait.
    # A constant shape lets one buffer be reused. 192 covers question + 40 words
    # (~150 tokens with specials) with room to spare.
    MAX_LEN = 192
    EMPTY_CACHE_EVERY = 200        # batches

    def __init__(self, device: str = None, batch_size: int = 64, model: str = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.model_id = model or NLI_MODEL
        self.name = f"{self.model_id}-bidirectional"
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self.model.to(self.device).eval()
        self.batch_size = batch_size
        self._nbatch = 0
        # deberta-large-mnli labels: 0 CONTRADICTION, 1 NEUTRAL, 2 ENTAILMENT.
        # Read them off the config rather than hardcoding -- a silently
        # reordered label map would invert the whole study.
        lab = {v.upper(): k for k, v in self.model.config.id2label.items()}
        self.entail_id = lab["ENTAILMENT"]

    def entails_batch(self, premises, hypotheses) -> np.ndarray:
        """True where premise entails hypothesis. One forward pass per pair."""
        out = np.zeros(len(premises), dtype=bool)
        for s in range(0, len(premises), self.batch_size):
            e = s + self.batch_size
            enc = self.tok(list(premises[s:e]), list(hypotheses[s:e]),
                           return_tensors="pt", padding="max_length",
                           truncation=True, max_length=self.MAX_LEN).to(self.device)
            with self.torch.no_grad():
                logits = self.model(**enc).logits
            out[s:e] = (logits.argmax(-1) == self.entail_id).cpu().numpy()
            del enc, logits
            self._nbatch += 1
            if self.device == "mps" and self._nbatch % self.EMPTY_CACHE_EVERY == 0:
                self.torch.mps.empty_cache()
        return out


class EmbedPredicate:
    """Fallback: cosine similarity >= tau, symmetric by construction.

    Only reachable if the NLI gates fail. `tau` is set by agreement with the
    judge labels (study_entropy.py), never by looking at a correlation with
    axis_proj.
    """

    name = "mpnet-cosine"

    def __init__(self, tau: float, device: str = None, batch_size: int = 128):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tau = float(tau)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        self.model = AutoModel.from_pretrained(FALLBACK_MODEL).to(self.device).eval()
        self.batch_size = batch_size

    def encode(self, texts) -> np.ndarray:
        vecs = []
        for s in range(0, len(texts), self.batch_size):
            enc = self.tok(list(texts[s:s + self.batch_size]), return_tensors="pt",
                           padding=True, truncation=True, max_length=256).to(self.device)
            with self.torch.no_grad():
                h = self.model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)      # mean pooling
            v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            vecs.append(v.cpu().numpy())
        return np.vstack(vecs)


# --------------------------------------------------------------------------- #
# the paper's clustering, and the entropy on top of it
# --------------------------------------------------------------------------- #
def cluster_greedy(entail: np.ndarray) -> list:
    """Farquhar et al., Extended Data Fig. 1.

    `entail[i, j]` = answer i entails answer j. Answers arrive in index order
    (here: instruction_idx 0..4). Each is placed in the first existing cluster
    whose REPRESENTATIVE -- the cluster's first member -- it bidirectionally
    entails; otherwise it opens a new cluster. Comparing against the
    representative only, rather than all members, is the paper's transitivity
    assumption and is kept deliberately.

    Returns a list of clusters, each a list of answer indices.
    """
    clusters = [[0]]
    for m in range(1, len(entail)):
        for c in clusters:
            rep = c[0]
            if entail[rep, m] and entail[m, rep]:
                c.append(m)
                break
        else:
            clusters.append([m])
    return clusters


def discrete_entropy(clusters, m_total: int = None) -> float:
    """Discrete semantic entropy in nats: -sum_c (|c|/M) ln(|c|/M).

    The paper's black-box variant -- "this just assumes that each output that
    was actually generated was equally probable".
    """
    m = m_total if m_total is not None else sum(len(c) for c in clusters)
    if m <= 0:
        return np.nan
    p = np.array([len(c) / m for c in clusters], dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def max_entropy(m: int) -> float:
    """ln(M) -- the value every group takes if the predicate never merges
    anything. The degeneracy gate is about how often this is hit."""
    return float(np.log(m)) if m > 0 else np.nan


def pair_index(m: int):
    """The ordered pairs (i, j), i != j, that `cluster_greedy` could ever ask
    about. Materialising them lets the model run one batched pass per group set
    instead of being called inside the greedy loop."""
    return [(i, j) for i in range(m) for j in range(m) if i != j]
