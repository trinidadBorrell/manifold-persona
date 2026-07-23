"""manifold-persona: geometry of persona/trait activations.

Builds on the Persona Vectors method (Chen, Arditi, Sleight, Evans, Lindsey,
arXiv:2507.21509) whose official code lives in the sibling ``persona_vectors``
repository. Prompt construction and activation extraction here are adapted
directly from that repo (see ``prompts.py`` and ``extract.py``).
"""

from .config import (
    PERSONA_VECTORS_DIR,
    TRAITS,
    MODEL_NAME,
    EMBEDDINGS_DIR,
    HF_TOKEN_PATH,
    HF_REPO_NAME,
    primary_layer,
)

__all__ = [
    "PERSONA_VECTORS_DIR",
    "TRAITS",
    "MODEL_NAME",
    "EMBEDDINGS_DIR",
    "HF_TOKEN_PATH",
    "HF_REPO_NAME",
    "primary_layer",
]
