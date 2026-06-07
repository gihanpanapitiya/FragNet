from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


# ── Requests ──────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    smiles: str
    prop_type: str = "Solubility"


class OptimizeRequest(BaseModel):
    smiles: str
    prop_type: str
    direction: str = "maximize"
    n_worst: int = 1
    max_candidates: int = 50
    top_k: int = 10
    frag_contribs: list[dict]
    seed_prediction: float
    locked_fragment_indices: list[int] = []
    library_source: str = "auto"           # "auto" | "chembl" | "reference"
    use_contribution_prior: bool = False   # sort replacements by mean/std contribution


class LLMSuggestRequest(BaseModel):
    smiles: str
    prop_type: str
    direction: str = "maximize"
    n_worst: int = 1
    n_suggestions: int = 8
    frag_contribs: list[dict]
    frag_atom_map: list[list[int]]
    seed_prediction: float
    locked_fragment_indices: list[int] = []


# ── Sub-objects ───────────────────────────────────────────────────────────────

class FragCentroid(BaseModel):
    fragment_index: int
    cx: float           # pixel x within the SVG viewBox
    cy: float           # pixel y within the SVG viewBox
    contribution: float
    atom_indices: list[int]


class CandidateResult(BaseModel):
    smiles: str
    prediction: float
    delta: float
    improvement: float
    mol_img_b64: str    # base64 PNG, 260×180


class LLMSuggestion(BaseModel):
    smiles: str
    rationale: str
    prediction: Optional[float] = None
    delta: Optional[float] = None
    improvement: Optional[float] = None
    mol_img_b64: str


# ── Responses ─────────────────────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    smiles: str
    prop_type: str
    prediction: float
    unit: str                       # "logS" | "logP"

    mol_svg_b64: str                # base64-encoded SVG; use as <img src="data:image/svg+xml;base64,...">
    mol_svg_width: int              # 600
    mol_svg_height: int             # 400
    fragment_centroids: list[FragCentroid]

    frag_atom_map: list[list[int]]

    # Contribution data (records from DataFrames)
    atom_contribs: list[dict]
    bond_contribs: list[dict]
    fbond_contribs: list[dict]
    frag_contribs: list[dict]
    frag_weights: list[dict]        # fragment attention weights
    connection_weights: list[dict]
    atoms_in_frags: dict[str, list] # str(frag_idx) → [atom indices]

    # Images: base64 PNG data-URIs (include the data:image/png;base64, prefix)
    img_atom_attn: str
    img_bond_attn: str
    img_frag_attn: str
    img_frag_highlight: str
    img_frag_attr: str


class OptimizeResponse(BaseModel):
    seed_smiles: str
    seed_prediction: float
    n_candidates_evaluated: int
    n_eligible_fragments: int
    worst_fragment_indices: list[int]
    candidates: list[CandidateResult]


class LLMSuggestResponse(BaseModel):
    suggestions: list[LLMSuggestion]
    n_scored: int
    n_improved: int
